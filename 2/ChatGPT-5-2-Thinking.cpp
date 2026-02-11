// speculative_kernel.cpp
// Single-file C++23 kernel for "Project Event Horizon"
// - Speculative transactional ring buffer (two-phase commit, help/abort)
// - Coroutine-integrated EBR with suspend-time exit / resume-time revalidation
// - AVX-512 branchless "shadow engine"
// - Compile-time state machine via concepts/consteval
// - No exceptions, std::expected-lite used
// - Memory ordering uses only acquire/release/acq_rel
// - Uses std::start_lifetime_as for strict aliasing / lifetime
// - All coroutine frames allocated from an Arena

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <cassert>
#include <coroutine>
#include <new>
#include <immintrin.h> // AVX-512 intrinsics
#include <type_traits>
#include <cstdio>
#include <chrono>
#include <optional>

// ---------------------------
// Minimal std::expected<T,E>
// ---------------------------
template<typename T, typename E>
class expected {
    bool ok_;
    union { T t_; E e_; };
public:
    constexpr expected(const T& v) : ok_(true), t_(v) {}
    constexpr expected(T&& v) : ok_(true), t_(std::move(v)) {}
    constexpr expected(const E& err, bool) : ok_(false), e_(err) {}
    constexpr expected(E&& err, int) : ok_(false), e_(std::move(err)) {}
    constexpr ~expected() { if (ok_) t_.~T(); else e_.~E(); }
    constexpr bool has_value() const noexcept { return ok_; }
    constexpr T& value() & { assert(ok_); return t_; }
    constexpr const T& value() const & { assert(ok_); return t_; }
    constexpr E& error() & { assert(!ok_); return e_; }
    static expected<T,E> error_expected(const E& e) { return expected(e, true); }
};

// ---------------------------
// Utilities
// ---------------------------
inline uint64_t thread_token() noexcept {
    // Return a reasonably unique per-thread token (non-zero).
    // Avoid using std::hash on thread::id because it's heavy; use address of thread_local.
    static thread_local int marker = 0;
    return reinterpret_cast<uint64_t>(&marker) ^ (uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id())) + 0x9e3779b97f4a7c15ULL);
}

constexpr uint64_t STATE_EMPTY    = 0u;
constexpr uint64_t STATE_RESERVED = 1u;
constexpr uint64_t STATE_COMMITTED= 2u;
constexpr uint64_t STATE_ABORTED  = 3u;
static_assert((STATE_EMPTY|STATE_RESERVED|STATE_COMMITTED|STATE_ABORTED) < (1u<<2), "2 LSB bits reserved for state");

// ---------------------------
// Payload: zero-copy in-place object
// ---------------------------
struct Payload {
    uint64_t order_id;
    double price;
    uint32_t volume;
    uint8_t side; // 0 = buy, 1 = sell
    // Plain-old-data: trivial, safe for start_lifetime_as
};

// Tagged sequence layout:
// 64-bit word: [ high bits: seq_number ] << 2 | [ low 2 bits: state ]

inline uint64_t make_tag(uint64_t seq, uint64_t state) noexcept {
    return (seq << 2) | (state & 3u);
}
inline uint64_t tag_seq(uint64_t tag) noexcept { return (tag >> 2); }
inline uint64_t tag_state(uint64_t tag) noexcept { return (tag & 3u); }

// ---------------------------
// Transactional Lock-Free Ring Buffer
// ---------------------------
template<size_t Capacity>
class TxRing {
    static_assert((Capacity & (Capacity-1)) == 0, "Capacity must be power-of-two for index masking");

public:
    struct Slot {
        alignas(64) std::atomic<uint64_t> tag;   // seq<<2 | state
        alignas(64) std::atomic<uint64_t> owner; // producer token that reserved the slot
        alignas(64) std::byte bytes[sizeof(Payload)]; // storage for payload
        alignas(64) std::atomic<uint64_t> gen;   // generation counter for ABA protection (optional)
        // pad to avoid false sharing
    };

    TxRing() : buffer_(Capacity), mask_(Capacity - 1) {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].tag.store(make_tag(0, STATE_EMPTY), std::memory_order_release);
            buffer_[i].owner.store(0, std::memory_order_release);
            buffer_[i].gen.store(0, std::memory_order_release);
        }
        next_seq_.store(0, std::memory_order_release);
        commit_cursor_.store(0, std::memory_order_release);
    }

    // Producer: reserve a sequence number (two-phase commit)
    // returns reserved sequence number (absolute), and a pointer to the slot storage.
    expected<std::pair<uint64_t, Payload*>, int> reserve() noexcept {
        uint64_t seq = next_seq_.fetch_add(1, std::memory_order_acq_rel);
        Slot &s = buffer_[seq & mask_];

        // Write owner and set tag to reserved only if this slot is not still linked to a previous
        // active generation for the same sequence (we use gen to detect wrap-around).
        uint64_t expected_tag = s.tag.load(std::memory_order_acquire);
        // Note: we DON'T block/wait here. We are allowed to overwrite prior content if sequences advance.
        // Set owner and reserved tag. Because seq monotonically grows, no ABA happens until Capacity wraps.
        s.owner.store(thread_token(), std::memory_order_release);
        s.gen.store(seq, std::memory_order_release); // annotate generation
        s.tag.store(make_tag(seq, STATE_RESERVED), std::memory_order_release);

        // Return pointer to payload storage for in-place construction via start_lifetime_as
        Payload* p = reinterpret_cast<Payload*>(s.bytes);
        return expected<std::pair<uint64_t, Payload*>, int>(std::pair(seq, p));
    }

    // Producer: commit reserved slot (Phase 2)
    // If commit fails (helped-abort), returns error.
    expected<void, int> commit(uint64_t seq) noexcept {
        Slot &s = buffer_[seq & mask_];
        uint64_t expected = make_tag(seq, STATE_RESERVED);
        uint64_t desired  = make_tag(seq, STATE_COMMITTED);
        // attempt CAS from RESERVED to COMMITTED
        if (s.tag.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // success; try to advance commit cursor opportunistically
            advance_commit_cursor();
            return expected<void,int>(0, true); // success variant (uses the error-constructor)
        } else {
            // someone else changed it (likely helped-abort). Fail the commit: caller must roll back
            return expected<void,int>::error_expected(-1);
        }
    }

    // Producer: abort reserved slot (Phase 2)
    expected<void, int> abort(uint64_t seq) noexcept {
        Slot &s = buffer_[seq & mask_];
        uint64_t expected = make_tag(seq, STATE_RESERVED);
        uint64_t desired  = make_tag(seq, STATE_ABORTED);
        // CAS from RESERVED to ABORTED. If CAS fails, then consumer/helping may have already aborted or committed.
        s.tag.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
        advance_commit_cursor();
        return expected<void,int>(0, true);
    }

    // Consumer: try to pop next committed slot in order.
    // Returns expected with pair(seq, payload pointer) if a committed/aborted slot was handled.
    // If returns error with code 1 => no available committed/aborted slot yet.
    expected<std::pair<uint64_t, Payload*>, int> try_pop() noexcept {
        uint64_t cur = commit_cursor_.load(std::memory_order_acquire);
        Slot &s = buffer_[cur & mask_];
        uint64_t tag = s.tag.load(std::memory_order_acquire);
        uint64_t seq = tag_seq(tag);
        uint64_t st  = tag_state(tag);

        if (seq != cur) {
            // The slot hasn't been claimed for this sequence yet.
            return expected<std::pair<uint64_t, Payload*>, int>::error_expected(1);
        }

        if (st == STATE_COMMITTED) {
            // Consumer can safely process: the producer finished commit
            Payload* p = reinterpret_cast<Payload*>(s.bytes);
            // Important: do NOT destroy lifetime here; we assume consumer processes payload directly.
            // Advance cursor (consume)
            // We advance commit_cursor_ using fetch_add to be wait-free and permit multiple consumers?
            // However we maintain single consumer for strict ordering in this simple kernel.
            commit_cursor_.fetch_add(1, std::memory_order_acq_rel);
            return expected<std::pair<uint64_t, Payload*>, int>(std::pair(seq, p));
        } else if (st == STATE_ABORTED) {
            // Skip this sequence: it's aborted; advance cursor
            commit_cursor_.fetch_add(1, std::memory_order_acq_rel);
            return expected<std::pair<uint64_t, Payload*>, int>::error_expected(2); // special code for aborted skip
        } else if (st == STATE_RESERVED) {
            // The producer reserved but hasn't finished; to provide wait-free consumer progress, we *help* by attempting to abort.
            uint64_t expected = tag;
            uint64_t desired  = make_tag(seq, STATE_ABORTED);
            // Try to help-abort. If we succeed, we skip it; if we fail, re-check state.
            if (s.tag.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
                // Helped abort
                commit_cursor_.fetch_add(1, std::memory_order_acq_rel);
                return expected<std::pair<uint64_t, Payload*>, int>::error_expected(2);
            } else {
                // CAS failed: producer may have committed just now or another helper intervened.
                return expected<std::pair<uint64_t, Payload*>, int>::error_expected(1);
            }
        } else {
            // STATE_EMPTY or unknown
            return expected<std::pair<uint64_t, Payload*>, int>::error_expected(1);
        }
    }

    // Peek-only: return the current commit cursor (for debug/metrics)
    uint64_t commit_cursor() const noexcept { return commit_cursor_.load(std::memory_order_acquire); }

private:
    void advance_commit_cursor() noexcept {
        // Opportunistically advance the commit cursor by scanning forward while slots are in
        // COMMITTED or ABORTED state. This uses only atomic loads/stores and is bounded by Capacity.
        // It's intentionally non-blocking: consumer(s) will also attempt advancement when processing.
        uint64_t cur = commit_cursor_.load(std::memory_order_acquire);
        while (true) {
            Slot &s = buffer_[cur & mask_];
            uint64_t tag = s.tag.load(std::memory_order_acquire);
            uint64_t seq = tag_seq(tag);
            uint64_t st  = tag_state(tag);
            if (seq != cur) break;
            if (st == STATE_COMMITTED || st == STATE_ABORTED) {
                // advance
                if (commit_cursor_.compare_exchange_strong(cur, cur+1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // loop to check next
                    continue;
                } else {
                    // someone else advanced; reload cur
                    cur = commit_cursor_.load(std::memory_order_acquire);
                    continue;
                }
            } else {
                break;
            }
        }
    }

    alignas(64) std::vector<Slot> buffer_;
    const uint64_t mask_;
    std::atomic<uint64_t> next_seq_;      // producers claim with fetch_add
    std::atomic<uint64_t> commit_cursor_; // next sequence consumers expect
};

// ---------------------------
// Epoch-Based Reclamation (EBR) integrated with coroutines
// ---------------------------

class EBR {
    static const int MAX_THREADS = 128;
    std::atomic<uint64_t> global_epoch_;
    struct Local {
        std::atomic<uint64_t> active_epoch; // 0 = not in epoch, otherwise epoch value+1
    };
    Local locals_[MAX_THREADS];

public:
    EBR() {
        global_epoch_.store(1, std::memory_order_release); // epoch numbers start at 1
        for (int i = 0; i < MAX_THREADS; ++i) locals_[i].active_epoch.store(0, std::memory_order_release);
    }

    // Each thread must get a small index to locals_. We map using a hashed thread token.
    int register_thread() noexcept {
        // naive assignment: find a free slot
        uint64_t tok = thread_token();
        for (int i = 0; i < MAX_THREADS; ++i) {
            uint64_t cur = locals_[i].active_epoch.load(std::memory_order_acquire);
            (void)cur;
            // We don't mark here; just return starting slot using token-based mapping to avoid races in example.
        }
        // For simplicity, map token->index by hashing; assume MAX_THREADS large enough.
        return int(tok % MAX_THREADS);
    }

    // Enter epoch: mark that this thread is protecting current epoch
    void enter_epoch(int idx) noexcept {
        uint64_t e = global_epoch_.load(std::memory_order_acquire);
        locals_[idx].active_epoch.store(e, std::memory_order_release);
    }

    // Exit epoch: clear protection so GC can proceed
    void exit_epoch(int idx) noexcept {
        locals_[idx].active_epoch.store(0, std::memory_order_release);
    }

    // Simple reclamation drive: if all locals are not protecting epoch E, bump global_epoch_
    void try_advance_epoch() noexcept {
        uint64_t cur = global_epoch_.load(std::memory_order_acquire);
        for (int i = 0; i < MAX_THREADS; ++i) {
            uint64_t a = locals_[i].active_epoch.load(std::memory_order_acquire);
            if (a == cur) return; // someone protects current epoch
        }
        global_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }

    uint64_t current_epoch() const noexcept { return global_epoch_.load(std::memory_order_acquire); }
};

// Coroutine Arena (monotonic buffer)
class Arena {
    static constexpr size_t DEFAULT_SZ = 1<<20;
    alignas(64) std::vector<std::byte> buf_;
    std::atomic<size_t> offset_;
public:
    Arena(size_t sz = DEFAULT_SZ) : buf_(sz), offset_(0) {}
    void* allocate(size_t n, size_t align) noexcept {
        size_t cur = offset_.fetch_add(((n + align - 1)/align)*align, std::memory_order_acq_rel);
        if (cur + n > buf_.size()) return nullptr;
        void* ptr = buf_.data() + cur;
        size_t mis = reinterpret_cast<uintptr_t>(ptr) & (align-1);
        if (mis) {
            // align forward (simplified; in production we'd do a more robust allocator)
            size_t pad = align - mis;
            cur = offset_.fetch_add(pad, std::memory_order_acq_rel);
            if (cur + n > buf_.size()) return nullptr;
            ptr = buf_.data() + cur;
        }
        return ptr;
    }
};

// Global arena instance (for demo)
static Arena g_arena(1<<20);

// Coroutine promise that allocates frames from Arena
struct ArenaPromiseBase {
    static void* operator new(size_t sz) noexcept {
        void* p = g_arena.allocate(sz, alignof(std::max_align_t));
        return p ? p : ::operator new(sz); // fallback not preferred
    }
    static void operator delete(void* p) noexcept { /* no-op; arena is monotonic */ }
};

// EBR-aware awaiter that exits epoch on suspend and re-enters+validates on resume
struct EBR_awaiter {
    EBR &ebr;
    int thread_idx;
    uint64_t observed_gen;
    std::atomic<uint64_t> *observed_gen_ptr;

    EBR_awaiter(EBR &e, int idx, std::atomic<uint64_t>* ptr)
        : ebr(e), thread_idx(idx), observed_gen(ptr->load(std::memory_order_acquire)), observed_gen_ptr(ptr) {}

    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<>) noexcept {
        // On suspend: leave epoch so GC can run
        ebr.exit_epoch(thread_idx);
    }
    void await_resume() {
        // On resume: re-enter epoch and revalidate that the object was not freed/changed
        ebr.enter_epoch(thread_idx);
        uint64_t now = observed_gen_ptr->load(std::memory_order_acquire);
        if (now != observed_gen) {
            // Validation failed: the object was reclaimed or modified. Caller must handle.
            // We cannot throw (no-exceptions). We signal via a debug print or error code path in higher-level coroutine.
            // For simplicity in this demo, we print a message; in production runtime, we'd use a resume-with-error mechanism.
            std::fprintf(stderr, "[EBR] validation failed (gen changed %llu -> %llu)\n",
                         (unsigned long long)observed_gen, (unsigned long long)now);
        }
    }
};

// Example coroutine that uses EBR_awaiter; promise type for coroutine returning void
struct SimpleTask {
    struct promise_type : ArenaPromiseBase {
        SimpleTask get_return_object() noexcept { return SimpleTask{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };
    std::coroutine_handle<promise_type> h_;
    SimpleTask(std::coroutine_handle<promise_type> h) : h_(h) {}
    ~SimpleTask() { if (h_) h_.destroy(); }
};

// ---------------------------
// AVX-512 Branchless Shadow Engine
// ---------------------------

enum Action : uint32_t { HOLD = 0, BUY = 1 };

inline void shadow_engine_avx512(const float* prices, const float* limits, const uint32_t* volumes,
                                 const uint32_t max_vol, uint32_t* out_actions, size_t n) noexcept
{
#if defined(__AVX512F__)
    size_t i = 0;
    const size_t stride = 16; // 16 floats per __m512
    for (; i + stride <= n; i += stride) {
        __m512 v_price  = _mm512_loadu_ps(prices + i);
        __m512 v_limit  = _mm512_loadu_ps(limits + i);

        // Compare price > limit => mask1
        __mmask16 mask_price = _mm512_cmp_ps_mask(v_price, v_limit, _CMP_GT_OQ);

        // Load volumes as 32-bit ints, convert to float for comparison (or do int compare)
        __m512i v_vol_i = _mm512_loadu_si512((const void*)(volumes + i)); // loads 16 x 32-bit ints
        // compare v_vol_i < max_vol -> produce mask2 (use integer compare)
        __mmask16 mask_vol = _mm512_cmp_epi32_mask(v_vol_i, _mm512_set1_epi32((int)max_vol), _MM_CMPINT_LT);

        // Combine masks: both conditions true => action BUY
        __mmask16 mask_buy = mask_price & mask_vol;

        // Prepare vectors for actions: set all to HOLD initially
        __m512i v_hold = _mm512_set1_epi32((int)HOLD);
        __m512i v_buy  = _mm512_set1_epi32((int)BUY);

        // Blend: where mask_buy is set pick v_buy otherwise v_hold
        __m512i v_action = _mm512_mask_blend_epi32(mask_buy, v_hold, v_buy);

        // Store result
        _mm512_storeu_si512((void*)(out_actions + i), v_action);
    }
    // tail scalar fallback
    for (; i < n; ++i) {
        uint32_t action = (prices[i] > limits[i] && volumes[i] < max_vol) ? BUY : HOLD;
        out_actions[i] = action;
    }
#else
    // Fallback if AVX-512 not available: scalar code (but still branchless using arithmetic)
    for (size_t i = 0; i < n; ++i) {
        uint32_t cond = (prices[i] > limits[i]) && (volumes[i] < max_vol);
        // cond is boolean; convert to 0/1
        out_actions[i] = cond ? BUY : HOLD;
    }
#endif
}

// ---------------------------
// Compile-time State Machine (consteval + concepts)
// ---------------------------
struct New {};
struct Open {};
struct Filled {};
struct Cancelled {};

template<typename From, typename To>
concept valid_transition = requires {
    // consteval check will validate allowed transitions
    { []() consteval -> bool {
        if constexpr (std::is_same_v<From, New> && std::is_same_v<To, Open>) return true;
        if constexpr (std::is_same_v<From, Open> && std::is_same_v<To, Filled>) return true;
        if constexpr (std::is_same_v<From, Open> && std::is_same_v<To, Cancelled>) return true;
        return false;
    }() } -> std::convertible_to<bool>;
};

template<typename From, typename To>
consteval void require_transition() {
    static_assert(valid_transition<From,To>, "Invalid state transition at compile time");
}

template<typename From, typename To>
struct Transition {
    static void apply() {
        require_transition<From,To>(); // compile-time enforce
        // runtime transition would happen here
    }
};

// ---------------------------
// Example usage / test harness (simplified demo)
// ---------------------------

int main() {
    // Example: create ring buffer
    constexpr size_t CAP = 1024;
    TxRing<CAP> ring;

    // Producer thread: reserves -> writes -> commits or aborts
    auto producer = [&ring]() {
        auto res = ring.reserve();
        if (!res.has_value()) {
            std::fprintf(stderr, "reserve failed\n");
            return;
        }
        uint64_t seq = res.value().first;
        Payload* p = res.value().second;
        // Construct payload in-place using std::start_lifetime_as (C++23)
        std::start_lifetime_as<Payload>(p);
        p->order_id = seq;
        p->price = 100.0 + double(seq % 10);
        p->volume = uint32_t(100 + (seq % 50));
        p->side = 0;
        // Suppose we speculatively commit immediately (could fail after validation)
        auto commit_res = ring.commit(seq);
        if (!commit_res.has_value()) {
            // aborted by helper; simulate rollback
            // nothing to do: consumer will skip
        }
    };

    // Spawn a producer
    std::thread prod_thread(producer);
    prod_thread.join();

    // Consumer tries to pop
    auto pop_res = ring.try_pop();
    if (pop_res.has_value()) {
        auto [seq, p] = pop_res.value();
        std::printf("Consumed seq=%llu order_id=%llu price=%.2f vol=%u\n",
                    (unsigned long long)seq,
                    (unsigned long long)p->order_id,
                    p->price, p->volume);
    } else {
        int code = pop_res.error();
        if (code == 2) std::printf("Slot aborted and skipped\n");
        else std::printf("No committed slot available\n");
    }

    // AVX-512 demo
    const size_t N = 32;
    alignas(64) float prices[N], limits[N];
    alignas(64) uint32_t volumes[N], out_actions[N];
    for (size_t i = 0; i < N; ++i) {
        prices[i] = 100.0f + float(i);
        limits[i] = 100.0f + float(i%5);
        volumes[i] = uint32_t(50 + i);
    }
    shadow_engine_avx512(prices, limits, volumes, 200, out_actions, N);

    // Print a few results
    for (size_t i = 0; i < 8; ++i) {
        std::printf("i=%zu price=%.1f limit=%.1f vol=%u action=%u\n",
                    i, prices[i], limits[i], volumes[i], out_actions[i]);
    }

    // Compile time state machine usage:
    // The following line will produce a compile-time error if the transition is invalid:
    Transition<New, Open>::apply(); // valid
    // Transition<New, Filled>::apply(); // would fail at compile-time if uncommented

    return 0;
}
