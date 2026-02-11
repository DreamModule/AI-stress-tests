// ============================================================================
// Project Event Horizon — Speculative Zero-Copy Kernel
// C++23 | Single-File | Deterministic Latency
// ============================================================================
#pragma once

#include <atomic>
#include <array>
#include <bit>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <type_traits>
#include <utility>

// --- Polyfill: std::expected (GCC 12 lacks <expected>) ---
#if __has_include(<expected>)
#include <expected>
#else
namespace std {
template <class E> class unexpected {
    E val_;
public:
    constexpr explicit unexpected(E e) noexcept : val_(e) {}
    constexpr E const& error() const noexcept { return val_; }
};
template <class T, class E> class expected {
    union { T val_; E err_; };
    bool has_;
public:
    constexpr expected(T v) noexcept : val_(v), has_(true) {}
    constexpr expected(unexpected<E> u) noexcept : err_(u.error()), has_(false) {}
    constexpr explicit operator bool() const noexcept { return has_; }
    constexpr T const& operator*() const noexcept { return val_; }
    constexpr T& operator*() noexcept { return val_; }
    constexpr T const* operator->() const noexcept { return &val_; }
    constexpr E error() const noexcept { return err_; }
};
} // namespace std
#endif

// --- Polyfill: std::start_lifetime_as (C++23 P2590, not in GCC 12) ---
namespace eh_detail {
template <class T>
[[nodiscard]] inline const T* start_lifetime_as(void* p) noexcept {
    // In C++23 this would use std::start_lifetime_as<T>.
    // We use placement-reinterpret + launder, which is the pre-C++23
    // equivalent under trivial-type constraints.
    return std::launder(reinterpret_cast<const T*>(p));
}
template <class T>
[[nodiscard]] inline T* start_lifetime_as_writable(void* p) noexcept {
    return std::launder(reinterpret_cast<T*>(p));
}
} // namespace eh_detail

// ============================================================================
// § 0 — Platform & Utility
// ============================================================================

namespace eh {

inline constexpr std::size_t CACHE_LINE = 64;

enum class ErrorCode : uint8_t {
    BufferFull,
    BufferEmpty,
    SlotAborted,
    EpochStale,
    ValidationFailed,
    InvalidTransition,
    ArenaExhausted,
};

template <class T>
using Result = std::expected<T, ErrorCode>;

// Prevent false sharing on every hot atomic.
template <class T>
struct alignas(CACHE_LINE) PaddedAtomic {
    std::atomic<T> value{};

    T load(std::memory_order mo) const noexcept { return value.load(mo); }
    void store(T v, std::memory_order mo) noexcept { value.store(v, mo); }
    T fetch_add(T v, std::memory_order mo) noexcept { return value.fetch_add(v, mo); }
    bool compare_exchange_strong(T& exp, T des, std::memory_order s, std::memory_order f) noexcept {
        return value.compare_exchange_strong(exp, des, s, f);
    }
};

// ============================================================================
// § 1 — Compile-Time Order State Machine
// ============================================================================

enum class OrderState : uint8_t {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
};

// --- Transition table encoded as type traits ---

template <OrderState From, OrderState To>
struct IsValidTransition : std::false_type {};

// New -> PartiallyFilled, Filled, Cancelled, Rejected
template <> struct IsValidTransition<OrderState::New, OrderState::PartiallyFilled> : std::true_type {};
template <> struct IsValidTransition<OrderState::New, OrderState::Filled>           : std::true_type {};
template <> struct IsValidTransition<OrderState::New, OrderState::Cancelled>        : std::true_type {};
template <> struct IsValidTransition<OrderState::New, OrderState::Rejected>         : std::true_type {};

// PartiallyFilled -> PartiallyFilled, Filled, Cancelled
template <> struct IsValidTransition<OrderState::PartiallyFilled, OrderState::PartiallyFilled> : std::true_type {};
template <> struct IsValidTransition<OrderState::PartiallyFilled, OrderState::Filled>          : std::true_type {};
template <> struct IsValidTransition<OrderState::PartiallyFilled, OrderState::Cancelled>       : std::true_type {};

// concept that gates the transition
template <OrderState From, OrderState To>
concept ValidTransition = IsValidTransition<From, To>::value;

// consteval enforcer — illegal transitions are compile errors
template <OrderState From, OrderState To>
    requires ValidTransition<From, To>
consteval OrderState transition() noexcept {
    return To;
}

// Typed wrapper so we carry state in the type system
template <OrderState S>
struct OrderTag {
    static constexpr OrderState state = S;

    template <OrderState Next>
        requires ValidTransition<S, Next>
    [[nodiscard]] consteval static OrderTag<Next> to() noexcept { return {}; }
};

// ============================================================================
// § 2 — Arena Allocator (monotonic, no-free, coroutine-frame target)
// ============================================================================

class Arena {
public:
    explicit Arena(void* buf, std::size_t cap) noexcept
        : base_(static_cast<char*>(buf)), capacity_(cap), offset_(0) {}

    [[nodiscard]] void* allocate(std::size_t size, std::size_t align = alignof(max_align_t)) noexcept {
        std::size_t cur = offset_.load(std::memory_order_acquire);
        for (;;) {
            std::size_t aligned = (cur + align - 1) & ~(align - 1);
            std::size_t next = aligned + size;
            if (next > capacity_) return nullptr;
            if (offset_.compare_exchange_strong(cur, next,
                    std::memory_order_release, std::memory_order_acquire))
                return base_ + aligned;
        }
    }

    void reset() noexcept { offset_.store(0, std::memory_order_release); }

private:
    char* base_;
    std::size_t capacity_;
    std::atomic<std::size_t> offset_;
};

// Global arena pointer for coroutine promise allocation.
inline thread_local Arena* tl_arena = nullptr;

// ============================================================================
// § 3 — Transactional Lock-Free Ring Buffer (Dual-Sequence-Barrier)
// ============================================================================
//
// Design:
//   Each slot carries a 64-bit "tag" atomic.  The tag encodes both the
//   *wrap generation* and the *slot state* in a single atomic word,
//   eliminating the need for a separate boolean validity flag.
//
//   Tag layout (64 bit):
//     [63:2]  generation (sequence / capacity)
//     [1:0]   phase:
//               00 = FREE      — slot available for producer claim
//               01 = RESERVED  — producer writing (Phase 1)
//               10 = COMMITTED — data valid, consumer may read
//               11 = ABORTED   — rolled back, consumer must skip
//
//   Consumer protocol:
//     The consumer tracks its own expected generation.  It spins on the
//     slot tag until the generation matches.  Then:
//       - COMMITTED → process, advance.
//       - ABORTED   → skip, advance (wait-free).
//       - RESERVED  → producer in-flight; consumer spin-waits only on
//                      this *specific* slot, but never blocks others.
//     Because generation + phase are in a single atomic load we get a
//     consistent snapshot without seq_cst.
//
//   Rollback:
//     Producer CAS tag from RESERVED|gen to ABORTED|gen.  The payload
//     is left dirty — consumers never touch it because state != COMMITTED.
//     A subsequent producer claiming the same slot in the *next* generation
//     will overwrite the payload and set a new tag, so no leak occurs.

inline constexpr uint64_t PHASE_FREE      = 0b00;
inline constexpr uint64_t PHASE_RESERVED  = 0b01;
inline constexpr uint64_t PHASE_COMMITTED = 0b10;
inline constexpr uint64_t PHASE_ABORTED   = 0b11;
inline constexpr uint64_t PHASE_MASK      = 0b11;

inline constexpr uint64_t make_tag(uint64_t gen, uint64_t phase) noexcept {
    return (gen << 2) | phase;
}
inline constexpr uint64_t tag_gen(uint64_t tag)   noexcept { return tag >> 2; }
inline constexpr uint64_t tag_phase(uint64_t tag) noexcept { return tag & PHASE_MASK; }

struct alignas(CACHE_LINE) OrderEvent {
    uint64_t order_id;
    float    price;
    float    volume;
    uint32_t symbol_idx;
    OrderState state;
    uint8_t  padding_[3]{};
};

template <std::size_t N>
    requires (std::has_single_bit(N))  // power-of-two enforced
class TransactionalRingBuffer {
    static constexpr uint64_t MASK = N - 1;

    struct alignas(CACHE_LINE) Slot {
        std::atomic<uint64_t> tag;
        alignas(alignof(OrderEvent)) char storage[sizeof(OrderEvent)];
    };

    std::array<Slot, N> slots_{};
    PaddedAtomic<uint64_t> claim_seq_;   // next sequence for producer claim
    PaddedAtomic<uint64_t> consume_seq_; // consumer cursor (informational)

public:
    TransactionalRingBuffer() noexcept {
        // Initialize all tags to FREE at generation 0.
        for (std::size_t i = 0; i < N; ++i)
            slots_[i].tag.store(make_tag(0, PHASE_FREE), std::memory_order_release);
        claim_seq_.store(0, std::memory_order_release);
        consume_seq_.store(0, std::memory_order_release);
    }

    // --- Producer API ---

    struct Reservation {
        uint64_t seq;
        OrderEvent* ptr;
    };

    [[nodiscard]] Result<Reservation> reserve() noexcept {
        uint64_t seq = claim_seq_.fetch_add(1, std::memory_order_acquire);
        uint64_t idx = seq & MASK;
        uint64_t gen = seq / N;

        Slot& s = slots_[idx];
        // Spin until slot is FREE at correct generation.
        uint64_t expected = make_tag(gen, PHASE_FREE);
        while (!s.tag.compare_exchange_strong(expected, make_tag(gen, PHASE_RESERVED),
                std::memory_order_release, std::memory_order_acquire)) {
            // If wrong generation, buffer is full / lapped.
            if (tag_gen(expected) != gen && tag_phase(expected) != PHASE_FREE)
                return std::unexpected(ErrorCode::BufferFull);
            expected = make_tag(gen, PHASE_FREE);
            _mm_pause();
        }

        // start_lifetime_as for strict-aliasing compliance (C++23)
        auto* evt = eh_detail::start_lifetime_as_writable<OrderEvent>(s.storage);
        return Reservation{ seq, evt };
    }

    void commit(uint64_t seq) noexcept {
        uint64_t idx = seq & MASK;
        uint64_t gen = seq / N;
        Slot& s = slots_[idx];
        // CAS RESERVED -> COMMITTED (same generation)
        uint64_t expected = make_tag(gen, PHASE_RESERVED);
        s.tag.compare_exchange_strong(expected, make_tag(gen, PHASE_COMMITTED),
            std::memory_order_release, std::memory_order_acquire);
    }

    void abort(uint64_t seq) noexcept {
        uint64_t idx = seq & MASK;
        uint64_t gen = seq / N;
        Slot& s = slots_[idx];
        // CAS RESERVED -> ABORTED (same generation)
        uint64_t expected = make_tag(gen, PHASE_RESERVED);
        s.tag.compare_exchange_strong(expected, make_tag(gen, PHASE_ABORTED),
            std::memory_order_release, std::memory_order_acquire);
    }

    // --- Consumer API ---

    struct ConsumeResult {
        const OrderEvent* evt;
        uint64_t seq;
    };

    // Wait-free try_consume: returns immediately with data or skip or empty.
    [[nodiscard]] Result<ConsumeResult> try_consume(uint64_t consumer_seq) noexcept {
        uint64_t idx = consumer_seq & MASK;
        uint64_t gen = consumer_seq / N;
        Slot& s = slots_[idx];

        uint64_t tag = s.tag.load(std::memory_order_acquire);
        if (tag_gen(tag) != gen)
            return std::unexpected(ErrorCode::BufferEmpty);

        uint64_t phase = tag_phase(tag);

        if (phase == PHASE_COMMITTED) {
            auto* evt = eh_detail::start_lifetime_as<OrderEvent>(s.storage);
            return ConsumeResult{ evt, consumer_seq };
        }

        if (phase == PHASE_ABORTED) {
            // Skip: release the slot for the next generation.
            s.tag.store(make_tag(gen + 1, PHASE_FREE), std::memory_order_release);
            return std::unexpected(ErrorCode::SlotAborted);
        }

        // RESERVED — producer still in-flight; not ready yet.
        return std::unexpected(ErrorCode::BufferEmpty);
    }

    // Release a consumed slot for reuse (next generation).
    void release(uint64_t seq) noexcept {
        uint64_t idx = seq & MASK;
        uint64_t gen = seq / N;
        Slot& s = slots_[idx];
        s.tag.store(make_tag(gen + 1, PHASE_FREE), std::memory_order_release);
    }
};

// ============================================================================
// § 4 — Epoch-Based Reclamation with Coroutine Integration
// ============================================================================

inline constexpr uint32_t MAX_THREADS = 64;

class EpochManager {
public:
    static constexpr uint64_t INACTIVE = UINT64_MAX;

    EpochManager() noexcept {
        global_epoch_.store(0, std::memory_order_release);
        for (auto& le : local_epochs_)
            le.store(INACTIVE, std::memory_order_release);
    }

    uint64_t enter(uint32_t tid) noexcept {
        uint64_t ge = global_epoch_.load(std::memory_order_acquire);
        local_epochs_[tid].store(ge, std::memory_order_release);
        return ge;
    }

    void exit(uint32_t tid) noexcept {
        local_epochs_[tid].store(INACTIVE, std::memory_order_release);
    }

    bool try_advance() noexcept {
        uint64_t cur = global_epoch_.load(std::memory_order_acquire);
        // Check all threads have either exited or caught up.
        for (uint32_t i = 0; i < MAX_THREADS; ++i) {
            uint64_t le = local_epochs_[i].load(std::memory_order_acquire);
            if (le != INACTIVE && le < cur) return false;
        }
        uint64_t next = cur + 1;
        return global_epoch_.compare_exchange_strong(cur, next,
            std::memory_order_release, std::memory_order_acquire);
    }

    uint64_t current_epoch() const noexcept {
        return global_epoch_.load(std::memory_order_acquire);
    }

    // Versioned pointer for ABA protection
    struct VersionedRef {
        void*    ptr;
        uint64_t version;   // ABA generation counter
        uint64_t epoch;     // epoch at which pointer was valid
    };

    struct VersionedSlot {
        std::atomic<uint64_t> version{0};
        std::atomic<bool>     deleted{false};

        [[nodiscard]] bool is_valid(uint64_t expected_version) const noexcept {
            return !deleted.load(std::memory_order_acquire)
                && version.load(std::memory_order_acquire) == expected_version;
        }
    };

private:
    PaddedAtomic<uint64_t> global_epoch_;
    alignas(CACHE_LINE) std::atomic<uint64_t> local_epochs_[MAX_THREADS];
};

inline EpochManager g_epoch_mgr;

// --- RAII Epoch Guard ---
struct EpochGuard {
    uint32_t tid;
    uint64_t epoch;

    explicit EpochGuard(uint32_t t) noexcept : tid(t), epoch(g_epoch_mgr.enter(t)) {}
    ~EpochGuard() noexcept { g_epoch_mgr.exit(tid); }

    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;
};

// ============================================================================
// § 4b — Coroutine Task + Epoch-Aware Awaiter
// ============================================================================

// Minimal coroutine Task that allocates on the thread-local Arena.
struct Task {
    struct promise_type {
        Task get_return_object() noexcept { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_never  initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend()   noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { __builtin_trap(); }

        // Arena allocation for coroutine frame
        static void* operator new(std::size_t size) noexcept {
            if (tl_arena) {
                void* p = tl_arena->allocate(size);
                if (p) return p;
            }
            __builtin_trap(); // Arena exhausted — hard fail, no exceptions
        }
        static void operator delete(void* /*p*/, std::size_t /*size*/) noexcept {
            // Arena is monotonic — no individual free.
        }
    };

    std::coroutine_handle<promise_type> handle_;

    explicit Task(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
    Task(Task&& o) noexcept : handle_(std::exchange(o.handle_, nullptr)) {}
    ~Task() { if (handle_) handle_.destroy(); }
};

// Epoch-aware awaiter:
//   On suspend → exits epoch (unblocks GC).
//   On resume  → re-enters epoch, re-validates versioned reference.
struct EpochAwaiter {
    uint32_t                     tid;
    EpochManager::VersionedSlot* slot;      // the object's version tracker
    uint64_t                     saved_ver; // version captured before suspend
    bool                         still_valid = false;

    bool await_ready() const noexcept { return false; } // always suspend

    void await_suspend(std::coroutine_handle<>) noexcept {
        // Exit epoch — let GC proceed while we're parked.
        g_epoch_mgr.exit(tid);
    }

    bool await_resume() noexcept {
        // Re-enter epoch.
        g_epoch_mgr.enter(tid);
        // Re-validate: check version hasn't changed and object not deleted.
        still_valid = slot->is_valid(saved_ver);
        return still_valid;
    }
};

// Helper to create the awaiter from current epoch guard + versioned slot.
inline EpochAwaiter epoch_safe_suspend(uint32_t tid,
                                       EpochManager::VersionedSlot& slot) noexcept {
    uint64_t ver = slot.version.load(std::memory_order_acquire);
    return EpochAwaiter{ tid, &slot, ver, false };
}

// ============================================================================
// § 5 — AVX-512 Branchless Shadow Engine
// ============================================================================
//
// Evaluates 16 orders simultaneously. For each:
//   if (price > limit && volume < max_vol) → action = BUY_SIGNAL
//   else                                   → action = HOLD_SIGNAL
//
// Entirely mask-based, zero branches in the hot path.

inline constexpr float BUY_SIGNAL  = 1.0f;
inline constexpr float HOLD_SIGNAL = 0.0f;

struct alignas(64) ShadowBatch {
    float prices[16];
    float volumes[16];
};

struct alignas(64) ShadowResult {
    float actions[16]; // BUY_SIGNAL or HOLD_SIGNAL
};

#ifdef __AVX512F__

inline void shadow_engine_evaluate(
    const ShadowBatch& batch,
    float price_limit,
    float max_volume,
    ShadowResult& out) noexcept
{
    // Load 16 prices and 16 volumes.
    __m512 v_prices  = _mm512_load_ps(batch.prices);
    __m512 v_volumes = _mm512_load_ps(batch.volumes);

    // Broadcast thresholds.
    __m512 v_limit   = _mm512_set1_ps(price_limit);
    __m512 v_maxvol  = _mm512_set1_ps(max_volume);

    // Branchless comparisons → mask registers.
    // price > limit
    __mmask16 price_ok = _mm512_cmp_ps_mask(v_prices, v_limit, _CMP_GT_OQ);
    // volume < max_vol
    __mmask16 vol_ok   = _mm512_cmp_ps_mask(v_volumes, v_maxvol, _CMP_LT_OQ);

    // Combined mask: both conditions true.
    __mmask16 buy_mask = _kand_mask16(price_ok, vol_ok);

    // Prepare both "branch" results.
    __m512 v_buy  = _mm512_set1_ps(BUY_SIGNAL);
    __m512 v_hold = _mm512_set1_ps(HOLD_SIGNAL);

    // Select: where buy_mask is set choose BUY, else HOLD.
    // blend: result[i] = mask[i] ? a[i] : b[i]
    __m512 v_result = _mm512_mask_blend_ps(buy_mask, v_hold, v_buy);

    _mm512_store_ps(out.actions, v_result);
}

// Extended shadow engine: also computes notional = price * volume for buys,
// zero for holds — all branchless.
inline void shadow_engine_with_notional(
    const ShadowBatch& batch,
    float price_limit,
    float max_volume,
    ShadowResult& actions_out,
    float notional_out[16]) noexcept
{
    __m512 v_prices  = _mm512_load_ps(batch.prices);
    __m512 v_volumes = _mm512_load_ps(batch.volumes);
    __m512 v_limit   = _mm512_set1_ps(price_limit);
    __m512 v_maxvol  = _mm512_set1_ps(max_volume);

    __mmask16 price_ok = _mm512_cmp_ps_mask(v_prices, v_limit, _CMP_GT_OQ);
    __mmask16 vol_ok   = _mm512_cmp_ps_mask(v_volumes, v_maxvol, _CMP_LT_OQ);
    __mmask16 buy_mask = _kand_mask16(price_ok, vol_ok);

    __m512 v_buy  = _mm512_set1_ps(BUY_SIGNAL);
    __m512 v_hold = _mm512_set1_ps(HOLD_SIGNAL);
    __m512 v_result = _mm512_mask_blend_ps(buy_mask, v_hold, v_buy);
    _mm512_store_ps(actions_out.actions, v_result);

    // Notional: price * volume where buy, else 0.0
    __m512 v_zero     = _mm512_setzero_ps();
    __m512 v_notional = _mm512_mul_ps(v_prices, v_volumes);
    __m512 v_masked   = _mm512_mask_blend_ps(buy_mask, v_zero, v_notional);
    _mm512_store_ps(notional_out, v_masked);
}

#endif // __AVX512F__

// ============================================================================
// § 6 — Putting it together: Speculative Order Processor
// ============================================================================

inline constexpr std::size_t RING_SIZE = 1024; // must be power-of-two

using OrderRing = TransactionalRingBuffer<RING_SIZE>;

// Simulated risk check (would be async in production).
inline bool run_risk_check(const OrderEvent& evt) noexcept {
    // Placeholder: reject if volume exceeds threshold.
    return evt.volume < 100000.0f;
}

// Producer: speculatively writes, then commits or aborts.
inline Result<uint64_t> speculative_submit(OrderRing& ring, const OrderEvent& evt) noexcept {
    auto res = ring.reserve();
    if (!res) return std::unexpected(res.error());

    auto [seq, ptr] = *res;

    // Phase 1: write speculatively.
    std::memcpy(ptr, &evt, sizeof(OrderEvent));
    std::atomic_thread_fence(std::memory_order_release);

    // Phase 2: risk validation.
    bool ok = run_risk_check(evt);

    // Commit or Abort based on validation.
    // No branch on the critical CAS path — both paths are single atomic ops.
    uint64_t idx = seq & (RING_SIZE - 1);
    uint64_t gen = seq / RING_SIZE;
    (void)idx; (void)gen;

    // We use the ring's commit/abort which are CAS-based.
    ok ? ring.commit(seq) : ring.abort(seq);

    return ok ? Result<uint64_t>{seq} : std::unexpected(ErrorCode::ValidationFailed);
}

// Consumer loop step: returns next valid event or error.
inline Result<OrderRing::ConsumeResult> consume_next(OrderRing& ring,
                                                     uint64_t& consumer_seq) noexcept {
    for (;;) {
        auto res = ring.try_consume(consumer_seq);
        if (res) {
            // Got a committed event. Caller must call ring.release() after processing.
            return *res;
        }
        // SlotAborted → already freed by try_consume, advance and retry.
        if (res.error() == ErrorCode::SlotAborted) {
            ++consumer_seq;
            continue;
        }
        // BufferEmpty → nothing to consume right now.
        return std::unexpected(res.error());
    }
}

} // namespace eh

// ============================================================================
// § 7 — ACID TEST: Hardware Interaction Analysis
// ============================================================================

/*
 * ACID TEST ANSWER:
 * =================
 *
 * "How does your Transactional Ring Buffer rollback mechanism interact with
 *  the L1/L2 Store Buffers? Explain how you prevent a consumer from reading
 *  a speculatively written value that was later aborted, specifically in the
 *  context of Store-to-Load forwarding failures."
 *
 * ANSWER:
 *
 * 1. THE STORE BUFFER PROBLEM
 *
 *    When a producer writes payload data (Phase 1) and then CAS-writes the
 *    tag to ABORTED (Phase 2-abort), these are two independent stores that
 *    enter the core's Store Buffer.  On x86-64, stores are retired in
 *    program order to the L1d cache (TSO guarantee), but Store-to-Load
 *    forwarding within the *same* core can make a load "see" a store that
 *    hasn't yet been globally visible.
 *
 *    The critical risk: a consumer on the SAME physical core (hyper-thread
 *    sibling) might observe the payload store via store-buffer forwarding
 *    but miss the tag transition to ABORTED — reading dirty speculative data.
 *
 * 2. HOW OUR DESIGN PREVENTS THIS
 *
 *    a) TAG-GATED ACCESS: The consumer NEVER reads the payload storage
 *       unless it first loads the tag and observes PHASE_COMMITTED.  The
 *       tag load uses memory_order_acquire, which on x86 compiles to a
 *       plain MOV but enforces compiler ordering — no subsequent loads
 *       (including the payload read) can be reordered before the tag load
 *       by the compiler.
 *
 *    b) SINGLE-WORD ATOMIC GATE: The tag is a single aligned uint64_t.
 *       x86 guarantees that naturally-aligned 8-byte loads/stores are
 *       atomic.  The consumer sees the tag in exactly one state:
 *       FREE / RESERVED / COMMITTED / ABORTED.  There is no torn read.
 *
 *    c) STORE-TO-LOAD FORWARDING IS IRRELEVANT CROSS-CORE: Store-buffer
 *       forwarding only occurs within the same physical core.  On x86 TSO,
 *       stores from Core A become visible to Core B only after they drain
 *       from the store buffer into L1d and are then coherent via MESIF/MOESI.
 *       This means a consumer on a different core will observe the tag
 *       store (ABORTED) and the payload store in program order — the tag
 *       CAS drains AFTER the payload memcpy because x86 maintains store
 *       order.
 *
 *    d) SAME-CORE (HT SIBLING) SCENARIO: If consumer runs on the same
 *       physical core, store-to-load forwarding could theoretically let it
 *       see the payload bytes.  However, the consumer's *first* operation
 *       is an acquire-load of the tag.  On x86, this load CANNOT be
 *       satisfied by forwarding from the *other* logical core's store
 *       buffer — store buffers are per-logical-core and not shared between
 *       hyper-thread siblings on Intel (since Sunny Cove / Ice Lake) for
 *       security (MDS mitigations).  So the tag load goes to L1d, which
 *       reflects globally-ordered stores.  By the time the tag reads
 *       COMMITTED, the payload stores have necessarily already drained to
 *       L1d (x86 store ordering).
 *
 *    e) THE ABORT PATH SPECIFICALLY:
 *       - Producer writes payload → stores enter store buffer.
 *       - Producer CAS tag to ABORTED → another store, ordered AFTER
 *         payload stores in the store buffer (x86 TSO).
 *       - Consumer loads tag → sees ABORTED → never reads payload.
 *       - The dirty payload remains in the cache line but is logically
 *         invisible.  When a future producer claims this slot at the next
 *         generation, it overwrites the payload and CAS-es a new tag,
 *         which naturally invalidates any stale cache line via coherence.
 *
 *    f) WHY A SIMPLE BOOLEAN FLAG IS INSUFFICIENT:
 *       A separate `std::atomic<bool> valid` field creates two independent
 *       cache lines (or at minimum two independent atomic objects).  A
 *       consumer could observe `valid == true` from a stale read while the
 *       payload is half-written, because the bool and payload are not in the
 *       same atomic domain.  Our design merges phase state INTO the sequence
 *       tag — a single atomic word that acts as the gatekeeper.  Acquiring
 *       the tag with the correct generation+COMMITTED state provides a
 *       happens-before edge that guarantees all prior payload stores are
 *       visible.
 *
 * 3. SUMMARY
 *
 *    The x86 TSO memory model, combined with our single-atomic-tag design
 *    and acquire/release ordering, ensures that:
 *      - A consumer never reads payload from an ABORTED slot.
 *      - Store-to-load forwarding cannot leak speculative payload across
 *        logical cores.
 *      - The release-store of the tag (commit or abort) acts as the
 *        publication barrier, and the consumer's acquire-load of the tag
 *        is the corresponding acquisition barrier.
 *      - No seq_cst fences or MFENCE instructions are needed — the
 *        natural x86 store ordering provides sufficient guarantees when
 *        all control flow is gated through a single tag word.
 */
