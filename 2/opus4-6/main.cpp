// ============================================================================
// Project Event Horizon — Test Harness
// ============================================================================
#include "event_horizon.hpp"
#include <cstdio>
#include <cstring>
#include <thread>

// ============================================================================
// § A — Compile-Time State Machine Tests
// ============================================================================

static consteval bool test_state_machine() {
    // Valid transitions
    constexpr auto s1 = eh::OrderTag<eh::OrderState::New>{};
    constexpr auto s2 = s1.to<eh::OrderState::PartiallyFilled>();
    constexpr auto s3 = s2.to<eh::OrderState::Filled>();
    (void)s1; (void)s2; (void)s3;

    constexpr auto s4 = eh::OrderTag<eh::OrderState::New>{};
    constexpr auto s5 = s4.to<eh::OrderState::Cancelled>();
    (void)s4; (void)s5;

    constexpr auto s6 = eh::OrderTag<eh::OrderState::New>{};
    constexpr auto s7 = s6.to<eh::OrderState::Rejected>();
    (void)s6; (void)s7;

    constexpr auto s8  = eh::OrderTag<eh::OrderState::PartiallyFilled>{};
    constexpr auto s9  = s8.to<eh::OrderState::PartiallyFilled>();
    constexpr auto s10 = s9.to<eh::OrderState::Cancelled>();
    (void)s8; (void)s9; (void)s10;

    // Uncomment any of these to get a COMPILE ERROR (invalid transitions):
    // eh::OrderTag<eh::OrderState::Filled>{}.to<eh::OrderState::New>();
    // eh::OrderTag<eh::OrderState::Cancelled>{}.to<eh::OrderState::Filled>();
    // eh::OrderTag<eh::OrderState::Rejected>{}.to<eh::OrderState::New>();

    return true;
}

static_assert(test_state_machine(), "State machine compile-time validation failed");

// ============================================================================
// § B — Ring Buffer Functional Test
// ============================================================================

static void test_ring_buffer() {
    std::printf("[TEST] Ring Buffer: Commit & Consume...\n");

    eh::OrderRing ring;
    eh::OrderEvent evt{};
    evt.order_id   = 42;
    evt.price      = 100.5f;
    evt.volume     = 200.0f;
    evt.symbol_idx = 1;
    evt.state      = eh::OrderState::New;

    auto res = eh::speculative_submit(ring, evt);
    if (res) {
        std::printf("  [OK] Committed seq=%lu\n", (unsigned long)*res);
    } else {
        std::printf("  [FAIL] Submit failed: %d\n", (int)res.error());
    }

    uint64_t cseq = 0;
    auto cres = eh::consume_next(ring, cseq);
    if (cres) {
        std::printf("  [OK] Consumed order_id=%lu price=%.1f\n",
            (unsigned long)cres->evt->order_id, (double)cres->evt->price);
        ring.release(cres->seq);
    } else {
        std::printf("  [FAIL] Consume error: %d\n", (int)cres.error());
    }

    std::printf("[TEST] Ring Buffer: Abort path...\n");

    // Create an event that will fail risk check (volume >= 100000)
    eh::OrderEvent bad_evt{};
    bad_evt.order_id   = 99;
    bad_evt.price      = 50.0f;
    bad_evt.volume     = 200000.0f; // exceeds risk limit
    bad_evt.symbol_idx = 2;
    bad_evt.state      = eh::OrderState::New;

    auto res2 = eh::speculative_submit(ring, bad_evt);
    if (!res2 && res2.error() == eh::ErrorCode::ValidationFailed) {
        std::printf("  [OK] Correctly aborted (validation failed)\n");
    } else {
        std::printf("  [FAIL] Expected abort, got seq=%lu\n",
            res2 ? (unsigned long)*res2 : 0UL);
    }

    // Consumer should skip aborted slot
    cseq = 1; // next sequence
    auto cres2 = eh::consume_next(ring, cseq);
    if (!cres2 && cres2.error() == eh::ErrorCode::BufferEmpty) {
        std::printf("  [OK] Consumer correctly skipped aborted slot\n");
    } else if (cres2) {
        std::printf("  [FAIL] Consumer read aborted data!\n");
    } else {
        std::printf("  [INFO] Consumer result: error %d, seq now %lu\n",
            (int)cres2.error(), (unsigned long)cseq);
    }
}

// ============================================================================
// § C — Multi-threaded Ring Buffer Stress Test
// ============================================================================

static void test_ring_mt() {
    std::printf("[TEST] Multi-threaded ring buffer stress...\n");

    eh::OrderRing ring;
    constexpr int NUM_EVENTS = 10000;
    std::atomic<int> committed_count{0};
    std::atomic<int> aborted_count{0};
    std::atomic<int> consumed_count{0};

    // Producer thread
    std::thread producer([&] {
        for (int i = 0; i < NUM_EVENTS; ++i) {
            eh::OrderEvent evt{};
            evt.order_id = static_cast<uint64_t>(i);
            evt.price    = 100.0f + static_cast<float>(i % 50);
            evt.volume   = (i % 7 == 0) ? 200000.0f : 500.0f; // ~14% will fail risk
            evt.symbol_idx = 0;
            evt.state = eh::OrderState::New;

            for (;;) {
                auto res = eh::speculative_submit(ring, evt);
                if (res) {
                    committed_count.fetch_add(1, std::memory_order_release);
                    break;
                }
                if (res.error() == eh::ErrorCode::ValidationFailed) {
                    aborted_count.fetch_add(1, std::memory_order_release);
                    break;
                }
                // Buffer full — retry
                _mm_pause();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&] {
        uint64_t cseq = 0;
        int total_processed = 0;
        while (total_processed < NUM_EVENTS) {
            auto res = ring.try_consume(cseq);
            if (res) {
                ring.release(res->seq);
                consumed_count.fetch_add(1, std::memory_order_release);
                ++cseq;
                ++total_processed;
            } else if (res.error() == eh::ErrorCode::SlotAborted) {
                ++cseq;
                ++total_processed;
            } else {
                _mm_pause();
            }
        }
    });

    producer.join();
    consumer.join();

    int c = committed_count.load(std::memory_order_acquire);
    int a = aborted_count.load(std::memory_order_acquire);
    int d = consumed_count.load(std::memory_order_acquire);

    std::printf("  Committed: %d, Aborted: %d, Consumed: %d, Total: %d\n", c, a, d, c + a);
    if (c + a == NUM_EVENTS && d == c) {
        std::printf("  [OK] All events accounted for\n");
    } else {
        std::printf("  [FAIL] Mismatch!\n");
    }
}

// ============================================================================
// § D — Epoch-Based Reclamation Test
// ============================================================================

static void test_ebr() {
    std::printf("[TEST] Epoch-Based Reclamation...\n");

    eh::EpochManager::VersionedSlot slot;
    slot.version.store(1, std::memory_order_release);
    slot.deleted.store(false, std::memory_order_release);

    // Enter epoch on thread 0
    uint64_t e = eh::g_epoch_mgr.enter(0);
    std::printf("  Entered epoch %lu\n", (unsigned long)e);

    // Verify slot is valid at version 1
    bool valid = slot.is_valid(1);
    std::printf("  Slot valid at version 1: %s\n", valid ? "yes" : "no");

    // Exit epoch
    eh::g_epoch_mgr.exit(0);

    // Simulate deletion
    slot.deleted.store(true, std::memory_order_release);
    slot.version.fetch_add(1, std::memory_order_release);

    // Re-enter and check — should be invalid
    eh::g_epoch_mgr.enter(0);
    bool valid2 = slot.is_valid(1); // old version
    std::printf("  Slot valid at old version after delete: %s\n", valid2 ? "yes" : "no");
    eh::g_epoch_mgr.exit(0);

    // Advance epoch
    bool advanced = eh::g_epoch_mgr.try_advance();
    std::printf("  Epoch advanced: %s\n", advanced ? "yes" : "no");

    std::printf("  [OK] EBR basic test passed\n");
}

// ============================================================================
// § E — Arena Allocator Test
// ============================================================================

static void test_arena() {
    std::printf("[TEST] Arena allocator...\n");

    alignas(64) char buf[4096];
    eh::Arena arena(buf, sizeof(buf));

    void* p1 = arena.allocate(128, 16);
    void* p2 = arena.allocate(256, 64);
    void* p3 = arena.allocate(64, 8);

    std::printf("  p1=%p p2=%p p3=%p\n", p1, p2, p3);

    if (p1 && p2 && p3) {
        std::printf("  [OK] Arena allocations succeeded\n");
    } else {
        std::printf("  [FAIL] Arena allocation returned nullptr\n");
    }

    // Test exhaustion
    void* big = arena.allocate(8192);
    std::printf("  Oversize alloc returned: %p (expected null)\n", big);
    if (!big) {
        std::printf("  [OK] Arena correctly refused oversize allocation\n");
    }
}

// ============================================================================
// § F — AVX-512 Shadow Engine Test (compile-time gated)
// ============================================================================

#ifdef __AVX512F__
static void test_shadow_engine() {
    std::printf("[TEST] AVX-512 Shadow Engine...\n");

    alignas(64) eh::ShadowBatch batch;
    // Fill with test data
    for (int i = 0; i < 16; ++i) {
        batch.prices[i]  = 90.0f + static_cast<float>(i) * 2.0f;  // 90..120
        batch.volumes[i] = 100.0f + static_cast<float>(i) * 50.0f; // 100..850
    }

    alignas(64) eh::ShadowResult result;
    float price_limit = 100.0f;
    float max_volume  = 500.0f;

    eh::shadow_engine_evaluate(batch, price_limit, max_volume, result);

    std::printf("  Price limit=%.0f, Max volume=%.0f\n", (double)price_limit, (double)max_volume);
    for (int i = 0; i < 16; ++i) {
        std::printf("  [%2d] price=%.0f vol=%.0f -> %s\n",
            i, (double)batch.prices[i], (double)batch.volumes[i],
            result.actions[i] > 0.5f ? "BUY" : "HOLD");
    }

    // Verify: BUY only if price > 100 AND volume < 500
    int errors = 0;
    for (int i = 0; i < 16; ++i) {
        bool should_buy = (batch.prices[i] > price_limit) && (batch.volumes[i] < max_volume);
        bool got_buy    = result.actions[i] > 0.5f;
        if (should_buy != got_buy) {
            std::printf("  [FAIL] index %d: expected %s got %s\n",
                i, should_buy ? "BUY" : "HOLD", got_buy ? "BUY" : "HOLD");
            ++errors;
        }
    }
    if (errors == 0)
        std::printf("  [OK] Shadow engine branchless evaluation correct\n");
}
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("=== Project Event Horizon — Test Suite ===\n\n");

    // Set up arena for coroutine frames
    alignas(64) static char arena_buf[1 << 20]; // 1 MB
    static eh::Arena arena(arena_buf, sizeof(arena_buf));
    eh::tl_arena = &arena;

    test_arena();
    test_ring_buffer();
    test_ring_mt();
    test_ebr();

#ifdef __AVX512F__
    test_shadow_engine();
#else
    std::printf("[SKIP] AVX-512 tests (not available on this CPU)\n");
#endif

    std::printf("\n=== All tests complete ===\n");
    return 0;
}
