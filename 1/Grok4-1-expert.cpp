// =============================================================================
// ECS Particle Simulation — C++20, Data-Oriented, Lock-Free
// =============================================================================
// Compile: g++ -std=c++20 -O3 -Wall -Wextra -march=native -o ecs ecs_particles.cpp -lpthread
// =============================================================================

#include <atomic>
#include <array>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// Constants
// =============================================================================

static constexpr std::size_t kCacheLine      = 64;
static constexpr std::size_t kEntityCount     = 100’000;
static constexpr std::size_t kSimIterations   = 100;
static constexpr std::size_t kChunkSize       = 4096;
static constexpr float       kDeltaTime       = 0.016f;   // ~60 FPS

static_assert(sizeof(float) == 4, “Expected IEEE-754 32-bit floats”);

// =============================================================================
// 1. Arena Allocator — cache-line aligned, monotonic bump allocator
//    Zero allocations in the hot loop; all memory pre-allocated.
// =============================================================================

class ArenaAllocator {
public:
explicit ArenaAllocator(std::size_t capacity) {
// aligned_alloc requires size to be a multiple of alignment
capacity_ = (capacity + kCacheLine - 1) & ~(kCacheLine - 1);
base_     = static_cast<std::byte*>(std::aligned_alloc(kCacheLine, capacity_));
if (!base_) {
std::fputs(“ArenaAllocator: allocation failed\n”, stderr);
std::abort();
}
offset_ = 0;
}

```
~ArenaAllocator() { std::free(base_); }

ArenaAllocator(const ArenaAllocator&)            = delete;
ArenaAllocator& operator=(const ArenaAllocator&) = delete;

[[nodiscard]] void* allocate(std::size_t size,
                             std::size_t alignment = kCacheLine) noexcept {
    std::size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + size > capacity_) {
        std::fputs("ArenaAllocator: out of memory\n", stderr);
        std::abort();
    }
    offset_ = aligned + size;
    return base_ + aligned;
}

template <typename T>
[[nodiscard]] T* alloc_array(std::size_t count) noexcept {
    constexpr std::size_t align = (alignof(T) < kCacheLine) ? kCacheLine : alignof(T);
    return static_cast<T*>(allocate(sizeof(T) * count, align));
}

void reset() noexcept { offset_ = 0; }

[[nodiscard]] std::size_t used()     const noexcept { return offset_;   }
[[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
```

private:
std::byte*  base_     = nullptr;
std::size_t capacity_ = 0;
std::size_t offset_   = 0;
};

// =============================================================================
// 2. Lock-Free Job System
//    - Persistent thread pool
//    - Atomic work-stealing via fetch_add on a shared index
//    - No mutex, no semaphore, no lock_guard — pure atomics
//    - Each hot atomic lives on its own cache line (no false sharing)
// =============================================================================

class JobSystem {
public:
using TaskFn = void (*)(std::size_t begin, std::size_t end, void* ctx);

```
explicit JobSystem(unsigned num_workers = 0) {
    unsigned hw = std::thread::hardware_concurrency();
    if (num_workers == 0)
        num_workers = (hw > 1) ? (hw - 1) : 1;

    shutdown_.store(false, std::memory_order_relaxed);
    generation_.store(0, std::memory_order_relaxed);

    workers_.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

~JobSystem() {
    shutdown_.store(true, std::memory_order_release);
    // Bump generation so sleeping workers wake up and see shutdown
    generation_.fetch_add(1, std::memory_order_release);
    for (auto& w : workers_) w.join();
}

JobSystem(const JobSystem&)            = delete;
JobSystem& operator=(const JobSystem&) = delete;

// Parallel for: splits [0, count) into chunks and dispatches to all threads.
// Main thread participates in work. Blocks until complete.
void parallel_for(std::size_t count, std::size_t chunk_size,
                  TaskFn func, void* ctx) noexcept {
    if (count == 0) return;

    const std::size_t num_chunks =
        (count + chunk_size - 1) / chunk_size;

    // Publish task parameters (non-atomic, guarded by release below)
    task_fn_    = func;
    task_ctx_   = ctx;
    task_count_ = count;
    task_chunk_ = chunk_size;

    // Reset work counter
    work_index_.store(0, std::memory_order_relaxed);
    remaining_.store(static_cast<std::int64_t>(num_chunks),
                     std::memory_order_relaxed);

    // Release fence: all stores above become visible after this
    generation_.fetch_add(1, std::memory_order_release);

    // Main thread also steals work
    consume_chunks();

    // Spin-wait until every chunk is done
    while (remaining_.load(std::memory_order_acquire) > 0) {
        #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
        #else
            std::this_thread::yield();
        #endif
    }
}
```

private:
void consume_chunks() noexcept {
for (;;) {
const std::size_t idx =
work_index_.fetch_add(1, std::memory_order_relaxed);
const std::size_t begin = idx * task_chunk_;
if (begin >= task_count_) break;
const std::size_t end =
(begin + task_chunk_ < task_count_) ? begin + task_chunk_
: task_count_;
task_fn_(begin, end, task_ctx_);
remaining_.fetch_sub(1, std::memory_order_release);
}
}

```
void worker_loop() noexcept {
    std::uint64_t local_gen = generation_.load(std::memory_order_relaxed);
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::uint64_t cur =
            generation_.load(std::memory_order_acquire);
        if (cur != local_gen) {
            local_gen = cur;
            consume_chunks();
        } else {
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #else
                std::this_thread::yield();
            #endif
        }
    }
}

// --- Each hot atomic on its own cache line to prevent false sharing ---

alignas(kCacheLine) std::atomic<bool>         shutdown_{false};
alignas(kCacheLine) std::atomic<std::uint64_t> generation_{0};
alignas(kCacheLine) std::atomic<std::size_t>   work_index_{0};
alignas(kCacheLine) std::atomic<std::int64_t>  remaining_{0};

// Task parameters — written by dispatcher before generation release fence,
// read by workers after generation acquire load.  No atomics needed.
TaskFn       task_fn_    = nullptr;
void*        task_ctx_   = nullptr;
std::size_t  task_count_ = 0;
std::size_t  task_chunk_ = 0;

std::vector<std::thread> workers_;   // allocated once at construction
```

};

// =============================================================================
// 3. ECS Core — Compile-time, zero-overhead component access
// =============================================================================

// — Concept: an SoA component must be allocatable from an arena —
template <typename T>
concept SoAComponent = requires(T t, ArenaAllocator& a, std::size_t n) {
{ t.allocate(a, n) } -> std::same_as<void>;
} && std::is_default_constructible_v<T>;

// — Compile-time membership check —
template <typename T, typename… Ts>
concept ContainedIn = (std::same_as<T, Ts> || …);

// — Compile-time type index within a pack —
template <typename T, typename… Ts>
struct TypeIndex;

template <typename T, typename… Rest>
struct TypeIndex<T, T, Rest…>
: std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename… Rest>
struct TypeIndex<T, U, Rest…>
: std::integral_constant<std::size_t, 1 + TypeIndex<T, Rest…>::value> {};

template <typename T, typename… Ts>
inline constexpr std::size_t type_index_v = TypeIndex<T, Ts…>::value;

// — World: owns all component storage —
template <SoAComponent… Cs>
class World {
public:
static constexpr std::size_t component_count = sizeof…(Cs);

```
void init(ArenaAllocator& arena, std::size_t entity_count) noexcept {
    count_ = entity_count;
    std::apply([&](auto&... c) {
        (c.allocate(arena, entity_count), ...);
    }, storage_);
}

template <SoAComponent C>
    requires ContainedIn<C, Cs...>
[[nodiscard]] C& get() noexcept {
    return std::get<type_index_v<C, Cs...>>(storage_);
}

template <SoAComponent C>
    requires ContainedIn<C, Cs...>
[[nodiscard]] const C& get() const noexcept {
    return std::get<type_index_v<C, Cs...>>(storage_);
}

[[nodiscard]] std::size_t size() const noexcept { return count_; }
```

private:
std::tuple<Cs…> storage_;
std::size_t       count_ = 0;
};

// =============================================================================
// 4. SoA Components — true Structure-of-Arrays for perfect vectorization
//    Each field is a separate contiguous, cache-line-aligned array.
// =============================================================================

struct PositionSoA {
float* x = nullptr;
float* y = nullptr;
float* z = nullptr;

```
void allocate(ArenaAllocator& arena, std::size_t n) noexcept {
    x = arena.alloc_array<float>(n);
    y = arena.alloc_array<float>(n);
    z = arena.alloc_array<float>(n);
}
```

};

struct VelocitySoA {
float* x = nullptr;
float* y = nullptr;
float* z = nullptr;

```
void allocate(ArenaAllocator& arena, std::size_t n) noexcept {
    x = arena.alloc_array<float>(n);
    y = arena.alloc_array<float>(n);
    z = arena.alloc_array<float>(n);
}
```

};

// Verify concepts at compile time
static_assert(SoAComponent<PositionSoA>);
static_assert(SoAComponent<VelocitySoA>);

// =============================================================================
// 5. Movement System — vectorization-friendly kernel
// =============================================================================

struct MovementCtx {
float*       px;
float*       py;
float*       pz;
const float* vx;
const float* vy;
const float* vz;
float        dt;
};

// Hot inner kernel: separate loops per axis for optimal auto-vectorization.
// Each loop touches exactly 2 streams → minimal register pressure, maximum
// throughput on wide SIMD (AVX2: 8 floats/cycle per loop).
// **restrict** guarantees no aliasing → compiler emits vmovups/vfmadd packs.
static void movement_kernel(std::size_t begin, std::size_t end,
void* raw_ctx) noexcept {
const auto* ctx = static_cast<const MovementCtx*>(raw_ctx);

```
float* __restrict__       px = ctx->px;
float* __restrict__       py = ctx->py;
float* __restrict__       pz = ctx->pz;
const float* __restrict__ vx = ctx->vx;
const float* __restrict__ vy = ctx->vy;
const float* __restrict__ vz = ctx->vz;
const float dt = ctx->dt;

// Three independent loops — each one is a trivial reduction with no
// cross-iteration dependency. GCC/Clang vectorize with AVX FMA packs.
#pragma GCC ivdep
for (std::size_t i = begin; i < end; ++i)
    px[i] += vx[i] * dt;

#pragma GCC ivdep
for (std::size_t i = begin; i < end; ++i)
    py[i] += vy[i] * dt;

#pragma GCC ivdep
for (std::size_t i = begin; i < end; ++i)
    pz[i] += vz[i] * dt;
```

}

// =============================================================================
// 6. Main — setup, simulate, benchmark
// =============================================================================

int main() {
using Clock = std::chrono::high_resolution_clock;

```
// ---- Arena: pre-allocate all simulation memory ----
// 6 arrays × 100k floats × 4 bytes ≈ 2.4 MB; 8 MB gives headroom
constexpr std::size_t kArenaSize = 8u * 1024u * 1024u;
ArenaAllocator arena(kArenaSize);

// ---- ECS World ----
World<PositionSoA, VelocitySoA> world;
world.init(arena, kEntityCount);

auto& pos = world.get<PositionSoA>();
auto& vel = world.get<VelocitySoA>();

// ---- Initialize entities with deterministic random data ----
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < kEntityCount; ++i) {
        pos.x[i] = pos_dist(rng);
        pos.y[i] = pos_dist(rng);
        pos.z[i] = pos_dist(rng);
        vel.x[i] = vel_dist(rng);
        vel.y[i] = vel_dist(rng);
        vel.z[i] = vel_dist(rng);
    }
}

// ---- Job System ----
JobSystem jobs;

// Snapshot entity 0 before simulation
const float x0_before = pos.x[0];
const float y0_before = pos.y[0];
const float z0_before = pos.z[0];

// ---- Simulation loop ----
MovementCtx ctx{
    pos.x, pos.y, pos.z,
    vel.x, vel.y, vel.z,
    kDeltaTime
};

const auto t_start = Clock::now();

for (std::size_t iter = 0; iter < kSimIterations; ++iter) {
    jobs.parallel_for(kEntityCount, kChunkSize, movement_kernel, &ctx);
}

const auto t_end = Clock::now();

// ---- Results ----
const double total_us =
    std::chrono::duration<double, std::micro>(t_end - t_start).count();
const double per_iter_us = total_us / static_cast<double>(kSimIterations);

std::printf("===== ECS Particle Simulation =====\n");
std::printf("Entities       : %zu\n", kEntityCount);
std::printf("Iterations     : %zu\n", kSimIterations);
std::printf("Worker threads : %u (+main)\n",
            std::thread::hardware_concurrency() > 1
                ? std::thread::hardware_concurrency() - 1 : 1u);
std::printf("Arena used     : %.2f KB / %.2f KB\n",
            static_cast<double>(arena.used()) / 1024.0,
            static_cast<double>(arena.capacity()) / 1024.0);
std::printf("-----------------------------------\n");
std::printf("Total time     : %.2f us (%.4f ms)\n",
            total_us, total_us / 1000.0);
std::printf("Per iteration  : %.2f us\n", per_iter_us);
std::printf("Throughput     : %.2f M entities/s\n",
            (static_cast<double>(kEntityCount) * kSimIterations)
                / (total_us));
std::printf("-----------------------------------\n");
std::printf("Entity 0 pos   : (%.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f)\n",
            static_cast<double>(x0_before),
            static_cast<double>(y0_before),
            static_cast<double>(z0_before),
            static_cast<double>(pos.x[0]),
            static_cast<double>(pos.y[0]),
            static_cast<double>(pos.z[0]));

// ---- Validation: Δpos should equal vel * dt * iterations ----
const float expected_x = x0_before + vel.x[0] * kDeltaTime * kSimIterations;
const float err = pos.x[0] - expected_x;
std::printf("Validation err : %.9f (fp accumulation)\n",
            static_cast<double>(err));

return 0;
```

}
