// ecs_particles.cpp
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>
#include <new>
#include <cassert>

// ---------------------- Arena Allocator (Monotonic) ----------------------
// Single preallocated buffer. allocate() is cache-line (64B) aligned.
// No deallocate() for individual allocations. Thread-safe only during init.
class Arena {
    uint8_t* base_;
    size_t capacity_;
    size_t offset_;
    static constexpr size_t CACHELINE = 64;
public:
    Arena(size_t capacity_bytes)
      : base_(nullptr), capacity_(capacity_bytes), offset_(0)
    {
        // allocate aligned to CACHELINE
        // operator new with alignment (C++17) is used here (allowed at init time)
        base_ = static_cast<uint8_t*>(::operator new(capacity_bytes, std::align_val_t{CACHELINE}));
    }
    ~Arena() {
        if (base_) ::operator delete(base_, std::align_val_t{CACHELINE});
    }
    // non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // allocate with given size and alignment (at least CACHELINE)
    void* allocate(size_t size, size_t alignment = CACHELINE) {
        alignment = std::max(alignment, CACHELINE);
        size_t current = offset_;
        // align up
        size_t mis = (reinterpret_cast<uintptr_t>(base_ + current)) & (alignment - 1);
        if (mis) current += (alignment - mis);
        if (current + size > capacity_) {
            return nullptr; // out-of-memory
        }
        void* ptr = base_ + current;
        offset_ = current + size;
        // guarantee alignment
        assert((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0);
        return ptr;
    }

    // get used bytes (for debugging)
    size_t used() const noexcept { return offset_; }
};

// ---------------------- Component declarations (tags + SoA traits) ----------------------
// We'll define Position and Velocity as components with two float fields each (x,y)
// To adhere to SoA we store each field in its own contiguous array.

struct Position {}; // tag
struct Velocity {}; // tag

// ComponentTraits: for each component type, define number of fields and their type (float)
template<typename C> struct ComponentTraits;

template<> struct ComponentTraits<Position> {
    static constexpr std::size_t fields = 2; // x, y
    using field_t = float;
    static constexpr const char* names[fields] = { "pos_x", "pos_y" };
};

template<> struct ComponentTraits<Velocity> {
    static constexpr std::size_t fields = 2; // vx, vy
    using field_t = float;
    static constexpr const char* names[fields] = { "vel_x", "vel_y" };
};

// ---------------------- Generic SoA Component Storage ----------------------
template<typename Component>
class ComponentStorage {
    using Traits = ComponentTraits<Component>;
    using FieldT = typename Traits::field_t;
    std::size_t entity_count_;
    // store pointers to each field array
    FieldT** fields_; // pointer to array of FieldT* of length Traits::fields
public:
    ComponentStorage() : entity_count_(0), fields_(nullptr) {}
    ~ComponentStorage() {
        // memory for fields_ is owned by Arena; do not free here.
    }
    // non-copyable
    ComponentStorage(const ComponentStorage&) = delete;
    ComponentStorage& operator=(const ComponentStorage&) = delete;

    // allocate storage for N entities from arena in a single large block:
    // layout: pointer table (FieldT* fields[fields]) followed by
    //         fields[0] data array (N*FieldT), fields[1] data array, ...
    bool init(Arena& arena, std::size_t entity_count) {
        entity_count_ = entity_count;
        constexpr std::size_t F = Traits::fields;
        // We want: a contiguous block for pointers (F * pointer) and F arrays of entity_count FieldT.
        size_t ptrs_size = sizeof(FieldT*) * F;
        size_t data_size = sizeof(FieldT) * entity_count_ * F;
        // Align data arrays to CACHELINE for vectorization
        size_t total = ptrs_size + data_size + 64;
        void* mem = arena.allocate(total, 64);
        if (!mem) return false;
        // place pointer table at mem
        uint8_t* p = static_cast<uint8_t*>(mem);
        fields_ = reinterpret_cast<FieldT**>(p);
        // data start aligned to 64
        uint8_t* data_start = p + ptrs_size;
        uintptr_t mis = reinterpret_cast<uintptr_t>(data_start) & (size_t(64)-1);
        if (mis) data_start += (64 - mis);
        // set pointers
        for (std::size_t i = 0; i < F; ++i) {
            fields_[i] = reinterpret_cast<FieldT*>(data_start + i * sizeof(FieldT) * entity_count_);
            // zero-initialize
            std::memset(fields_[i], 0, sizeof(FieldT) * entity_count_);
            // ensure alignment
            assert((reinterpret_cast<uintptr_t>(fields_[i]) & (uintptr_t(64)-1)) == 0);
        }
        return true;
    }

    // access pointer to field index f
    FieldT* field_ptr(std::size_t f) noexcept {
        return fields_[f];
    }
    const FieldT* field_ptr(std::size_t f) const noexcept {
        return fields_[f];
    }
    std::size_t size() const noexcept { return entity_count_; }

    // helper getters for common case of 2 fields (x,y)
    FieldT* x_ptr() noexcept { static_assert(Traits::fields >= 1); return field_ptr(0); }
    FieldT* y_ptr() noexcept { static_assert(Traits::fields >= 2); return field_ptr(1); }
};

// ---------------------- ECS with compile-time registration ----------------------
template<typename... Components>
class ECS {
    static_assert(sizeof...(Components) > 0, "Need at least one component type.");
    std::tuple<ComponentStorage<Components>...> storages_;
    std::size_t entity_count_;
    Arena& arena_;
public:
    ECS(Arena& arena, std::size_t entity_count) : storages_(), entity_count_(entity_count), arena_(arena) {
        // Initialize each storage
        (std::get<ComponentStorage<Components>>(storages_).init(arena_, entity_count_), ...);
    }

    // Get storage for component C
    template<typename C>
    ComponentStorage<C>& get() {
        return std::get<ComponentStorage<C>>(storages_);
    }

    std::size_t entity_count() const noexcept { return entity_count_; }
};

// ---------------------- Simple lock-free Job System (chunk-based) ----------------------
// Model: main thread divides work into simple independent chunks.
// Workers fetch next chunk index with an atomic fetch_add (lock-free).
struct JobSystem {
    unsigned worker_count_;
    JobSystem(unsigned workers = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1u)
      : worker_count_(workers > 0 ? workers : 1u) {}

    // Run a function on [0..num_chunks) possibly in parallel.
    // The user function must accept (chunk_index).
    template<typename Func>
    void parallel_for(std::size_t num_chunks, Func&& f) const {
        if (num_chunks == 0) return;
        // Atomic counter of next chunk
        std::atomic<std::size_t> next_chunk;
        next_chunk.store(0, std::memory_order_release);

        // worker lambda
        auto worker = [&next_chunk,&f,num_chunks]() {
            while (true) {
                std::size_t idx = next_chunk.fetch_add(1, std::memory_order_acq_rel);
                if (idx >= num_chunks) break;
                f(idx);
            }
        };

        // if only one worker, run inline for best perf
        if (worker_count_ == 1) {
            worker();
            return;
        }

        // spawn threads
        std::vector<std::thread> threads;
        threads.reserve(worker_count_);
        for (unsigned i = 0; i < worker_count_ - 1; ++i) {
            threads.emplace_back(worker);
        }
        // run one worker in main thread
        worker();
        for (auto& t : threads) t.join();
    }
};

// ---------------------- Simulation: 100k entities, Position+Velocity ----------------------
int main() {
    constexpr std::size_t ENTITIES = 100000;
    constexpr std::size_t ITERATIONS = 100;
    constexpr std::size_t CHUNK_SIZE = 256; // tuned for cache and vectorization

    // Preallocate arena: enough for Position and Velocity
    // Each component: fields * ENTITIES * sizeof(float) + pointers table; we allocate generous space
    size_t approx_bytes = (sizeof(float) * 2 * ENTITIES * 2) + 1024*1024; // safety
    Arena arena(approx_bytes + 65536);

    // Create ECS with Position and Velocity components
    ECS<Position, Velocity> ecs(arena, ENTITIES);

    // Get pointers to SoA arrays
    auto& pos_store = ecs.get<Position>();
    auto& vel_store = ecs.get<Velocity>();

    float* pos_x = pos_store.x_ptr();
    float* pos_y = pos_store.y_ptr();
    float* vel_x = vel_store.x_ptr();
    float* vel_y = vel_store.y_ptr();

    // Initialize velocities (simple deterministic pattern)
    for (std::size_t i = 0; i < ENTITIES; ++i) {
        vel_x[i] = 0.1f * (1 + (i & 7)); // variety but simple
        vel_y[i] = 0.05f * (1 + ((i >> 3) & 7));
        pos_x[i] = 0.0f;
        pos_y[i] = 0.0f;
    }

    JobSystem jobs; // uses hardware_concurrency

    const std::size_t num_chunks = (ENTITIES + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Main simulation loop: 100 iterations
    for (std::size_t iter = 0; iter < ITERATIONS; ++iter) {
        // Use parallel_for with a lightweight chunk job; job body is vectorization-friendly.
        jobs.parallel_for(num_chunks, [&](std::size_t chunk_idx) {
            std::size_t start = chunk_idx * CHUNK_SIZE;
            std::size_t end = start + CHUNK_SIZE;
            if (end > ENTITIES) end = ENTITIES;

            // Hint to compiler: use raw pointers and simple indexed loop for auto-vectorization.
            float* px = pos_x + start;
            float* py = pos_y + start;
            float* vx = vel_x + start;
            float* vy = vel_y + start;
            std::size_t len = end - start;

            // Simple loop that the compiler can vectorize.
            for (std::size_t i = 0; i < len; ++i) {
                // no UB: arrays are preallocated and in-bounds
                px[i] += vx[i];
                py[i] += vy[i];
            }
        });
        // After iteration we could do synchronization or memory fences if needed.
    }

    // Quick checksum to prevent optimizing everything away
    double sx = 0.0, sy = 0.0;
    for (std::size_t i = 0; i < ENTITIES; ++i) {
        sx += pos_x[i];
        sy += pos_y[i];
    }
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "Done. checksum: " << sx << " , " << sy << "\n";
    return 0;
}
