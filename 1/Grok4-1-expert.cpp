#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

// Custom monotonic arena allocator (lock-free for multi-threaded allocations)
class MonotonicArena {
public:
    explicit MonotonicArena(size_t size) {
        buffer_ = static_cast<char*>(std::aligned_alloc(CACHE_LINE_SIZE, size));
        if (!buffer_) {
            throw std::bad_alloc{};
        }
        capacity_ = size;
        offset_.store(0, std::memory_order_relaxed);
    }

    ~MonotonicArena() {
        std::free(buffer_);
    }

    MonotonicArena(const MonotonicArena&) = delete;
    MonotonicArena& operator=(const MonotonicArena&) = delete;

    void* allocate(size_t bytes, size_t alignment = CACHE_LINE_SIZE) {
        size_t current = offset_.load(std::memory_order_relaxed);
        size_t aligned_offset;
        size_t new_offset;
        do {
            aligned_offset = (current + alignment - 1) & ~(alignment - 1);
            new_offset = aligned_offset + bytes;
            if (new_offset > capacity_) {
                return nullptr; // Out of memory (no throw for performance)
            }
        } while (!offset_.compare_exchange_weak(current, new_offset,
                                                std::memory_order_release,
                                                std::memory_order_relaxed));
        return buffer_ + aligned_offset;
    }

private:
    char* buffer_ = nullptr;
    size_t capacity_ = 0;
    std::atomic<size_t> offset_{0};
};

// Custom allocator for std::vector using MonotonicArena
template <typename T>
class ArenaAllocator {
public:
    using value_type = T;

    explicit ArenaAllocator(MonotonicArena& arena) : arena_(arena) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other) noexcept : arena_(other.arena_) {}

    T* allocate(size_t n) {
        void* ptr = arena_.allocate(n * sizeof(T), alignof(T));
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T*, size_t) noexcept {
        // No-op: monotonic, no dealloc
    }

    bool operator==(const ArenaAllocator&) const noexcept { return true; }
    bool operator!=(const ArenaAllocator&) const noexcept { return false; }

private:
    MonotonicArena& arena_;
};

// Concept for components: must be trivially copyable for memcpy safety and performance
template <typename T>
concept Component = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

// Component storage: Dense vector for data, sparse for entity mapping (SoA, cache-friendly)
template <Component C>
class ComponentStorage {
public:
    explicit ComponentStorage(MonotonicArena& arena, size_t max_entities)
        : dense_(ArenaAllocator<C>(arena)),
          entity_to_dense_(max_entities, INVALID_INDEX, ArenaAllocator<size_t>(arena)),
          dense_to_entity_(ArenaAllocator<size_t>(arena)) {
        dense_.reserve(max_entities);
        dense_to_entity_.reserve(max_entities);
    }

    size_t add(size_t entity, C&& component) {
        if (entity_to_dense_[entity] != INVALID_INDEX) {
            return INVALID_INDEX; // Already exists
        }
        size_t index = dense_.size();
        dense_.push_back(std::move(component));
        dense_to_entity_.push_back(entity);
        entity_to_dense_[entity] = index;
        return index;
    }

    C& get(size_t entity) {
        size_t index = entity_to_dense_[entity];
        return dense_[index];
    }

    const C& get(size_t entity) const {
        size_t index = entity_to_dense_[entity];
        return dense_[index];
    }

    size_t size() const { return dense_.size(); }

    C* data() { return dense_.data(); }
    const C* data() const { return dense_.data(); }

    size_t dense_index(size_t entity) const { return entity_to_dense_[entity]; }

private:
    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
    std::vector<C, ArenaAllocator<C>> dense_;
    std::vector<size_t, ArenaAllocator<size_t>> entity_to_dense_;
    std::vector<size_t, ArenaAllocator<size_t>> dense_to_entity_;
};

// World: Manages entities and component storages (compile-time via tuple)
template <Component... Cs>
class World {
public:
    explicit World(size_t max_entities) : max_entities_(max_entities), next_entity_(0) {
        // Pre-allocate arena: estimate size (positions + velocities + mappings)
        size_t estimated_size = max_entities * (sizeof(Cs) + ... + 0) +
                                max_entities * (sizeof(size_t) * 2 * sizeof...(Cs)) +
                                CACHE_LINE_SIZE * 10; // Padding
        arena_ = std::make_unique<MonotonicArena>(estimated_size);
        init_storages(std::index_sequence_for<Cs...>{});
    }

    size_t create_entity() {
        if (next_entity_ >= max_entities_) {
            return INVALID_ENTITY;
        }
        return next_entity_++;
    }

    template <Component C>
    void add_component(size_t entity, C&& component) {
        get_storage<C>().add(entity, std::move(component));
    }

    template <Component C>
    C& get_component(size_t entity) {
        return get_storage<C>().get(entity);
    }

    template <Component C>
    const C* component_data() const {
        return get_storage<C>().data();
    }

    template <Component C>
    C* component_data() {
        return get_storage<C>().data();
    }

    template <Component C>
    size_t component_size() const {
        return get_storage<C>().size();
    }

private:
    static constexpr size_t INVALID_ENTITY = static_cast<size_t>(-1);
    size_t max_entities_;
    size_t next_entity_;
    std::unique_ptr<MonotonicArena> arena_;
    std::tuple<ComponentStorage<Cs>...> storages_;

    template <size_t... Is>
    void init_storages(std::index_sequence<Is...>) {
        (std::get<Is>(storages_) = ComponentStorage<Cs>(*arena_, max_entities_), ...);
    }

    template <Component C>
    ComponentStorage<C>& get_storage() {
        return std::get<ComponentStorage<C>>(storages_);
    }

    template <Component C>
    const ComponentStorage<C>& get_storage() const {
        return std::get<ComponentStorage<C>>(storages_);
    }
};

// Simple lock-free job system using std::thread and atomic counter
class JobSystem {
public:
    explicit JobSystem(size_t num_threads) : num_threads_(num_threads), threads_(num_threads) {}

    ~JobSystem() {
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    template <typename Func>
    void dispatch(size_t num_jobs, Func&& job_func) {
        if (num_jobs == 0) return;

        completed_.store(0, std::memory_order_relaxed);
        size_t jobs_per_thread = num_jobs / num_threads_;
        size_t remainder = num_jobs % num_threads_;

        size_t start = 0;
        for (size_t i = 0; i < num_threads_; ++i) {
            size_t end = start + jobs_per_thread + (i < remainder ? 1 : 0);
            threads_[i] = std::thread([this, start, end, &job_func]() {
                for (size_t j = start; j < end; ++j) {
                    job_func(j);
                }
                completed_.fetch_add(1, std::memory_order_release);
            });
            start = end;
        }

        // Wait lock-free
        size_t done;
        do {
            done = completed_.load(std::memory_order_acquire);
        } while (done < num_threads_);
    }

private:
    size_t num_threads_;
    std::vector<std::thread> threads_;
    std::atomic<size_t> completed_{0};
};

// Components for particle simulation
struct Position {
    float x, y, z;
};

struct Velocity {
    float vx, vy, vz;
};

static_assert(Component<Position>);
static_assert(Component<Velocity>);

// Main simulation
int main() {
    constexpr size_t NUM_ENTITIES = 100'000;
    constexpr size_t NUM_ITERATIONS = 100;
    constexpr size_t NUM_THREADS = std::thread::hardware_concurrency();

    World<Position, Velocity> world(NUM_ENTITIES);
    JobSystem job_system(NUM_THREADS);

    // Create entities with components (parallel creation not needed, but could be)
    for (size_t i = 0; i < NUM_ENTITIES; ++i) {
        size_t entity = world.create_entity();
        world.add_component(entity, Position{0.0f, 0.0f, 0.0f});
        world.add_component(entity, Velocity{1.0f, 1.0f, 1.0f});
    }

    // Simulation loop
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Assume all entities have both components, size is same
        size_t num_particles = world.component_size<Position>();

        // Dispatch jobs: each job updates a chunk
        size_t chunk_size = (num_particles + NUM_THREADS - 1) / NUM_THREADS;
        job_system.dispatch(NUM_THREADS, [&](size_t thread_id) {
            size_t start = thread_id * chunk_size;
            size_t end = std::min(start + chunk_size, num_particles);

            auto* positions = world.component_data<Position>();
            auto* velocities = world.component_data<Velocity>();

            // Vectorizable loops (simple arithmetic, no dependencies)
            for (size_t i = start; i < end; ++i) {
                positions[i].x += velocities[i].vx;
            }
            for (size_t i = start; i < end; ++i) {
                positions[i].y += velocities[i].vy;
            }
            for (size_t i = start; i < end; ++i) {
                positions[i].z += velocities[i].vz;
            }
        });
    }

    // Optional: print one for verification
    std::cout << "Final position of entity 0: "
              << world.get_component<Position>(0).x << ", "
              << world.get_component<Position>(0).y << ", "
              << world.get_component<Position>(0).z << std::endl;

    return 0;
}
