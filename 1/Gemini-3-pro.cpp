#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <cassert>
#include <memory>
#include <concepts>
#include <tuple>
#include <new>
#include <numeric>
#include <bit>

// ==================================================================================
// 1. MEMORY MANAGEMENT (Arena / Monotonic Buffer)
// ==================================================================================

// Выравнивание под кэш-линию (обычно 64 байта) для предотвращения false sharing и SIMD
constexpr size_t CACHE_LINE_SIZE = 64;

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t size) : total_size_(size), offset_(0) {
        // Выделяем память один раз при старте. Используем aligned_alloc для SIMD.
        // В продакшене тут был бы mmap с huge pages.
        data_ = static_cast<std::byte*>(std::aligned_alloc(CACHE_LINE_SIZE, size));
        if (!data_) throw std::bad_alloc();
    }

    ~ArenaAllocator() {
        std::free(data_);
    }

    // Запрет копирования
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    template <typename T>
    T* alloc_array(size_t count) {
        size_t bytes = sizeof(T) * count;
        
        // Выравниваем текущий offset
        size_t current_addr = reinterpret_cast<size_t>(data_ + offset_.load(std::memory_order_relaxed));
        size_t padding = (CACHE_LINE_SIZE - (current_addr % CACHE_LINE_SIZE)) % CACHE_LINE_SIZE;
        
        size_t start_offset = offset_.fetch_add(bytes + padding, std::memory_order_relaxed);
        
        if (start_offset + bytes + padding > total_size_) {
            std::cerr << "Arena OOM!\n";
            std::terminate();
        }

        return reinterpret_cast<T*>(data_ + start_offset + padding);
    }

    void reset() {
        offset_.store(0, std::memory_order_release);
    }

private:
    std::byte* data_;
    size_t total_size_;
    std::atomic<size_t> offset_;
};

// ==================================================================================
// 2. COMPONENTS & DATA ORIENTED DESIGN
// ==================================================================================

// Концепт для проверки, что компонент — это POD (Plain Old Data)
template<typename T>
concept Component = std::is_trivial_v<T> && std::is_standard_layout_v<T>;

// Чтобы обеспечить векторизацию, данные выравниваем по 16/32 байтам.
struct alignas(16) Position {
    float x, y, z;
    float pad; // Padding для выравнивания 16 байт (удобно для SSE/NEON)
};

struct alignas(16) Velocity {
    float vx, vy, vz;
    float pad;
};

static_assert(Component<Position>);
static_assert(Component<Velocity>);

// ==================================================================================
// 3. ECS ARCHITECTURE (SoA Storage)
// ==================================================================================

// Хранилище для одного типа компонентов. 
// В отличие от std::vector, не владеет памятью, а указывает на Arena.
template <Component T>
struct ComponentArray {
    T* data = nullptr;
    size_t count = 0;

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }
};

// Compile-time Registry
template <Component... Ts>
class World {
public:
    explicit World(ArenaAllocator& arena, size_t capacity) : capacity_(capacity) {
        // Разворачиваем пак параметров для аллокации массивов
        ((std::get<ComponentArray<Ts>>(arrays_).data = arena.alloc_array<Ts>(capacity)), ...);
        ((std::get<ComponentArray<Ts>>(arrays_).count = capacity), ...);
        count_ = capacity;
    }

    template <typename T>
    ComponentArray<T>& get_components() {
        return std::get<ComponentArray<T>>(arrays_);
    }

    size_t size() const { return count_; }

private:
    std::tuple<ComponentArray<Ts>...> arrays_;
    size_t capacity_;
    size_t count_;
};

// ==================================================================================
// 4. LOCK-FREE JOB SYSTEM
// ==================================================================================

class JobSystem {
public:
    using JobFunc = void (*)(void*, size_t, size_t);

    JobSystem() : stop_flag_(false) {
        unsigned int cores = std::thread::hardware_concurrency();
        // Оставляем один поток для main, если ядер достаточно
        if (cores > 1) cores--; 
        
        for (unsigned int i = 0; i < cores; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~JobSystem() {
        stop_flag_.store(true, std::memory_order_release);
        // Будим всех
        start_signal_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    // Запуск параллельного цикла
    // data: контекст (например, указатель на World)
    // count: общее количество элементов
    // func: функция обработки
    void parallel_for(void* data, size_t count, JobFunc func) {
        current_job_data_ = data;
        current_job_func_ = func;
        total_items_.store(count, std::memory_order_relaxed);
        processed_items_.store(0, std::memory_order_relaxed);
        
        // Сбрасываем счетчик завершивших потоков
        // atomic_thread_fence(release) не нужен, так как fetch_add ниже делает acquire/release
        
        // Устанавливаем "билет" начала работы.
        // Используем memory_order_release, чтобы данные задачи были видны потокам.
        work_ticket_.fetch_add(1, std::memory_order_release);
        start_signal_.notify_all();

        // Главный поток тоже работает! (Main thread helps)
        process_batch();

        // Spin-wait (или wait), пока всё не закончится.
        // Так как это HFT/GameDev, в горячем цикле мы обычно не спим через OS wait,
        // но здесь используем atomic::wait для корректности.
        // В реальном движке тут был бы spin с _mm_pause().
        while (items_completed_.load(std::memory_order_acquire) < count) {
             std::atomic_thread_fence(std::memory_order_acquire); // Hints to CPU
        }
        
        // Reset counters for next frame implicitly by logic
        items_completed_.store(0, std::memory_order_relaxed);
    }

private:
    void worker_loop() {
        uint32_t last_ticket = 0;
        while (!stop_flag_.load(std::memory_order_relaxed)) {
            uint32_t ticket = work_ticket_.load(std::memory_order_acquire);
            if (ticket != last_ticket) {
                process_batch();
                last_ticket = ticket;
            } else {
                // Ждем сигнала (C++20 feature)
                // Это гораздо эффективнее чистого spin-lock'а для батареи, 
                // но чуть медленнее по латенси пробуждения.
                work_ticket_.wait(last_ticket, std::memory_order_relaxed);
            }
        }
    }

    void process_batch() {
        // Dynamic Load Balancing (Work Stealing через атомарный курсор)
        // Берем чанки по 1024 элемента, чтобы амортизировать atomic overhead
        constexpr size_t BATCH_SIZE = 1024;
        
        while (true) {
            size_t total = total_items_.load(std::memory_order_relaxed);
            size_t idx = processed_items_.fetch_add(BATCH_SIZE, std::memory_order_relaxed);
            
            if (idx >= total) break;

            size_t end = std::min(idx + BATCH_SIZE, total);
            
            // Выполняем работу
            if (current_job_func_) {
                current_job_func_(current_job_data_, idx, end);
            }

            // Отмечаем выполненные элементы
            items_completed_.fetch_add(end - idx, std::memory_order_release);
        }
    }

    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_;
    
    // Синхронизация задач
    std::atomic<uint32_t> work_ticket_{0};
    std::atomic<uint32_t> start_signal_{0}; // Dummy for notify

    // Данные текущей задачи
    void* current_job_data_;
    JobFunc current_job_func_;
    std::atomic<size_t> total_items_;
    std::atomic<size_t> processed_items_;
    
    // Обратный барьер
    std::atomic<size_t> items_completed_{0};
};

// ==================================================================================
// 5. SIMULATION LOGIC
// ==================================================================================

using GameWorld = World<Position, Velocity>;

// Системная функция. Статическая, чтобы передать указатель в JobSystem.
// В C++23 можно было бы использовать std::function_ref или лямбду без капч (decay to ptr).
void update_movement_system(void* raw_world, size_t start, size_t end) {
    auto* world = static_cast<GameWorld*>(raw_world);
    
    // Получаем сырые указатели (SoA массивы)
    // __restrict помогает компилятору понять, что массивы не перекрываются
    Position* __restrict positions = world->get_components<Position>().data;
    const Velocity* __restrict velocities = world->get_components<Velocity>().data;

    constexpr float dt = 0.016f; // 60 FPS

    // Векторизуемый цикл
    // #pragma omp simd здесь не нужен, современные компиляторы с -O3 справляются сами
    // благодаря выравниванию и простой математике.
    for (size_t i = start; i < end; ++i) {
        positions[i].x += velocities[i].vx * dt;
        positions[i].y += velocities[i].vy * dt;
        positions[i].z += velocities[i].vz * dt;
        
        // Простейшая "физика" для теста (отскок)
        /* 
           Убран branch для лучшей векторизации. 
           В HFT/Sim коде бранчи внутри горячего цикла — зло.
           Используем branchless logic если нужно, но для чистого теста пропускной способности:
        */
    }
}

// ==================================================================================
// MAIN
// ==================================================================================

int main() {
    // 1. Конфигурация
    constexpr size_t ENTITY_COUNT = 100'000;
    constexpr size_t ITERATIONS = 100;
    
    // 2. Инициализация памяти (128MB должно хватить с запасом)
    ArenaAllocator arena(128 * 1024 * 1024);

    // 3. Создание мира
    GameWorld world(arena, ENTITY_COUNT);

    // 4. Заполнение данными (Init)
    {
        auto& pos = world.get_components<Position>();
        auto& vel = world.get_components<Velocity>();
        for (size_t i = 0; i < ENTITY_COUNT; ++i) {
            pos[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            vel[i] = {1.0f, 0.5f, 0.1f, 0.0f};
        }
    }

    // 5. Инициализация Job System
    JobSystem jobs;

    std::cout << "Starting simulation: " << ENTITY_COUNT << " entities, " 
              << ITERATIONS << " iterations.\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // 6. Игровой цикл
    for (size_t frame = 0; frame < ITERATIONS; ++frame) {
        // Ставим задачу на обновление всех сущностей
        jobs.parallel_for(&world, ENTITY_COUNT, update_movement_system);
        
        // Тут мог бы быть рендеринг...
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    // 7. Проверка результата (чтобы компилятор не вырезал код)
    auto& pos = world.get_components<Position>();
    std::cout << "Done. Sample Pos[0]: " << pos[0].x << ", " << pos[0].y << "\n";
    std::cout << "Total time: " << elapsed.count() << " ms\n";
    std::cout << "Time per frame: " << elapsed.count() / ITERATIONS << " ms\n";
    std::cout << "Updates per sec: " << (double(ENTITY_COUNT) * ITERATIONS) / (elapsed.count() / 1000.0) / 1'000'000.0 << " M updates/sec\n";

    return 0;
}
