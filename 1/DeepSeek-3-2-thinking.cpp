#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Константы симуляции
// -----------------------------------------------------------------------------
constexpr size_t ENTITY_COUNT = 100'000;
constexpr int    ITERATIONS   = 100;
constexpr float  DT           = 0.01f;

// -----------------------------------------------------------------------------
// MonotonicBufferResource – линейный аллокатор, выравнивание 64 байта
// -----------------------------------------------------------------------------
class MonotonicBufferResource {
public:
    explicit MonotonicBufferResource(std::size_t size)
        : buffer_{static_cast<std::byte*>(::operator new(size, std::align_val_t(64)))}
        , size_{size}
        , used_{0}
    {
        if (!buffer_) throw std::bad_alloc();
    }

    ~MonotonicBufferResource() {
        ::operator delete(buffer_, std::align_val_t(64));
    }

    MonotonicBufferResource(const MonotonicBufferResource&) = delete;
    MonotonicBufferResource& operator=(const MonotonicBufferResource&) = delete;

    void* allocate(std::size_t bytes, std::size_t alignment) {
        std::size_t space = size_ - used_;
        void* ptr = buffer_ + used_;
        if (std::align(alignment, bytes, ptr, space)) {
            std::size_t offset = static_cast<std::byte*>(ptr) - buffer_;
            used_ = offset + bytes;
            return ptr;
        }
        throw std::bad_alloc();
    }

    void deallocate(void*, std::size_t, std::size_t) noexcept {}

    std::size_t used() const noexcept { return used_; }
    std::size_t size() const noexcept { return size_; }

private:
    std::byte* buffer_;
    std::size_t size_;
    std::size_t used_;
};

// -----------------------------------------------------------------------------
// ArenaAllocator – STL‑совместимый аллокатор поверх MonotonicBufferResource
// -----------------------------------------------------------------------------
template<typename T>
class ArenaAllocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    explicit ArenaAllocator(MonotonicBufferResource& resource) noexcept
        : resource_(&resource) {}

    template<typename U>
    ArenaAllocator(const ArenaAllocator<U>& other) noexcept
        : resource_(other.resource_) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        std::size_t alignment = std::max(alignof(T), std::size_t(64));
        void* ptr = resource_->allocate(bytes, alignment);
        return static_cast<T*>(ptr);
    }

    void deallocate(T*, std::size_t) noexcept {
        // MonotonicBuffer не освобождает память поштучно
    }

    template<typename U>
    bool operator==(const ArenaAllocator<U>& other) const noexcept {
        return resource_ == other.resource_;
    }

    template<typename U>
    bool operator!=(const ArenaAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    MonotonicBufferResource* resource_;
    template<typename U> friend class ArenaAllocator;
};

// -----------------------------------------------------------------------------
// Компоненты (POD)
// -----------------------------------------------------------------------------
struct Position { float x, y, z; };
struct Velocity { float vx, vy, vz; };

// -----------------------------------------------------------------------------
// Компонентные хранилища (Structure-of-Arrays) – явные специализации
// -----------------------------------------------------------------------------
template<typename Component>
struct ComponentStorage; // primary template не определён (compile‑time регистрация)

template<>
struct ComponentStorage<Position> {
    using Alloc = ArenaAllocator<float>;
    std::vector<float, Alloc> x, y, z;

    explicit ComponentStorage(MonotonicBufferResource& res)
        : x(Alloc(res)), y(Alloc(res)), z(Alloc(res)) {}

    void push_back(const Position& p) {
        x.push_back(p.x);
        y.push_back(p.y);
        z.push_back(p.z);
    }

    void reserve(size_t n) {
        x.reserve(n);
        y.reserve(n);
        z.reserve(n);
    }

    size_t size() const noexcept { return x.size(); }
};

template<>
struct ComponentStorage<Velocity> {
    using Alloc = ArenaAllocator<float>;
    std::vector<float, Alloc> vx, vy, vz;

    explicit ComponentStorage(MonotonicBufferResource& res)
        : vx(Alloc(res)), vy(Alloc(res)), vz(Alloc(res)) {}

    void push_back(const Velocity& v) {
        vx.push_back(v.vx);
        vy.push_back(v.vy);
        vz.push_back(v.vz);
    }

    void reserve(size_t n) {
        vx.reserve(n);
        vy.reserve(n);
        vz.reserve(n);
    }

    size_t size() const noexcept { return vx.size(); }
};

// -----------------------------------------------------------------------------
// Концепт Component: тип считается компонентом, если для него определена
// специализация ComponentStorage.
// -----------------------------------------------------------------------------
template<typename T>
concept ComponentConcept = requires {
    typename ComponentStorage<T>;
};

// -----------------------------------------------------------------------------
// ECSWorld – мир сущностей, параметризованный списком компонентов
// -----------------------------------------------------------------------------
template<ComponentConcept... Components>
class ECSWorld {
public:
    explicit ECSWorld(MonotonicBufferResource& resource)
        : storages_{ComponentStorage<Components>(resource)...} {}

    // Создаёт одну сущность, добавляя все переданные компоненты.
    void createEntity(Components... comps) {
        addComponents(std::index_sequence_for<Components...>{}, comps...);
    }

    // Резервирует память под N сущностей во всех хранилищах.
    void reserveEntities(size_t n) {
        (std::get<ComponentStorage<Components>>(storages_).reserve(n), ...);
    }

    // Количество сущностей (предполагается, что все хранилища синхронны).
    size_t entityCount() const noexcept {
        return std::get<0>(storages_).size();
    }

    // Доступ к хранилищу конкретного компонента.
    template<typename Component>
    ComponentStorage<Component>& getStorage() {
        return std::get<ComponentStorage<Component>>(storages_);
    }

private:
    template<std::size_t... Is>
    void addComponents(std::index_sequence<Is...>, Components... comps) {
        (std::get<Is>(storages_).push_back(comps), ...);
    }

    std::tuple<ComponentStorage<Components>...> storages_;
};

// Тип нашего мира: только Position и Velocity.
using MyWorld = ECSWorld<Position, Velocity>;

// -----------------------------------------------------------------------------
// Lock‑free пул потоков с простым распределением диапазонов
// -----------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_{false}
        , ready_{false}
        , remaining_tasks_{0}
        , task_func_{nullptr}
        , task_arg_{nullptr}
    {
        threads_.reserve(num_threads);
        ranges_.resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this, i] { worker(i); });
        }
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_release);
        ready_.store(true, std::memory_order_release);
        for (auto& t : threads_)
            if (t.joinable()) t.join();
    }

    // Запускает функцию func(start, end) параллельно для диапазона [0, count).
    // func должна быть вызываемой с сигнатурой void(size_t, size_t).
    template<typename Func>
    void parallel_for(size_t count, Func&& func) {
        if (count == 0) return;
        const size_t num_threads = threads_.size();
        const size_t chunk_size = (count + num_threads - 1) / num_threads;

        // Раздаём каждому потоку непересекающийся диапазон.
        size_t start = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t end = std::min(start + chunk_size, count);
            ranges_[i] = {start, end};
            start = end;
        }

        // Сохраняем пользовательскую функцию в локальной переменной.
        // Указатель на неё передадим потокам через task_arg_.
        using FuncType = std::decay_t<Func>;
        FuncType local_func = std::forward<Func>(func);
        auto wrapper = [](size_t s, size_t e, void* ctx) {
            (*static_cast<FuncType*>(ctx))(s, e);
        };

        task_func_ = wrapper;
        task_arg_  = &local_func;

        // Сигнал потокам начать работу.
        remaining_tasks_.store(num_threads, std::memory_order_release);
        ready_.store(true, std::memory_order_release);

        // Ожидание завершения всех потоков (активное ожидание допустимо в HFT).
        while (remaining_tasks_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }

        // Сброс флага готовности.
        ready_.store(false, std::memory_order_release);
    }

private:
    void worker(size_t index) {
        while (!stop_.load(std::memory_order_acquire)) {
            // Ждём сигнала ready_.
            while (!ready_.load(std::memory_order_acquire) &&
                   !stop_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (stop_.load(std::memory_order_acquire)) break;

            // Выполняем свою порцию работы.
            auto [start, end] = ranges_[index];
            if (start < end) {
                task_func_(start, end, task_arg_);
            }

            // Сообщаем, что задача выполнена.
            remaining_tasks_.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> threads_;
    std::vector<std::pair<size_t, size_t>> ranges_;
    std::atomic<bool> stop_;
    std::atomic<bool> ready_;
    std::atomic<size_t> remaining_tasks_;
    void (*task_func_)(size_t, size_t, void*);
    void* task_arg_;
};

// -----------------------------------------------------------------------------
// Система обновления позиций (векторизуемый цикл)
// -----------------------------------------------------------------------------
void update_system(ComponentStorage<Position>& pos,
                   ComponentStorage<Velocity>& vel,
                   size_t start, size_t end,
                   float dt) noexcept {
    for (size_t i = start; i < end; ++i) {
        pos.x[i] += vel.vx[i] * dt;
        pos.y[i] += vel.vy[i] * dt;
        pos.z[i] += vel.vz[i] * dt;
    }
}

// -----------------------------------------------------------------------------
// Главная функция
// -----------------------------------------------------------------------------
int main() {
    // 1. Выделяем огромный буфер (200 МБ) – всё размещение будет из него.
    constexpr size_t BUFFER_SIZE = 200 * 1024 * 1024;
    MonotonicBufferResource resource(BUFFER_SIZE);

    // 2. Создаём мир с компонентами Position и Velocity.
    MyWorld world(resource);
    world.reserveEntities(ENTITY_COUNT);

    // 3. Инициализируем сущности случайными значениями.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-1000.0f, 1000.0f);
    std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < ENTITY_COUNT; ++i) {
        Position p{pos_dist(gen), pos_dist(gen), pos_dist(gen)};
        Velocity v{vel_dist(gen), vel_dist(gen), vel_dist(gen)};
        world.createEntity(p, v);
    }

    // 4. Пул потоков (lock‑free).
    ThreadPool pool;

    // 5. Основной цикл симуляции.
    auto& pos_storage = world.getStorage<Position>();
    auto& vel_storage = world.getStorage<Velocity>();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        pool.parallel_for(ENTITY_COUNT, [&](size_t start, size_t end) {
            update_system(pos_storage, vel_storage, start, end, DT);
        });
    }

    std::cout << "Simulation finished.\n"
              << "Memory used: " << resource.used() << " bytes\n";
    return 0;
}
