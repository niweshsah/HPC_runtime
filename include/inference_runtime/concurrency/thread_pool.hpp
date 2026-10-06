#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "inference_runtime/concurrency/blocking_queue.hpp"

namespace inference_runtime {

class ThreadPool {
   public:
    explicit ThreadPool(std::size_t worker_count, std::size_t task_capacity = 64)
        : task_queue_(task_capacity) {
        if (worker_count == 0) {
            throw std::invalid_argument("Thread pool needs at least one worker");
        }
        worker_threads_.reserve(worker_count);
        try {
            for (std::size_t index = 0; index < worker_count; ++index) {
                worker_threads_.emplace_back([this] { executeQueuedTasks(); });
            }
        } catch (...) {
            requestShutdown();
            waitForWorkers();
            throw;
        }
    }

    ~ThreadPool() {
        requestShutdown();
        waitForWorkers();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename Callable, typename... Arguments>
    auto submit(Callable&& callable, Arguments&&... arguments)
        -> std::future<std::invoke_result_t<std::decay_t<Callable>, std::decay_t<Arguments>...>> {
        auto packaged_task =
            packageTask(std::forward<Callable>(callable), std::forward<Arguments>(arguments)...);
        auto result_future = packaged_task->get_future();
        std::function<void()> task = [packaged_task] { (*packaged_task)(); };
        if (!task_queue_.push(std::move(task))) {
            throw std::runtime_error("Thread pool is shutting down");
        }
        return result_future;
    }

    // Network I/O threads use this to avoid blocking when their handler pool is saturated.
    template <typename Callable>
    auto trySubmit(Callable&& callable)
        -> std::optional<std::future<std::invoke_result_t<std::decay_t<Callable>>>> {
        auto packaged_task = packageTask(std::forward<Callable>(callable));
        auto result_future = packaged_task->get_future();
        std::function<void()> task = [packaged_task] { (*packaged_task)(); };
        if (!task_queue_.tryPush(std::move(task))) {
            return std::nullopt;
        }
        return std::move(result_future);
    }

    void requestShutdown() { task_queue_.shutdown(); }

    // Called by owners, never from a task executing in this pool.
    void waitForWorkers() {
        std::lock_guard<std::mutex> lock(join_mutex_);
        for (auto& worker_thread : worker_threads_) {
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }

   private:
    template <typename Callable, typename... Arguments>
    static auto packageTask(Callable&& callable, Arguments&&... arguments) {
        using Result = std::invoke_result_t<std::decay_t<Callable>, std::decay_t<Arguments>...>;
        auto invocation = [function = std::forward<Callable>(callable),
                           captured_arguments = std::make_tuple(
                               std::forward<Arguments>(arguments)...)]() mutable -> Result {
            return std::apply(std::move(function), std::move(captured_arguments));
        };
        // std::function requires copyable closures; the packaged task is shared only
        // between this wrapper and the queue, while the future owns its result state.
        return std::make_shared<std::packaged_task<Result()>>(std::move(invocation));
    }

    void executeQueuedTasks() {
        while (auto task = task_queue_.pop()) {
            (*task)();  // packaged_task transports user exceptions into its future.
        }
    }

    BlockingQueue<std::function<void()>> task_queue_;
    std::vector<std::thread> worker_threads_;
    std::mutex join_mutex_;
};

}  // namespace inference_runtime
