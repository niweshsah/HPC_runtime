#include "inference_runtime/concurrency/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

using inference_runtime::ThreadPool;

TEST(ThreadPool, ReturnsResultsAndAcceptsMoveOnlyArguments) {
    ThreadPool thread_pool(2);
    auto result = thread_pool.submit([](std::unique_ptr<int> input) { return *input * 2; },
                                     std::make_unique<int>(21));
    EXPECT_EQ(result.get(), 42);
    EXPECT_THROW(ThreadPool(0), std::invalid_argument);
}

TEST(ThreadPool, TaskExceptionsReachFutureAndWorkersContinue) {
    ThreadPool thread_pool(1);
    auto failure = thread_pool.submit([]() -> int { throw std::logic_error("task failed"); });
    EXPECT_THROW(static_cast<void>(failure.get()), std::logic_error);
    EXPECT_EQ(thread_pool.submit([] { return 17; }).get(), 17);
}

TEST(ThreadPool, ExecutesConcurrentlyAndDrainsDuringShutdown) {
    ThreadPool thread_pool(4, 16);
    std::promise<void> release_workers;
    auto release_signal = release_workers.get_future().share();
    std::atomic<int> started{0};
    std::promise<void> all_started;
    std::vector<std::future<void>> results;
    for (int index = 0; index < 4; ++index) {
        results.push_back(thread_pool.submit([&] {
            if (started.fetch_add(1) == 3) {
                all_started.set_value();
            }
            release_signal.wait();
        }));
    }
    all_started.get_future().wait();
    auto final_result = thread_pool.submit([] { return 42; });
    thread_pool.requestShutdown();
    EXPECT_THROW(thread_pool.submit([] {}), std::runtime_error);
    release_workers.set_value();
    thread_pool.waitForWorkers();
    for (auto& result : results) {
        result.get();
    }
    EXPECT_EQ(final_result.get(), 42);
    thread_pool.waitForWorkers();
}

TEST(ThreadPool, BoundedTaskQueueSupportsNonblockingAdmission) {
    ThreadPool thread_pool(1, 1);
    std::promise<void> task_started;
    std::promise<void> release_task;
    auto release_signal = release_task.get_future().share();
    auto first_result = thread_pool.submit([&] {
        task_started.set_value();
        release_signal.wait();
    });
    task_started.get_future().wait();
    auto second_result = thread_pool.trySubmit([] { return 2; });
    EXPECT_TRUE(second_result);
    EXPECT_FALSE(thread_pool.trySubmit([] { return 3; }));
    release_task.set_value();
    first_result.get();
    if (second_result) {
        EXPECT_EQ(second_result->get(), 2);
    }
}
