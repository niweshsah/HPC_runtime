#include "inference_runtime/concurrency/blocking_queue.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using inference_runtime::BlockingQueue;
using namespace std::chrono_literals;

TEST(BlockingQueue, RejectsZeroCapacity) {
    EXPECT_THROW(BlockingQueue<int>(0), std::invalid_argument);
}

TEST(BlockingQueue, PreservesOrderAndMoveOnlyOwnership) {
    BlockingQueue<std::unique_ptr<int>> request_queue(2);
    auto input = std::make_unique<int>(42);
    EXPECT_TRUE(request_queue.push(std::move(input)));
    EXPECT_EQ(input, nullptr);
    auto removed = request_queue.pop();
    ASSERT_TRUE(removed);
    EXPECT_EQ(**removed, 42);
    EXPECT_TRUE(request_queue.empty());
}

TEST(BlockingQueue, ConsumerWaitsUntilInsertion) {
    BlockingQueue<int> request_queue(1);
    auto consumer = std::async(std::launch::async, [&] { return request_queue.pop(); });
    EXPECT_EQ(consumer.wait_for(20ms), std::future_status::timeout);
    EXPECT_TRUE(request_queue.push(17));
    EXPECT_EQ(consumer.get(), 17);
}

TEST(BlockingQueue, ProducerWaitsUntilRemoval) {
    BlockingQueue<int> request_queue(1);
    ASSERT_TRUE(request_queue.push(1));
    auto producer = std::async(std::launch::async, [&] { return request_queue.push(2); });
    EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(request_queue.pop(), 1);
    EXPECT_TRUE(producer.get());
    EXPECT_EQ(request_queue.pop(), 2);
}

TEST(BlockingQueue, ShutdownWakesBothSidesAndDrains) {
    BlockingQueue<int> full_queue(1);
    ASSERT_TRUE(full_queue.push(1));
    auto producer = std::async(std::launch::async, [&] { return full_queue.push(2); });
    full_queue.shutdown();
    EXPECT_FALSE(producer.get());
    EXPECT_EQ(full_queue.pop(), 1);
    EXPECT_FALSE(full_queue.pop());
    EXPECT_TRUE(full_queue.isShutdown());
    BlockingQueue<int> empty_queue(1);
    auto consumer = std::async(std::launch::async, [&] { return empty_queue.pop(); });
    empty_queue.shutdown();
    EXPECT_FALSE(consumer.get());
    empty_queue.shutdown();
}

TEST(BlockingQueue, FailedInsertionPreservesOwnership) {
    BlockingQueue<std::unique_ptr<int>> request_queue(1);
    request_queue.shutdown();
    auto input = std::make_unique<int>(42);
    EXPECT_FALSE(request_queue.push(std::move(input)));
    EXPECT_EQ(*input, 42);
}

TEST(BlockingQueue, MultipleProducersAndConsumers) {
    BlockingQueue<int> request_queue(7);
    std::atomic<int> consumed_count{0};
    std::atomic<int> consumed_sum{0};
    std::vector<std::thread> consumers;
    std::vector<std::thread> producers;
    for (int index = 0; index < 4; ++index) {
        consumers.emplace_back([&] {
            while (auto item = request_queue.pop()) {
                consumed_sum.fetch_add(*item);
                consumed_count.fetch_add(1);
            }
        });
        producers.emplace_back([&, index] {
            for (int item = 0; item < 100; ++item) {
                EXPECT_TRUE(request_queue.push(index * 100 + item));
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }
    request_queue.shutdown();
    for (auto& consumer : consumers) {
        consumer.join();
    }
    EXPECT_EQ(consumed_count, 400);
    EXPECT_EQ(consumed_sum, 399 * 400 / 2);
    EXPECT_LE(request_queue.maximumObservedSize(), 7U);
}

TEST(BlockingQueue, DeadlineAndNonblockingInsertion) {
    BlockingQueue<int> request_queue(1);
    EXPECT_FALSE(request_queue.popUntil(std::chrono::steady_clock::now()));
    EXPECT_TRUE(request_queue.tryPush(1));
    EXPECT_FALSE(request_queue.tryPush(2));
    EXPECT_EQ(request_queue.popUntil(std::chrono::steady_clock::now()), 1);
}

TEST(BlockingQueue, CopiesLvaluesWhenTheElementTypeSupportsCopying) {
    BlockingQueue<int> request_queue(1);
    const int input_value = 17;
    EXPECT_TRUE(request_queue.push(input_value));
    EXPECT_EQ(request_queue.pop(), input_value);
}
