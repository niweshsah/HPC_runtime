#include "inference_runtime/inference/batch_scheduler.hpp"

#include <gtest/gtest.h>

using namespace inference_runtime;
using namespace std::chrono_literals;

namespace {
InferenceRequest makeRequest(
    std::uint64_t request_id,
    std::chrono::steady_clock::time_point received_at = std::chrono::steady_clock::now()) {
    return {request_id, {0.1F, 0.2F, 0.3F}, received_at, {}};
}
}  // namespace

TEST(BatchScheduler, BuildsFullBatchInOrder) {
    BlockingQueue<InferenceRequest> request_queue(8);
    BatchScheduler scheduler(request_queue, {3, 1s});
    for (std::uint64_t request_id = 1; request_id <= 3; ++request_id) {
        ASSERT_TRUE(request_queue.push(makeRequest(request_id)));
    }
    auto batch = scheduler.buildNextBatch();
    ASSERT_EQ(batch.size(), 3U);
    EXPECT_EQ(batch[0].request_id, 1U);
    EXPECT_EQ(batch[2].request_id, 3U);
}

TEST(BatchScheduler, ExpiredOldestRequestDoesNotWaitForNewerRequests) {
    BlockingQueue<InferenceRequest> request_queue(8);
    BatchScheduler scheduler(request_queue, {4, 5ms});
    ASSERT_TRUE(request_queue.push(makeRequest(1, std::chrono::steady_clock::now() - 1s)));
    ASSERT_TRUE(request_queue.push(makeRequest(2)));
    EXPECT_EQ(scheduler.buildNextBatch().size(), 1U);
    request_queue.shutdown();
    EXPECT_EQ(scheduler.buildNextBatch().size(), 1U);
}

TEST(BatchScheduler, LowTrafficProducesPartialBatchOnTimeout) {
    BlockingQueue<InferenceRequest> request_queue(8);
    BatchScheduler scheduler(request_queue, {4, 10ms});
    ASSERT_TRUE(request_queue.push(makeRequest(1)));
    auto result = std::async(std::launch::async, [&] { return scheduler.buildNextBatch(); });
    EXPECT_EQ(result.wait_for(2s), std::future_status::ready);
    request_queue.shutdown();
    EXPECT_EQ(result.get().size(), 1U);
}

TEST(BatchScheduler, ShutdownFlushesPartialBatchAndEndsConsumer) {
    BlockingQueue<InferenceRequest> request_queue(8);
    BatchScheduler scheduler(request_queue, {4, 10s});
    ASSERT_TRUE(request_queue.push(makeRequest(1)));
    auto result = std::async(std::launch::async, [&] { return scheduler.buildNextBatch(); });
    request_queue.shutdown();
    EXPECT_EQ(result.get().size(), 1U);
    EXPECT_TRUE(scheduler.buildNextBatch().empty());
}

TEST(BatchScheduler, HighTrafficDrainsWithoutLosingRequests) {
    BlockingQueue<InferenceRequest> request_queue(100);
    BatchScheduler scheduler(request_queue, {8, 1s});
    for (std::uint64_t request_id = 0; request_id < 100; ++request_id) {
        ASSERT_TRUE(request_queue.push(makeRequest(request_id)));
    }
    request_queue.shutdown();
    std::size_t request_count = 0;
    while (true) {
        auto batch = scheduler.buildNextBatch();
        if (batch.empty()) {
            break;
        }
        EXPECT_LE(batch.size(), 8U);
        request_count += batch.size();
    }
    EXPECT_EQ(request_count, 100U);
}

TEST(BatchScheduler, EarlierArrivalUpdatesTheDeadlineDespiteEnqueueOrder) {
    BlockingQueue<InferenceRequest> request_queue(8);
    BatchScheduler scheduler(request_queue, {4, 1s});
    ASSERT_TRUE(request_queue.push(makeRequest(1)));
    ASSERT_TRUE(request_queue.push(makeRequest(2, std::chrono::steady_clock::now() - 2s)));
    ASSERT_TRUE(request_queue.push(makeRequest(3)));
    EXPECT_EQ(scheduler.buildNextBatch().size(), 2U);
    request_queue.shutdown();
    EXPECT_EQ(scheduler.buildNextBatch().size(), 1U);
}
