#include <gtest/gtest.h>

#include <numeric>

#include "inference_runtime/metrics/runtime_metrics.hpp"

using namespace inference_runtime;

TEST(LatencyTracker, NearestRankUsesAllSortedSamples) {
    std::vector<double> samples(100);
    std::iota(samples.rbegin(), samples.rend(), 1.0);
    const auto percentiles = LatencyTracker::calculatePercentiles(samples);
    EXPECT_EQ(percentiles.p50_ms, 50);
    EXPECT_EQ(percentiles.p95_ms, 95);
    EXPECT_EQ(percentiles.p99_ms, 99);
    EXPECT_EQ(percentiles.sample_count, 100U);
    EXPECT_EQ(LatencyTracker::calculatePercentiles({7}).p99_ms, 7);
    EXPECT_EQ(LatencyTracker::calculatePercentiles({}).sample_count, 0U);
}

TEST(LatencyTracker, RollingWindowRemainsBounded) {
    LatencyTracker tracker(3);
    tracker.recordLatency(100);
    tracker.recordLatency(1);
    tracker.recordLatency(2);
    tracker.recordLatency(3);
    const auto percentiles = tracker.getPercentiles();
    EXPECT_EQ(percentiles.sample_count, 3U);
    EXPECT_EQ(percentiles.p50_ms, 2);
    EXPECT_EQ(percentiles.p99_ms, 3);
}

TEST(RuntimeMetrics, ComputesLifetimeAveragesAndTerminalCounters) {
    RuntimeMetricsCollector collector;
    collector.recordAccepted();
    collector.recordAccepted();
    collector.recordRejected();
    collector.recordBatch(2);
    PredictionResult result;
    result.total_latency = Milliseconds(10);
    result.queue_waiting_time = Milliseconds(3);
    result.inference_time = Milliseconds(4);
    collector.recordCompletion(result);
    result.status = PredictionStatus::execution_failed;
    result.total_latency = Milliseconds(20);
    collector.recordCompletion(result);
    collector.markStopped();
    const auto metrics = collector.snapshot();
    EXPECT_EQ(metrics.completed_requests, 2U);
    EXPECT_EQ(metrics.failed_requests, 1U);
    EXPECT_EQ(metrics.rejected_requests, 1U);
    EXPECT_EQ(metrics.average_latency_ms, 15);
    EXPECT_EQ(metrics.average_batch_size, 2);
    EXPECT_EQ(metrics.average_queue_wait_ms, 3);
    EXPECT_GT(metrics.throughput_rps, 0);
}
