#pragma once

#include <mutex>

#include "inference_runtime/inference/prediction_result.hpp"
#include "inference_runtime/metrics/latency_tracker.hpp"

namespace inference_runtime {

struct RuntimeMetrics {
    std::uint64_t accepted_requests{0};
    std::uint64_t rejected_requests{0};
    std::uint64_t completed_requests{0};
    std::uint64_t failed_requests{0};
    std::size_t current_queue_depth{0};
    std::size_t maximum_queue_depth{0};
    std::uint64_t total_batches_executed{0};
    double average_batch_size{0};
    double average_latency_ms{0};
    double average_inference_latency_ms{0};
    double average_queue_wait_ms{0};
    double throughput_rps{0};
    LatencyPercentiles latency_percentiles;
};

class RuntimeMetricsCollector {
   public:
    void recordAccepted();
    void recordRejected();
    void recordBatch(std::size_t batch_size);
    void recordCompletion(const PredictionResult& result);
    void markStopped();
    RuntimeMetrics snapshot() const;

   private:
    mutable std::mutex metrics_mutex_;
    RuntimeMetrics metrics_;
    LatencyTracker latency_tracker_;
    double total_latency_ms_{0};
    double total_queue_wait_ms_{0};
    double total_inference_latency_ms_{0};
    std::uint64_t batched_request_count_{0};
    const std::chrono::steady_clock::time_point started_at_{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point stopped_at_{};
};

}  // namespace inference_runtime
