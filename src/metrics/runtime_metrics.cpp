#include "inference_runtime/metrics/runtime_metrics.hpp"

namespace inference_runtime {

void RuntimeMetricsCollector::recordAccepted() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++metrics_.accepted_requests;
}
void RuntimeMetricsCollector::recordRejected() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++metrics_.rejected_requests;
}
void RuntimeMetricsCollector::recordBatch(std::size_t batch_size) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++metrics_.total_batches_executed;
    batched_request_count_ += batch_size;
}
void RuntimeMetricsCollector::recordCompletion(const PredictionResult& result) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++metrics_.completed_requests;
    if (!result.succeeded()) {
        ++metrics_.failed_requests;
    }
    total_latency_ms_ += result.total_latency.count();
    total_queue_wait_ms_ += result.queue_waiting_time.count();
    total_inference_latency_ms_ += result.inference_time.count();
    latency_tracker_.recordLatency(result.total_latency.count());
}
void RuntimeMetricsCollector::markStopped() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (stopped_at_ == std::chrono::steady_clock::time_point{}) {
        stopped_at_ = std::chrono::steady_clock::now();
    }
}
RuntimeMetrics RuntimeMetricsCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto snapshot = metrics_;
    if (snapshot.completed_requests > 0) {
        const double completed = static_cast<double>(snapshot.completed_requests);
        snapshot.average_latency_ms = total_latency_ms_ / completed;
        snapshot.average_queue_wait_ms = total_queue_wait_ms_ / completed;
        snapshot.average_inference_latency_ms = total_inference_latency_ms_ / completed;
    }
    if (snapshot.total_batches_executed > 0) {
        snapshot.average_batch_size =
            static_cast<double>(batched_request_count_) / snapshot.total_batches_executed;
    }
    const auto measurement_end = stopped_at_ == std::chrono::steady_clock::time_point{}
                                     ? std::chrono::steady_clock::now()
                                     : stopped_at_;
    const double elapsed = std::chrono::duration<double>(measurement_end - started_at_).count();
    snapshot.throughput_rps = elapsed > 0 ? snapshot.completed_requests / elapsed : 0;
    snapshot.latency_percentiles = latency_tracker_.getPercentiles();
    return snapshot;
}

}  // namespace inference_runtime
