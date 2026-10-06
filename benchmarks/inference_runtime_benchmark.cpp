#include <deque>
#include <numeric>

#include "benchmark.hpp"

namespace inference_runtime::benchmark {

double calculateBufferChecksum(FloatTensorView buffer) {
    return std::accumulate(buffer.values, buffer.values + buffer.element_count, 0.0);
}

namespace {
struct ClientMeasurements {
    std::vector<PredictionResult> predictions;
    std::chrono::steady_clock::time_point finished_at;
};
}  // namespace

void summarizeLatencies(BenchmarkResult& result, const std::vector<double>& latency_samples) {
    result.percentiles = LatencyTracker::calculatePercentiles(latency_samples);
    if (!latency_samples.empty()) {
        result.average_latency_ms =
            std::accumulate(latency_samples.begin(), latency_samples.end(), 0.0) /
            latency_samples.size();
    }
}

BenchmarkResult benchmarkRuntime(const BenchmarkConfig& config) {
    InferenceService service(config.service);
    std::vector<float> sample_input(service.inputElementCount(), 0.25F);
    std::vector<std::future<PredictionResult>> warmup_results;
    for (std::size_t index = 0; index < config.warmup_requests; ++index) {
        warmup_results.push_back(service.submit(sample_input));
    }
    for (auto& result : warmup_results) {
        if (!result.get().succeeded()) {
            throw std::runtime_error("Warmup inference failed");
        }
    }
    const auto baseline_metrics = service.getMetrics();
    std::atomic<std::size_t> next_request{0};
    std::promise<void> start_clients;
    auto start_signal = start_clients.get_future().share();
    std::vector<std::future<ClientMeasurements>> client_results;
    client_results.reserve(config.clients);
    try {
        for (std::size_t client_index = 0; client_index < config.clients; ++client_index) {
            client_results.push_back(std::async(std::launch::async, [&] {
                std::vector<PredictionResult> completed;
                std::deque<std::future<PredictionResult>> pending;
                start_signal.wait();
                bool submission_finished = false;
                while (!submission_finished || !pending.empty()) {
                    while (!submission_finished && pending.size() < config.outstanding_per_client) {
                        if (next_request.fetch_add(1, std::memory_order_relaxed) >=
                            config.requests) {
                            submission_finished = true;
                        } else {
                            pending.push_back(service.submit(sample_input));
                        }
                    }
                    if (!pending.empty()) {
                        completed.push_back(pending.front().get());
                        pending.pop_front();
                    }
                }
                return ClientMeasurements{std::move(completed), std::chrono::steady_clock::now()};
            }));
        }
    } catch (...) {
        // Wake partially created clients before future destructors join them.
        next_request.store(config.requests);
        start_clients.set_value();
        throw;
    }
    const auto measurement_started_at = std::chrono::steady_clock::now();
    start_clients.set_value();
    BenchmarkResult result;
    std::vector<double> latency_samples;
    latency_samples.reserve(config.requests);
    double queue_wait_sum = 0;
    double inference_time_sum = 0;
    auto measurement_finished_at = measurement_started_at;
    for (auto& client : client_results) {
        auto measurements = client.get();
        measurement_finished_at = std::max(measurement_finished_at, measurements.finished_at);
        for (const auto& prediction : measurements.predictions) {
            ++result.completed_requests;
            if (!prediction.succeeded()) {
                ++result.failed_requests;
            }
            latency_samples.push_back(prediction.total_latency.count());
            queue_wait_sum += prediction.queue_waiting_time.count();
            inference_time_sum += prediction.inference_time.count();
        }
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(measurement_finished_at - measurement_started_at).count();
    service.waitForShutdown();
    const auto metrics = service.getMetrics();
    const auto measured_batches =
        metrics.total_batches_executed - baseline_metrics.total_batches_executed;
    result.average_batch_size =
        measured_batches > 0 ? static_cast<double>(result.completed_requests) / measured_batches
                             : 0;
    result.maximum_queue_depth = metrics.maximum_queue_depth;
    if (result.completed_requests > 0) {
        result.average_queue_wait_ms = queue_wait_sum / result.completed_requests;
        result.average_inference_time_ms = inference_time_sum / result.completed_requests;
    }
    summarizeLatencies(result, latency_samples);
    return result;
}

}  // namespace inference_runtime::benchmark
