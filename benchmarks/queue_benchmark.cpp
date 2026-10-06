#include "benchmark.hpp"

namespace inference_runtime::benchmark {

BenchmarkResult benchmarkQueue(const BenchmarkConfig& config) {
    using Timestamp = std::chrono::steady_clock::time_point;
    BlockingQueue<Timestamp> request_queue(config.service.maximum_queue_size);
    std::atomic<std::size_t> next_request{0};
    std::vector<std::future<std::vector<double>>> consumers;
    std::vector<std::future<void>> producers;
    QueueShutdownGuard queue_shutdown_guard(request_queue);
    const auto started_at = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < config.service.worker_threads; ++index) {
        consumers.push_back(std::async(std::launch::async, [&] {
            try {
                std::vector<double> samples;
                while (auto received_at = request_queue.pop()) {
                    samples.push_back(
                        Milliseconds(std::chrono::steady_clock::now() - *received_at).count());
                }
                return samples;
            } catch (...) {
                request_queue.shutdown();
                throw;
            }
        }));
    }
    for (std::size_t index = 0; index < config.clients; ++index) {
        producers.push_back(std::async(std::launch::async, [&] {
            while (next_request.fetch_add(1, std::memory_order_relaxed) < config.requests) {
                if (!request_queue.push(std::chrono::steady_clock::now())) {
                    break;
                }
            }
        }));
    }
    for (auto& producer : producers) {
        producer.get();
    }
    request_queue.shutdown();
    std::vector<double> samples;
    for (auto& consumer : consumers) {
        auto local_samples = consumer.get();
        samples.insert(samples.end(), local_samples.begin(), local_samples.end());
    }
    BenchmarkResult result;
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    result.completed_requests = samples.size();
    result.maximum_queue_depth = request_queue.maximumObservedSize();
    summarizeLatencies(result, samples);
    result.average_queue_wait_ms = result.average_latency_ms;
    return result;
}

}  // namespace inference_runtime::benchmark
