#include "benchmark.hpp"

namespace inference_runtime::benchmark {

BenchmarkResult benchmarkBatching(const BenchmarkConfig& config) {
    BlockingQueue<InferenceRequest> request_queue(config.service.maximum_queue_size);
    BatchScheduler scheduler(request_queue, config.service.batching);
    std::atomic<std::size_t> next_request{0};
    std::size_t batch_count = 0;
    std::future<std::vector<double>> consumer;
    std::vector<std::future<void>> producers;
    QueueShutdownGuard queue_shutdown_guard(request_queue);
    const auto started_at = std::chrono::steady_clock::now();
    consumer = std::async(std::launch::async, [&] {
        try {
            std::vector<double> samples;
            while (true) {
                const auto batch = scheduler.buildNextBatch();
                if (batch.empty()) {
                    break;
                }
                ++batch_count;
                const auto dispatched_at = std::chrono::steady_clock::now();
                for (const auto& request : batch) {
                    samples.push_back(
                        Milliseconds(dispatched_at - request.request_received_at).count());
                }
            }
            return samples;
        } catch (...) {
            request_queue.shutdown();
            throw;
        }
    });
    for (std::size_t index = 0; index < config.clients; ++index) {
        producers.push_back(std::async(std::launch::async, [&] {
            while (true) {
                const auto request_id = next_request.fetch_add(1, std::memory_order_relaxed);
                if (request_id >= config.requests) {
                    break;
                }
                if (!request_queue.push({request_id, {}, std::chrono::steady_clock::now(), {}})) {
                    break;
                }
            }
        }));
    }
    for (auto& producer : producers) {
        producer.get();
    }
    request_queue.shutdown();
    auto samples = consumer.get();
    BenchmarkResult result;
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    result.completed_requests = samples.size();
    result.average_batch_size = static_cast<double>(samples.size()) / batch_count;
    result.maximum_queue_depth = request_queue.maximumObservedSize();
    summarizeLatencies(result, samples);
    result.average_queue_wait_ms = result.average_latency_ms;
    return result;
}

}  // namespace inference_runtime::benchmark
