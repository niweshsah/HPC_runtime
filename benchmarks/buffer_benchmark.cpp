#include <algorithm>

#include "benchmark.hpp"

namespace inference_runtime::benchmark {

BenchmarkResult benchmarkBuffers(const BenchmarkConfig& config) {
    InferenceEngine engine(config.service.engine);
    if (config.service.batching.maximum_batch_size >
        std::numeric_limits<std::size_t>::max() / engine.inputElementCount()) {
        throw std::invalid_argument("Buffer benchmark size overflows");
    }
    const auto buffer_size =
        config.service.batching.maximum_batch_size * engine.inputElementCount();
    std::unique_ptr<ReusableBufferPool> buffer_pool;
    if (config.service.reuse_input_buffers) {
        buffer_pool = std::make_unique<ReusableBufferPool>(config.clients, buffer_size);
    }
    std::atomic<std::size_t> next_request{0};
    std::vector<std::future<std::vector<double>>> clients;
    const auto started_at = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < config.clients; ++index) {
        clients.push_back(std::async(std::launch::async, [&] {
            std::vector<double> samples;
            while (next_request.fetch_add(1, std::memory_order_relaxed) < config.requests) {
                const auto operation_started_at = std::chrono::steady_clock::now();
                {
                    std::optional<ReusableBufferPool::BufferHandle> leased_buffer;
                    std::vector<float> allocated_buffer;
                    if (buffer_pool) {
                        leased_buffer.emplace(buffer_pool->acquire());
                    } else {
                        allocated_buffer.resize(buffer_size);
                    }
                    auto* storage = leased_buffer ? leased_buffer->data() : allocated_buffer.data();
                    std::fill(storage, storage + buffer_size, 0.25F);
                    // A separate translation unit consumes all initialized values,
                    // keeping allocation and memory writes observable without LTO.
                    if (calculateBufferChecksum({storage, buffer_size}) != buffer_size * 0.25) {
                        throw std::runtime_error("Invalid buffer contents");
                    }
                }
                samples.push_back(
                    Milliseconds(std::chrono::steady_clock::now() - operation_started_at).count());
            }
            return samples;
        }));
    }
    std::vector<double> samples;
    for (auto& client : clients) {
        auto local_samples = client.get();
        samples.insert(samples.end(), local_samples.begin(), local_samples.end());
    }
    BenchmarkResult result;
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    result.completed_requests = samples.size();
    summarizeLatencies(result, samples);
    return result;
}

}  // namespace inference_runtime::benchmark
