#pragma once

#include <string>

#include "inference_runtime/server/inference_service.hpp"

namespace inference_runtime::benchmark {

template <typename Queue>
class QueueShutdownGuard {
   public:
    explicit QueueShutdownGuard(Queue& request_queue) : request_queue_(request_queue) {}
    ~QueueShutdownGuard() { request_queue_.shutdown(); }
    QueueShutdownGuard(const QueueShutdownGuard&) = delete;
    QueueShutdownGuard& operator=(const QueueShutdownGuard&) = delete;

   private:
    Queue& request_queue_;
};

struct BenchmarkConfig {
    std::string configuration{"custom"};
    std::size_t requests{10000};
    std::size_t clients{16};
    std::size_t outstanding_per_client{16};
    std::size_t warmup_requests{64};
    InferenceServiceConfig service;
};

struct BenchmarkResult {
    std::size_t completed_requests{0};
    std::size_t failed_requests{0};
    double elapsed_seconds{0};
    double average_latency_ms{0};
    double average_queue_wait_ms{0};
    double average_inference_time_ms{0};
    double average_batch_size{0};
    std::size_t maximum_queue_depth{0};
    LatencyPercentiles percentiles;
};

BenchmarkResult benchmarkRuntime(const BenchmarkConfig& config);
BenchmarkResult benchmarkQueue(const BenchmarkConfig& config);
BenchmarkResult benchmarkBatching(const BenchmarkConfig& config);
BenchmarkResult benchmarkBuffers(const BenchmarkConfig& config);
void summarizeLatencies(BenchmarkResult& result, const std::vector<double>& latency_samples);
double calculateBufferChecksum(FloatTensorView buffer);

}  // namespace inference_runtime::benchmark
