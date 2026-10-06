#pragma once

#include <atomic>
#include <functional>

#include "inference_runtime/concurrency/thread_pool.hpp"
#include "inference_runtime/inference/batch_scheduler.hpp"
#include "inference_runtime/inference/inference_engine.hpp"
#include "inference_runtime/logging/structured_logger.hpp"
#include "inference_runtime/memory/reusable_buffer_pool.hpp"
#include "inference_runtime/metrics/runtime_metrics.hpp"

namespace inference_runtime {

struct InferenceServiceConfig {
    std::size_t worker_threads{4};
    std::size_t maximum_queue_size{1024};
    BatchSchedulerConfig batching;
    InferenceEngineConfig engine;
    bool reuse_input_buffers{false};
};

class InferenceService {
   public:
    // Composition hook for deterministic tests or another synchronous batch executor.
    using BatchExecutor = std::function<std::vector<ModelPrediction>(FloatTensorView, std::size_t)>;
    explicit InferenceService(const InferenceServiceConfig& config, BatchExecutor executor = {});
    ~InferenceService();
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    std::future<PredictionResult> submit(std::vector<float> input_tensor);
    void requestShutdown();
    void waitForShutdown();
    RuntimeMetrics getMetrics() const;
    bool isShuttingDown() const noexcept;
    std::size_t inputElementCount() const noexcept;

   private:
    void scheduleBatches();
    void executeBatch(RequestBatch& batch);
    void completeFailedBatch(RequestBatch& batch, const std::string& error_message);
    std::future<PredictionResult> rejectRequest(InferenceRequest& request, PredictionStatus status,
                                                const char* error_message);

    const InferenceServiceConfig config_;
    InferenceEngine inference_engine_;
    BatchExecutor batch_executor_;
    BlockingQueue<InferenceRequest> request_queue_;
    BatchScheduler batch_scheduler_;
    RuntimeMetricsCollector metrics_;
    StructuredLogger logger_;
    std::unique_ptr<ReusableBufferPool> input_buffer_pool_;
    ThreadPool worker_pool_;
    std::thread scheduler_thread_;
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<std::uint64_t> next_request_id_{1};
    std::mutex shutdown_join_mutex_;
};

}  // namespace inference_runtime
