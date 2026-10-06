#include "inference_runtime/server/inference_service.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace inference_runtime {

InferenceService::InferenceService(const InferenceServiceConfig& config, BatchExecutor executor)
    : config_(config),
      inference_engine_(config.engine),
      batch_executor_(std::move(executor)),
      request_queue_(config.maximum_queue_size),
      batch_scheduler_(request_queue_, config.batching),
      worker_pool_(config.worker_threads, config.worker_threads) {
    if (config.batching.maximum_batch_size >
        std::numeric_limits<std::size_t>::max() / inputElementCount()) {
        throw std::invalid_argument("Batch input size overflows");
    }
    if (!batch_executor_) {
        batch_executor_ = [this](FloatTensorView input, std::size_t batch_size) {
            return inference_engine_.execute(input, batch_size);
        };
    }
    if (config.reuse_input_buffers) {
        input_buffer_pool_ = std::make_unique<ReusableBufferPool>(
            config.worker_threads, config.batching.maximum_batch_size * inputElementCount());
    }
    logger_.write("info", "service_starting");
    logger_.write("info", "model_loaded",
                  config.engine.backend == InferenceBackend::mock ? "deterministic mock"
                                                                  : config.engine.model_path);
    scheduler_thread_ = std::thread([this] { scheduleBatches(); });
    logger_.write("info", "worker_pool_started");
}

InferenceService::~InferenceService() {
    requestShutdown();
    waitForShutdown();
}

std::future<PredictionResult> InferenceService::rejectRequest(InferenceRequest& request,
                                                              PredictionStatus status,
                                                              const char* error_message) {
    auto result_future = request.result_promise.get_future();
    PredictionResult result;
    result.request_id = request.request_id;
    result.status = status;
    result.error_message = error_message;
    result.total_latency = std::chrono::steady_clock::now() - request.request_received_at;
    metrics_.recordRejected();
    request.result_promise.set_value(std::move(result));
    return result_future;
}

std::future<PredictionResult> InferenceService::submit(std::vector<float> input_tensor) {
    InferenceRequest request{next_request_id_.fetch_add(1, std::memory_order_relaxed),
                             std::move(input_tensor),
                             std::chrono::steady_clock::now(),
                             {}};
    if (isShuttingDown()) {
        return rejectRequest(request, PredictionStatus::shutting_down, "Service is shutting down");
    }
    if (request.input_tensor.size() != inputElementCount() ||
        !std::all_of(request.input_tensor.begin(), request.input_tensor.end(),
                     [](float value) { return std::isfinite(value); })) {
        return rejectRequest(request, PredictionStatus::invalid_input,
                             "Expected a finite tensor matching the configured input shape");
    }
    auto result_future = request.result_promise.get_future();
    if (!request_queue_.push(std::move(request))) {
        PredictionResult failure;
        failure.request_id = request.request_id;
        failure.status = PredictionStatus::shutting_down;
        failure.error_message = "Shutdown interrupted request admission";
        failure.total_latency = std::chrono::steady_clock::now() - request.request_received_at;
        metrics_.recordRejected();
        request.result_promise.set_value(std::move(failure));
        return result_future;
    }
    metrics_.recordAccepted();
    return result_future;
}

void InferenceService::scheduleBatches() {
    try {
        while (true) {
            auto batch = std::make_shared<RequestBatch>(batch_scheduler_.buildNextBatch());
            if (batch->empty()) {
                break;
            }
            try {
                // Retain the batch until enqueue succeeds so dispatch failures can
                // still fulfill every promise instead of producing broken futures.
                static_cast<void>(worker_pool_.submit([this, batch] { executeBatch(*batch); }));
            } catch (const std::exception& error) {
                completeFailedBatch(*batch, error.what());
            }
        }
    } catch (const std::exception& error) {
        logger_.write("error", "scheduler_failure", error.what());
        requestShutdown();
        while (auto request = request_queue_.pop()) {
            RequestBatch batch;
            batch.push_back(std::move(*request));
            completeFailedBatch(batch, error.what());
        }
    }
    worker_pool_.requestShutdown();
}

void InferenceService::executeBatch(RequestBatch& batch) {
    const auto worker_started_at = std::chrono::steady_clock::now();
    auto inference_started_at = worker_started_at;
    auto inference_finished_at = worker_started_at;
    metrics_.recordBatch(batch.size());
    std::vector<ModelPrediction> predictions;
    std::string error_message;
    bool execution_failed = false;
    try {
        const auto batch_element_count = batch.size() * inputElementCount();
        std::optional<ReusableBufferPool::BufferHandle> pooled_buffer;
        std::vector<float> allocated_buffer;
        if (input_buffer_pool_) {
            pooled_buffer.emplace(input_buffer_pool_->acquire());
        } else {
            allocated_buffer.resize(batch_element_count);
        }
        float* batch_input = pooled_buffer ? pooled_buffer->data() : allocated_buffer.data();
        auto destination = batch_input;
        for (const auto& request : batch) {
            destination =
                std::copy(request.input_tensor.begin(), request.input_tensor.end(), destination);
        }
        inference_started_at = std::chrono::steady_clock::now();
        predictions = batch_executor_({batch_input, batch_element_count}, batch.size());
        inference_finished_at = std::chrono::steady_clock::now();
        if (predictions.size() != batch.size()) {
            throw std::runtime_error("Backend returned an incorrect prediction count");
        }
        if (!std::all_of(predictions.begin(), predictions.end(), [](const auto& prediction) {
                return std::isfinite(prediction.confidence) && prediction.confidence >= 0 &&
                       prediction.confidence <= 1;
            })) {
            throw std::runtime_error("Backend returned invalid confidence scores");
        }
    } catch (const std::exception& error) {
        execution_failed = true;
        inference_finished_at = std::chrono::steady_clock::now();
        error_message = error.what();
        if (error_message.empty()) {
            error_message = "Inference execution failed";
        }
    } catch (...) {
        execution_failed = true;
        inference_finished_at = std::chrono::steady_clock::now();
        error_message = "Unknown inference execution failure";
    }
    if (execution_failed) {
        logger_.write("error", "batch_execution_failure", error_message);
    }
    for (std::size_t index = 0; index < batch.size(); ++index) {
        auto& request = batch[index];
        PredictionResult result;
        result.request_id = request.request_id;
        result.queue_waiting_time = worker_started_at - request.request_received_at;
        result.inference_time = inference_finished_at - inference_started_at;
        result.total_latency = std::chrono::steady_clock::now() - request.request_received_at;
        if (!execution_failed) {
            result.predicted_class_index = predictions[index].class_index;
            result.confidence_score = predictions[index].confidence;
        } else {
            result.status = PredictionStatus::execution_failed;
            result.error_message = error_message;
        }
        metrics_.recordCompletion(result);
        request.result_promise.set_value(std::move(result));
    }
}

void InferenceService::completeFailedBatch(RequestBatch& batch, const std::string& error_message) {
    for (auto& request : batch) {
        PredictionResult result;
        result.request_id = request.request_id;
        result.status = PredictionStatus::execution_failed;
        result.error_message = error_message;
        result.total_latency = std::chrono::steady_clock::now() - request.request_received_at;
        result.queue_waiting_time = result.total_latency;
        metrics_.recordCompletion(result);
        request.result_promise.set_value(std::move(result));
    }
}

void InferenceService::requestShutdown() {
    if (!shutdown_requested_.exchange(true)) {
        request_queue_.shutdown();
        logger_.write("info", "shutdown_requested");
    }
}

void InferenceService::waitForShutdown() {
    requestShutdown();
    std::lock_guard<std::mutex> lock(shutdown_join_mutex_);
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
        worker_pool_.waitForWorkers();
        metrics_.markStopped();
        logger_.write("info", "service_stopped");
    }
}

RuntimeMetrics InferenceService::getMetrics() const {
    auto snapshot = metrics_.snapshot();
    snapshot.current_queue_depth = request_queue_.size();
    snapshot.maximum_queue_depth = request_queue_.maximumObservedSize();
    return snapshot;
}
bool InferenceService::isShuttingDown() const noexcept { return shutdown_requested_.load(); }
std::size_t InferenceService::inputElementCount() const noexcept {
    return inference_engine_.inputElementCount();
}

}  // namespace inference_runtime
