#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <vector>

#include "inference_runtime/inference/prediction_result.hpp"

namespace inference_runtime {

// Represents a single inference request, containing the request ID, input tensor data, the time the request was received, and a promise to hold the prediction result.
struct InferenceRequest {
    std::uint64_t request_id;
    std::vector<float> input_tensor;
    std::chrono::steady_clock::time_point request_received_at;
    std::promise<PredictionResult> result_promise;
};

using RequestBatch = std::vector<InferenceRequest>;

}  // namespace inference_runtime
