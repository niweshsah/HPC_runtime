#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace inference_runtime {

using Milliseconds = std::chrono::duration<double, std::milli>;

enum class PredictionStatus { success, invalid_input, shutting_down, execution_failed };
const char* predictionStatusName(PredictionStatus status) noexcept;

struct ModelPrediction {
    std::size_t class_index{0};
    float confidence{0.0F};
};

struct PredictionResult {
    std::uint64_t request_id{0};
    std::size_t predicted_class_index{0};
    float confidence_score{0.0F};
    Milliseconds queue_waiting_time{0};
    Milliseconds inference_time{0};
    Milliseconds total_latency{0};
    PredictionStatus status{PredictionStatus::success};
    std::string error_message;

    bool succeeded() const noexcept { return status == PredictionStatus::success; }
};

}  // namespace inference_runtime
