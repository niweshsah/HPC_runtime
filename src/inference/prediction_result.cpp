#include "inference_runtime/inference/prediction_result.hpp"

namespace inference_runtime {
const char* predictionStatusName(PredictionStatus status) noexcept {
    switch (status) {
        case PredictionStatus::success:
            return "success";
        case PredictionStatus::invalid_input:
            return "invalid_input";
        case PredictionStatus::shutting_down:
            return "shutting_down";
        case PredictionStatus::execution_failed:
            return "execution_failed";
    }
    return "unknown";
}
}  // namespace inference_runtime
