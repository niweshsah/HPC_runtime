#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inference_runtime/inference/prediction_result.hpp"

namespace inference_runtime {

// Non-owning C++17 view. The caller keeps storage alive for the synchronous call.
struct FloatTensorView {
    const float* values;
    std::size_t element_count;
};

enum class InferenceBackend { mock, onnx };

struct InferenceEngineConfig {
    InferenceBackend backend{InferenceBackend::mock};
    std::string model_path;
    std::vector<std::int64_t> sample_shape{3};
    int intra_operation_threads{1};
    std::chrono::microseconds mock_execution_delay{0};
};

class InferenceEngine {
   public:
    explicit InferenceEngine(const InferenceEngineConfig& config);
    ~InferenceEngine();
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    std::vector<ModelPrediction> execute(FloatTensorView batch_input, std::size_t batch_size) const;
    std::size_t inputElementCount() const noexcept;
    static bool onnxAvailable() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace inference_runtime
