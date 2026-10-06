#include "inference_runtime/inference/inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#ifdef INFERENCE_RUNTIME_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace inference_runtime {

struct InferenceEngine::Implementation {
    InferenceEngineConfig config;
    std::size_t input_element_count{1};
#ifdef INFERENCE_RUNTIME_HAS_ONNX
    std::unique_ptr<Ort::Env> environment;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::size_t class_count{0};

    void loadModel() {
        if (config.model_path.empty()) {
            throw std::invalid_argument("Model path is required");
        }
        environment = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "inference_runtime");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.intra_operation_threads);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session = std::make_unique<Ort::Session>(*environment, config.model_path.c_str(),
                                                 session_options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
            throw std::invalid_argument("Model must have one tensor input and one logits output");
        }
        const auto input_type = session->GetInputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_info.GetShape();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input_shape.size() != config.sample_shape.size() + 1 || input_shape[0] >= 0 ||
            !std::equal(config.sample_shape.begin(), config.sample_shape.end(),
                        input_shape.begin() + 1)) {
            throw std::invalid_argument(
                "Model requires float32 input with dynamic batch and configured sample shape");
        }
        const auto output_type = session->GetOutputTypeInfo(0);
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        const auto output_shape = output_info.GetShape();
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_shape.size() != 2 || output_shape[0] >= 0 || output_shape[1] <= 0) {
            throw std::invalid_argument(
                "Model output must be float32 logits shaped [dynamic batch, classes]");
        }
        class_count = static_cast<std::size_t>(output_shape[1]);
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_name = session->GetOutputNameAllocated(0, allocator).get();
    }

    std::vector<ModelPrediction> executeOnnx(FloatTensorView batch_input, std::size_t batch_size) {
        std::vector<std::int64_t> batch_shape{static_cast<std::int64_t>(batch_size)};
        batch_shape.insert(batch_shape.end(), config.sample_shape.begin(),
                           config.sample_shape.end());
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // The ONNX wrapper requires float*, but inference treats the caller's input as read-only.
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(batch_input.values), batch_input.element_count,
            batch_shape.data(), batch_shape.size());
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        auto output_tensors =
            session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        const auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 2 || output_shape[0] != static_cast<std::int64_t>(batch_size) ||
            output_shape[1] != static_cast<std::int64_t>(class_count)) {
            throw std::runtime_error("Model returned an unexpected logits shape");
        }
        const auto* logits = output_tensors[0].GetTensorData<float>();
        std::vector<ModelPrediction> predictions;
        predictions.reserve(batch_size);
        for (std::size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            const auto* sample_logits = logits + sample_index * class_count;
            for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
                if (!std::isfinite(sample_logits[class_index])) {
                    throw std::runtime_error("Model returned non-finite logits");
                }
            }
            const auto* maximum_logit =
                std::max_element(sample_logits, sample_logits + class_count);
            double exponential_sum = 0;
            for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
                exponential_sum +=
                    std::exp(static_cast<double>(sample_logits[class_index]) - *maximum_logit);
            }
            predictions.push_back({static_cast<std::size_t>(maximum_logit - sample_logits),
                                   static_cast<float>(1.0 / exponential_sum)});
        }
        return predictions;
    }
#endif
};

InferenceEngine::InferenceEngine(const InferenceEngineConfig& config)
    : implementation_(std::make_unique<Implementation>()) {
    implementation_->config = config;
    if (config.sample_shape.empty() || config.intra_operation_threads < 1 ||
        config.mock_execution_delay.count() < 0) {
        throw std::invalid_argument("Invalid engine configuration");
    }
    for (auto dimension : config.sample_shape) {
        if (dimension <= 0 ||
            static_cast<std::size_t>(dimension) >
                std::numeric_limits<std::size_t>::max() / implementation_->input_element_count) {
            throw std::invalid_argument("Invalid or overflowing input shape");
        }
        implementation_->input_element_count *= static_cast<std::size_t>(dimension);
    }
    if (config.backend == InferenceBackend::onnx) {
#ifdef INFERENCE_RUNTIME_HAS_ONNX
        implementation_->loadModel();
#else
        throw std::runtime_error("ONNX backend is not enabled in this build");
#endif
    }
}

InferenceEngine::~InferenceEngine() = default;

std::size_t InferenceEngine::inputElementCount() const noexcept {
    return implementation_->input_element_count;
}

bool InferenceEngine::onnxAvailable() noexcept {
#ifdef INFERENCE_RUNTIME_HAS_ONNX
    return true;
#else
    return false;
#endif
}

std::vector<ModelPrediction> InferenceEngine::execute(FloatTensorView batch_input,
                                                      std::size_t batch_size) const {
    const auto sample_size = inputElementCount();
    if (batch_size == 0 ||
        batch_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        batch_size > std::numeric_limits<std::size_t>::max() / sample_size ||
        batch_input.element_count != batch_size * sample_size || batch_input.values == nullptr) {
        throw std::invalid_argument("Batch tensor size does not match the configured shape");
    }
#ifdef INFERENCE_RUNTIME_HAS_ONNX
    if (implementation_->config.backend == InferenceBackend::onnx) {
        return implementation_->executeOnnx(batch_input, batch_size);
    }
#endif
    if (implementation_->config.mock_execution_delay.count() > 0) {
        std::this_thread::sleep_for(implementation_->config.mock_execution_delay);
    }
    std::vector<ModelPrediction> predictions;
    predictions.reserve(batch_size);
    for (std::size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
        double sum = 0.0;
        for (std::size_t element_index = 0; element_index < sample_size; ++element_index) {
            const float input_value =
                batch_input.values[sample_index * sample_size + element_index];
            if (!std::isfinite(input_value)) {
                throw std::invalid_argument("Input contains a non-finite value");
            }
            sum += input_value;
        }
        const double average = sum / static_cast<double>(sample_size);
        predictions.push_back({average >= 0.0 ? 1U : 0U,
                               static_cast<float>(1.0 / (1.0 + std::exp(-std::abs(average))))});
    }
    return predictions;
}

}  // namespace inference_runtime
