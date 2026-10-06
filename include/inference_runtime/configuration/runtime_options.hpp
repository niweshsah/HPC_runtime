#pragma once

#include "inference_runtime/configuration/command_line.hpp"
#include "inference_runtime/server/inference_service.hpp"

namespace inference_runtime {

inline InferenceServiceConfig parseServiceConfiguration(const CommandLine& options,
                                                        std::size_t default_input_size) {
    if (options.has("--input-size") && options.has("--input-shape")) {
        throw std::invalid_argument("Choose either --input-size or --input-shape");
    }
    InferenceServiceConfig config;
    config.worker_threads = options.count("--workers", 4);
    config.maximum_queue_size = options.count("--queue-size", 1024);
    config.batching.maximum_batch_size = options.count("--batch-size", 8);
    config.batching.maximum_batch_wait =
        std::chrono::milliseconds(options.durationCount("--batch-wait-ms", 5));
    config.reuse_input_buffers = options.has("--reuse-buffers");
    config.engine.sample_shape =
        parseTensorShape(options.has("--input-shape")
                             ? options.value("--input-shape")
                             : std::to_string(options.count("--input-size", default_input_size)));
    if (options.has("--model")) {
        config.engine.backend = InferenceBackend::onnx;
        config.engine.model_path = options.value("--model");
        if (!options.has("--input-shape") && !options.has("--input-size")) {
            config.engine.sample_shape = {3, 224, 224};
        }
    }
    const auto onnx_threads = options.count("--onnx-threads", 1);
    if (onnx_threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ONNX thread count is too large");
    }
    config.engine.intra_operation_threads = static_cast<int>(onnx_threads);
    return config;
}

}  // namespace inference_runtime
