#include "inference_runtime/inference/inference_engine.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

using namespace inference_runtime;

TEST(InferenceEngine, MockIsDeterministicAndValidatesShape) {
    InferenceEngine engine({});
    std::vector<float> input{1, 2, 3, -1, -2, -3};
    const auto predictions = engine.execute({input.data(), input.size()}, 2);
    ASSERT_EQ(predictions.size(), 2U);
    EXPECT_EQ(predictions[0].class_index, 1U);
    EXPECT_EQ(predictions[1].class_index, 0U);
    EXPECT_EQ(predictions[0].confidence, predictions[1].confidence);
    EXPECT_THROW(static_cast<void>(engine.execute({input.data(), 5}, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(engine.execute({nullptr, 3}, 1)), std::invalid_argument);
}

TEST(InferenceEngine, OnnxSelectionNeverFallsBackToMock) {
    InferenceEngineConfig config;
    config.backend = InferenceBackend::onnx;
    config.model_path = "/nonexistent/inference-runtime-model.onnx";
    EXPECT_THROW(InferenceEngine engine(config), std::exception);
}

TEST(InferenceEngine, RealModelBatchMatchesSinglePrediction) {
    const char* model_path = std::getenv("INFERENCE_RUNTIME_TEST_MODEL");
    if (!InferenceEngine::onnxAvailable() || model_path == nullptr) {
        GTEST_SKIP() << "Set INFERENCE_RUNTIME_TEST_MODEL in an ONNX-enabled build";
    }
    InferenceEngineConfig config;
    config.backend = InferenceBackend::onnx;
    config.model_path = model_path;
    config.sample_shape = {3, 224, 224};
    InferenceEngine engine(config);
    std::vector<float> input(engine.inputElementCount() * 3, 0.25F);
    const auto single = engine.execute({input.data(), engine.inputElementCount()}, 1);
    const auto batched = engine.execute({input.data(), input.size()}, 3);
    ASSERT_EQ(single.size(), 1U);
    ASSERT_EQ(batched.size(), 3U);
    for (const auto& prediction : batched) {
        EXPECT_EQ(prediction.class_index, single[0].class_index);
        EXPECT_LT(prediction.class_index, 1000U);
        EXPECT_GT(prediction.confidence, 0);
        EXPECT_LE(prediction.confidence, 1);
        EXPECT_NEAR(prediction.confidence, single[0].confidence, 1e-4);
    }
}
