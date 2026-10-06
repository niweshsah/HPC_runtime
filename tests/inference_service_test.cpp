#include "inference_runtime/server/inference_service.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <set>

using namespace inference_runtime;
using namespace std::chrono_literals;

TEST(InferenceService, SuccessfulDeterministicMockPrediction) {
    InferenceService service({});
    auto result = service.submit({0.1F, 0.2F, 0.3F}).get();
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(result.predicted_class_index, 1U);
    EXPECT_GT(result.confidence_score, 0.5F);
    EXPECT_GE(result.total_latency.count(), result.inference_time.count());
    service.waitForShutdown();
    const auto metrics = service.getMetrics();
    EXPECT_EQ(metrics.accepted_requests, 1U);
    EXPECT_EQ(metrics.completed_requests, 1U);
    EXPECT_EQ(metrics.failed_requests, 0U);
    EXPECT_EQ(metrics.latency_percentiles.sample_count, 1U);
}

TEST(InferenceService, ConcurrentRequestsHaveUniqueIdsAndComplete) {
    InferenceServiceConfig config;
    config.maximum_queue_size = 7;
    InferenceService service(config);
    std::vector<std::future<std::vector<std::uint64_t>>> clients;
    for (int index = 0; index < 8; ++index) {
        clients.push_back(std::async(std::launch::async, [&] {
            std::vector<std::uint64_t> request_ids;
            for (int request = 0; request < 30; ++request) {
                const auto result = service.submit({1, 2, 3}).get();
                EXPECT_TRUE(result.succeeded());
                request_ids.push_back(result.request_id);
            }
            return request_ids;
        }));
    }
    std::set<std::uint64_t> request_ids;
    for (auto& client : clients) {
        for (auto request_id : client.get()) {
            request_ids.insert(request_id);
        }
    }
    service.waitForShutdown();
    EXPECT_EQ(request_ids.size(), 240U);
    EXPECT_EQ(service.getMetrics().completed_requests, 240U);
    EXPECT_LE(service.getMetrics().maximum_queue_depth, 7U);
}

TEST(InferenceService, RejectsInvalidInputAndSubmissionsAfterShutdown) {
    InferenceService service({});
    EXPECT_EQ(service.submit({}).get().status, PredictionStatus::invalid_input);
    EXPECT_EQ(service.submit({1, 2, std::numeric_limits<float>::infinity()}).get().status,
              PredictionStatus::invalid_input);
    service.requestShutdown();
    EXPECT_EQ(service.submit({1, 2, 3}).get().status, PredictionStatus::shutting_down);
    service.waitForShutdown();
    EXPECT_EQ(service.getMetrics().rejected_requests, 3U);
    EXPECT_EQ(service.getMetrics().completed_requests, 0U);
}

TEST(InferenceService, BackendFailuresResolveEveryPromiseAndDoNotStopWorkers) {
    InferenceService service({}, [](FloatTensorView, std::size_t) -> std::vector<ModelPrediction> {
        throw std::runtime_error("Injected backend failure");
    });
    std::vector<std::future<PredictionResult>> results;
    for (int index = 0; index < 12; ++index) {
        results.push_back(service.submit({1, 2, 3}));
    }
    service.waitForShutdown();
    for (auto& result : results) {
        auto failure = result.get();
        EXPECT_EQ(failure.status, PredictionStatus::execution_failed);
        EXPECT_EQ(failure.error_message, "Injected backend failure");
    }
    EXPECT_EQ(service.getMetrics().failed_requests, 12U);
}

TEST(InferenceService, ShutdownWakesBlockedProducerAndDrainsAcceptedWork) {
    InferenceServiceConfig config;
    config.worker_threads = 1;
    config.maximum_queue_size = 1;
    config.batching = {1, 0ms};
    std::promise<void> execution_started;
    std::promise<void> release_execution;
    auto release_signal = release_execution.get_future().share();
    std::atomic<bool> first_execution{true};
    InferenceService service(config, [&](FloatTensorView, std::size_t batch_size) {
        if (first_execution.exchange(false)) {
            execution_started.set_value();
        }
        release_signal.wait();
        return std::vector<ModelPrediction>(batch_size, {1, 0.9F});
    });
    std::vector<std::future<PredictionResult>> accepted_results;
    accepted_results.push_back(service.submit({1, 2, 3}));
    execution_started.get_future().wait();
    // One executing batch, one pending task, one dispatching batch, one queued request.
    for (int index = 0; index < 3; ++index) {
        accepted_results.push_back(service.submit({1, 2, 3}));
    }
    auto blocked_producer =
        std::async(std::launch::async, [&] { return service.submit({1, 2, 3}); });
    EXPECT_EQ(blocked_producer.wait_for(20ms), std::future_status::timeout);
    service.requestShutdown();
    EXPECT_EQ(blocked_producer.get().get().status, PredictionStatus::shutting_down);
    release_execution.set_value();
    service.waitForShutdown();
    for (auto& result : accepted_results) {
        EXPECT_TRUE(result.get().succeeded());
    }
    EXPECT_EQ(service.getMetrics().accepted_requests, 4U);
    EXPECT_EQ(service.getMetrics().completed_requests, 4U);
}

TEST(InferenceService, ConcurrentShutdownCallersJoinSafely) {
    InferenceService service({});
    auto result = service.submit({1, 2, 3});
    auto first_shutdown = std::async(std::launch::async, [&] { service.waitForShutdown(); });
    auto second_shutdown = std::async(std::launch::async, [&] { service.waitForShutdown(); });
    first_shutdown.get();
    second_shutdown.get();
    EXPECT_TRUE(result.get().succeeded());
}

TEST(InferenceService, BufferReusePreservesPredictionsForPartialBatches) {
    InferenceServiceConfig config;
    config.reuse_input_buffers = true;
    InferenceService service(config);
    auto positive_result = service.submit({1, 2, 3});
    auto negative_result = service.submit({-1, -2, -3});
    service.waitForShutdown();
    EXPECT_EQ(positive_result.get().predicted_class_index, 1U);
    EXPECT_EQ(negative_result.get().predicted_class_index, 0U);
}

TEST(InferenceService, EmptyExceptionDiagnosticStillProducesFailure) {
    InferenceService service({}, [](FloatTensorView, std::size_t) -> std::vector<ModelPrediction> {
        throw std::runtime_error("");
    });
    const auto result = service.submit({1, 2, 3}).get();
    EXPECT_EQ(result.status, PredictionStatus::execution_failed);
    EXPECT_FALSE(result.error_message.empty());
    service.waitForShutdown();
    EXPECT_EQ(service.getMetrics().failed_requests, 1U);
}
