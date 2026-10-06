#include "http_routes.hpp"

#include <cmath>

namespace inference_runtime {
namespace http = boost::beast::http;
using Json = nlohmann::json;

HttpResponse createJsonResponse(http::status status, Json body) {
    HttpResponse response{status, 11};
    response.set(http::field::content_type, "application/json");
    response.set(http::field::server, "inference-runtime/1.0");
    // One request per connection keeps shutdown and session ownership easy to inspect.
    response.keep_alive(false);
    response.body() = body.dump();
    response.prepare_payload();
    return response;
}

namespace {
Json metricsJson(const RuntimeMetrics& metrics) {
    return {{"accepted_requests", metrics.accepted_requests},
            {"rejected_requests", metrics.rejected_requests},
            {"completed_requests", metrics.completed_requests},
            {"failed_requests", metrics.failed_requests},
            {"current_queue_depth", metrics.current_queue_depth},
            {"maximum_queue_depth", metrics.maximum_queue_depth},
            {"total_batches_executed", metrics.total_batches_executed},
            {"average_batch_size", metrics.average_batch_size},
            {"average_latency_ms", metrics.average_latency_ms},
            {"average_inference_latency_ms", metrics.average_inference_latency_ms},
            {"average_queue_wait_ms", metrics.average_queue_wait_ms},
            {"p50_latency_ms", metrics.latency_percentiles.p50_ms},
            {"p95_latency_ms", metrics.latency_percentiles.p95_ms},
            {"p99_latency_ms", metrics.latency_percentiles.p99_ms},
            {"latency_sample_count", metrics.latency_percentiles.sample_count},
            {"latency_sample_capacity", 65536},
            {"throughput_rps", metrics.throughput_rps}};
}

HttpResponse predict(const HttpRequest& request, InferenceService& service) {
    if (service.isShuttingDown()) {
        return createJsonResponse(http::status::service_unavailable,
                                  {{"error", "Service is shutting down"}});
    }
    std::vector<float> input_tensor;
    try {
        const auto body = Json::parse(request.body());
        if (!body.is_object() || !body.contains("input") || !body["input"].is_array() ||
            body["input"].size() != service.inputElementCount()) {
            return createJsonResponse(
                http::status::bad_request,
                {{"error", "Expected an input array matching the configured tensor shape"}});
        }
        input_tensor.reserve(service.inputElementCount());
        for (const auto& element : body["input"]) {
            if (!element.is_number()) {
                throw std::invalid_argument("Input values must be numeric");
            }
            const float input_value = element.get<float>();
            if (!std::isfinite(input_value)) {
                throw std::invalid_argument("Input values must be finite float32");
            }
            input_tensor.push_back(input_value);
        }
    } catch (const std::exception& error) {
        return createJsonResponse(http::status::bad_request, {{"error", error.what()}});
    }
    const auto result = service.submit(std::move(input_tensor)).get();
    http::status response_status = http::status::ok;
    if (result.status == PredictionStatus::invalid_input) {
        response_status = http::status::bad_request;
    } else if (result.status == PredictionStatus::shutting_down) {
        response_status = http::status::service_unavailable;
    } else if (!result.succeeded()) {
        response_status = http::status::internal_server_error;
    }
    return createJsonResponse(response_status,
                              {{"request_id", result.request_id},
                               {"predicted_class_index", result.predicted_class_index},
                               {"confidence_score", result.confidence_score},
                               {"status", predictionStatusName(result.status)},
                               {"error_message", result.error_message},
                               {"queue_waiting_time_ms", result.queue_waiting_time.count()},
                               {"inference_time_ms", result.inference_time.count()},
                               {"total_latency_ms", result.total_latency.count()}});
}
}  // namespace

HttpResponse buildHttpResponse(const HttpRequest& request, InferenceService& service) {
    if (request.target() == "/health" && request.method() == http::verb::get) {
        return createJsonResponse(
            service.isShuttingDown() ? http::status::service_unavailable : http::status::ok,
            {{"status", service.isShuttingDown() ? "shutting_down" : "healthy"},
             {"input_element_count", service.inputElementCount()}});
    }
    if (request.target() == "/metrics" && request.method() == http::verb::get) {
        return createJsonResponse(http::status::ok, metricsJson(service.getMetrics()));
    }
    if (request.target() == "/predict" && request.method() == http::verb::post) {
        return predict(request, service);
    }
    if (request.target() == "/predict" || request.target() == "/metrics" ||
        request.target() == "/health") {
        return createJsonResponse(http::status::method_not_allowed,
                                  {{"error", "Method is not supported"}});
    }
    return createJsonResponse(http::status::not_found, {{"error", "Route not found"}});
}

}  // namespace inference_runtime
