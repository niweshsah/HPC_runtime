#include "http_routes.hpp"

#include <gtest/gtest.h>

using namespace inference_runtime;
namespace http = boost::beast::http;

TEST(HttpRoutes, ValidPredictionAndMetricsAreJson) {
    InferenceService service({});
    HttpRequest request{http::verb::post, "/predict", 11};
    request.body() = R"({"input":[0.1,0.2,0.3]})";
    const auto response = buildHttpResponse(request, service);
    EXPECT_EQ(response.result(), http::status::ok);
    const auto prediction = nlohmann::json::parse(response.body());
    EXPECT_EQ(prediction["status"], "success");
    const auto metrics = buildHttpResponse({http::verb::get, "/metrics", 11}, service);
    EXPECT_EQ(nlohmann::json::parse(metrics.body())["completed_requests"], 1);
}

TEST(HttpRoutes, ValidatesTensorElementsShapeAndMethods) {
    InferenceService service({});
    for (const char* body : {"invalid json", R"({"input":[1]})", R"({"input":[1,2,"3"]})",
                             R"({"input":[1,2,1e99]})", "[]"}) {
        HttpRequest request{http::verb::post, "/predict", 11};
        request.body() = body;
        EXPECT_EQ(buildHttpResponse(request, service).result(), http::status::bad_request);
    }
    EXPECT_EQ(buildHttpResponse({http::verb::get, "/predict", 11}, service).result(),
              http::status::method_not_allowed);
    EXPECT_EQ(buildHttpResponse({http::verb::get, "/unknown", 11}, service).result(),
              http::status::not_found);
    service.requestShutdown();
    EXPECT_EQ(buildHttpResponse({http::verb::get, "/health", 11}, service).result(),
              http::status::service_unavailable);
}

TEST(HttpRoutes, BackendFailureBecomesExplicitHttpError) {
    InferenceService service({}, [](FloatTensorView, std::size_t) -> std::vector<ModelPrediction> {
        throw std::runtime_error("backend failed");
    });
    HttpRequest request{http::verb::post, "/predict", 11};
    request.body() = R"({"input":[1,2,3]})";
    const auto response = buildHttpResponse(request, service);
    EXPECT_EQ(response.result(), http::status::internal_server_error);
    EXPECT_EQ(nlohmann::json::parse(response.body())["status"], "execution_failed");
}
