#include "inference_runtime/server/http_server.hpp"

#include <gtest/gtest.h>

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

using namespace inference_runtime;
using namespace std::chrono_literals;
namespace http = boost::beast::http;

namespace {
unsigned int sendRequest(unsigned short port, http::verb method, const char* target,
                         const std::string& body = {}) {
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::socket connection(io_context);
    connection.connect({boost::asio::ip::make_address("127.0.0.1"), port});
    http::request<http::string_body> request{method, target, 11};
    request.set(http::field::host, "localhost");
    request.body() = body;
    request.prepare_payload();
    http::write(connection, request);
    boost::beast::flat_buffer response_buffer;
    http::response<http::string_body> response;
    http::read(connection, response_buffer, response);
    return response.result_int();
}

struct ReleaseExecutionOnExit {
    std::promise<void>& release_execution;
    ~ReleaseExecutionOnExit() {
        try {
            release_execution.set_value();
        } catch (const std::future_error&) {
        }
    }
};
}  // namespace

TEST(HttpServer, SaturatedHandlersRejectRequestsWhileHealthRemainsResponsive) {
    std::promise<void> execution_started;
    std::promise<void> release_execution;
    auto release_signal = release_execution.get_future().share();
    std::atomic<bool> first_execution{true};
    InferenceServiceConfig runtime_config;
    runtime_config.batching = {1, 0ms};
    InferenceService service(runtime_config, [&](FloatTensorView, std::size_t batch_size) {
        if (first_execution.exchange(false)) {
            execution_started.set_value();
        }
        release_signal.wait();
        return std::vector<ModelPrediction>(batch_size, {1, 0.9F});
    });
    HttpServerConfig http_config;
    http_config.port = 0;
    http_config.handler_threads = 1;
    http_config.handler_queue_capacity = 1;
    HttpServer server(http_config, service);
    const auto port = server.listeningPort();
    server.startListening();
    std::vector<std::future<unsigned int>> responses;
    // Release before future destructors wait if a socket operation throws.
    ReleaseExecutionOnExit release_on_exit{release_execution};
    responses.push_back(std::async(std::launch::async, [&] {
        return sendRequest(port, http::verb::post, "/predict", R"({"input":[1,2,3]})");
    }));
    execution_started.get_future().wait();
    for (int index = 0; index < 8; ++index) {
        responses.push_back(std::async(std::launch::async, [&] {
            return sendRequest(port, http::verb::post, "/predict", R"({"input":[1,2,3]})");
        }));
    }
    EXPECT_EQ(sendRequest(port, http::verb::get, "/health"), 200U);
    std::vector<unsigned int> completed_statuses;
    for (auto& response : responses) {
        if (response.wait_for(100ms) == std::future_status::ready) {
            completed_statuses.push_back(response.get());
        }
    }
    EXPECT_FALSE(completed_statuses.empty());
    for (auto status : completed_statuses) {
        EXPECT_EQ(status, 503U);
    }
    release_execution.set_value();
    for (auto& response : responses) {
        if (response.valid()) {
            EXPECT_EQ(response.get(), 200U);
        }
    }
    server.waitForShutdown();
    service.waitForShutdown();
    EXPECT_EQ(service.getMetrics().completed_requests, 2U);
}
