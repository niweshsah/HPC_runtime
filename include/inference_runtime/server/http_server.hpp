#pragma once

#include <memory>
#include <string>

#include "inference_runtime/server/inference_service.hpp"

namespace inference_runtime {

struct HttpServerConfig {
    std::string address{"127.0.0.1"};
    unsigned short port{8080};
    std::size_t handler_threads{4};
    std::size_t handler_queue_capacity{128};
    std::size_t maximum_connections{256};
    std::size_t maximum_body_bytes{16 * 1024 * 1024};
    std::chrono::seconds request_timeout{30};
};

class HttpServer {
   public:
    HttpServer(HttpServerConfig config, InferenceService& service);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    void startListening();
    void requestShutdown();
    void waitForShutdown();
    unsigned short listeningPort() const;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace inference_runtime
