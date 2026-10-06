#include "inference_runtime/server/http_server.hpp"

#include <algorithm>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>

#include "http_routes.hpp"

namespace inference_runtime {
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using Tcp = asio::ip::tcp;

namespace {

class HttpSession : public std::enable_shared_from_this<HttpSession> {
   public:
    HttpSession(Tcp::socket socket, const HttpServerConfig& config, InferenceService& service,
                ThreadPool& handler_pool, StructuredLogger& logger)
        : stream_(std::move(socket)),
          deadline_(stream_.get_executor()),
          config_(config),
          service_(service),
          handler_pool_(handler_pool),
          logger_(logger) {}

    void readRequest() {
        deadline_.expires_after(config_.request_timeout);
        deadline_.async_wait([session = shared_from_this()](beast::error_code error) {
            if (!error) {
                session->closeConnection();
            }
        });
        request_parser_.body_limit(config_.maximum_body_bytes);
        request_parser_.header_limit(8192);
        http::async_read(stream_, read_buffer_, request_parser_,
                         [session = shared_from_this()](beast::error_code error, std::size_t) {
                             session->finishReading(error);
                         });
    }

    void closeIdleConnection() {
        if (!handler_dispatched_) {
            closeConnection();
        }
    }

   private:
    void finishReading(beast::error_code error) {
        if (closed_) {
            return;
        }
        if (error) {
            if (error == http::error::body_limit) {
                writeResponse(createJsonResponse(http::status::payload_too_large,
                                                 {{"error", "Request body exceeds limit"}}));
            } else if (error == http::error::end_of_stream ||
                       error == asio::error::operation_aborted) {
                closeConnection();
            } else {
                writeResponse(createJsonResponse(http::status::bad_request,
                                                 {{"error", "Invalid HTTP request"}}));
            }
            return;
        }
        auto request = request_parser_.release();
        if (request.method() == http::verb::get && request.target() == "/health") {
            writeResponse(buildHttpResponse(request, service_));
            return;
        }
        handler_dispatched_ = true;
        auto admitted =
            handler_pool_.trySubmit([session = shared_from_this(), request = std::move(request)] {
                HttpResponse response;
                try {
                    response = buildHttpResponse(request, session->service_);
                } catch (const std::exception& error) {
                    response = createJsonResponse(http::status::internal_server_error,
                                                  {{"error", error.what()}});
                }
                asio::post(session->stream_.get_executor(),
                           [session, response = std::move(response)]() mutable {
                               if (!session->closed_) {
                                   session->writeResponse(std::move(response));
                               }
                           });
            });
        if (!admitted) {
            logger_.write("warning", "http_request_rejected", "Handler queue is full or closing");
            writeResponse(
                createJsonResponse(http::status::service_unavailable,
                                   {{"error", "HTTP handler admission queue is full or closing"}}));
        }
    }

    void writeResponse(HttpResponse response) {
        response_ = std::move(response);
        http::async_write(stream_, response_,
                          [session = shared_from_this()](beast::error_code, std::size_t) {
                              session->closeConnection();
                          });
    }

    void closeConnection() {
        if (closed_) {
            return;
        }
        closed_ = true;
        beast::error_code ignored;
        deadline_.cancel();
        stream_.socket().shutdown(Tcp::socket::shutdown_both, ignored);
        stream_.socket().close(ignored);
    }

    beast::tcp_stream stream_;
    asio::steady_timer deadline_;
    const HttpServerConfig config_;
    InferenceService& service_;
    ThreadPool& handler_pool_;
    StructuredLogger& logger_;
    beast::flat_buffer read_buffer_;
    http::request_parser<http::string_body> request_parser_;
    HttpResponse response_;
    bool handler_dispatched_{false};
    bool closed_{false};
};

}  // namespace

struct HttpServer::Implementation {
    HttpServerConfig config;
    InferenceService& service;
    asio::io_context io_context{1};
    asio::executor_work_guard<asio::io_context::executor_type> work_guard{
        asio::make_work_guard(io_context)};
    Tcp::acceptor listener{io_context};
    asio::steady_timer accept_retry_timer{io_context};
    ThreadPool handler_pool;
    StructuredLogger logger;
    std::vector<std::weak_ptr<HttpSession>> sessions;
    std::thread io_thread;
    std::mutex lifecycle_mutex;
    bool listening_started{false};
    bool shutdown_requested{false};
    bool listener_closed{false};

    Implementation(HttpServerConfig server_config, InferenceService& inference_service)
        : config(std::move(server_config)),
          service(inference_service),
          handler_pool(config.handler_threads, config.handler_queue_capacity) {
        if (config.maximum_connections == 0 || config.maximum_body_bytes == 0 ||
            config.request_timeout.count() <= 0) {
            throw std::invalid_argument("Invalid HTTP limits");
        }
        const Tcp::endpoint endpoint{asio::ip::make_address(config.address), config.port};
        listener.open(endpoint.protocol());
        listener.set_option(asio::socket_base::reuse_address(true));
        listener.bind(endpoint);
        listener.listen(asio::socket_base::max_listen_connections);
    }

    void acceptNextConnection() {
        listener.async_accept([this](beast::error_code error, Tcp::socket socket) {
            if (listener_closed) {
                return;
            }
            if (error) {
                logger.write("warning", "http_accept_failed", error.message());
                // Resource exhaustion must not turn the accept loop into a spin loop.
                accept_retry_timer.expires_after(std::chrono::milliseconds(100));
                accept_retry_timer.async_wait([this](beast::error_code retry_error) {
                    if (!retry_error && !listener_closed) {
                        acceptNextConnection();
                    }
                });
                return;
            }
            if (!error) {
                sessions.erase(
                    std::remove_if(sessions.begin(), sessions.end(),
                                   [](const auto& session) { return session.expired(); }),
                    sessions.end());
                if (sessions.size() < config.maximum_connections) {
                    auto session = std::make_shared<HttpSession>(std::move(socket), config, service,
                                                                 handler_pool, logger);
                    sessions.push_back(session);
                    session->readRequest();
                }
            }
            acceptNextConnection();
        });
    }
};

HttpServer::HttpServer(HttpServerConfig config, InferenceService& service)
    : implementation_(std::make_unique<Implementation>(std::move(config), service)) {}
HttpServer::~HttpServer() { waitForShutdown(); }

void HttpServer::startListening() {
    auto& implementation = *implementation_;
    std::lock_guard<std::mutex> lock(implementation.lifecycle_mutex);
    if (implementation.shutdown_requested || implementation.listening_started) {
        throw std::logic_error("HTTP server cannot be started again");
    }
    implementation.acceptNextConnection();
    implementation.io_thread = std::thread([&implementation] { implementation.io_context.run(); });
    implementation.listening_started = true;
}

void HttpServer::requestShutdown() {
    auto& implementation = *implementation_;
    std::lock_guard<std::mutex> lock(implementation.lifecycle_mutex);
    if (implementation.shutdown_requested) {
        return;
    }
    implementation.shutdown_requested = true;
    implementation.service.requestShutdown();
    asio::post(implementation.io_context, [&implementation] {
        implementation.listener_closed = true;
        beast::error_code ignored;
        implementation.listener.close(ignored);
        implementation.accept_retry_timer.cancel();
        for (auto& weak_session : implementation.sessions) {
            if (auto session = weak_session.lock()) {
                session->closeIdleConnection();
            }
        }
    });
    implementation.handler_pool.requestShutdown();
}

void HttpServer::waitForShutdown() {
    requestShutdown();
    auto& implementation = *implementation_;
    std::lock_guard<std::mutex> lock(implementation.lifecycle_mutex);
    implementation.handler_pool.waitForWorkers();
    implementation.work_guard.reset();
    if (implementation.io_thread.joinable()) {
        implementation.io_thread.join();
    }
}

unsigned short HttpServer::listeningPort() const {
    return implementation_->listener.local_endpoint().port();
}

}  // namespace inference_runtime
