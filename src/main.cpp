#include <iostream>

#include "inference_runtime/configuration/command_line.hpp"
#include "inference_runtime/configuration/runtime_options.hpp"
#include "inference_runtime/server/http_server.hpp"
#include "inference_runtime/shutdown/shutdown_controller.hpp"

using namespace inference_runtime;

namespace {
HttpServerConfig parseHttpConfiguration(const CommandLine& options) {
    HttpServerConfig config;
    config.address = options.value("--address", "127.0.0.1");
    const auto port = options.count("--port", 8080, 0);
    if (port > 65535) {
        throw std::invalid_argument("Port exceeds 65535");
    }
    config.port = static_cast<unsigned short>(port);
    config.handler_threads = options.count("--http-threads", 4);
    config.handler_queue_capacity = options.count("--http-queue-size", 128);
    config.maximum_connections = options.count("--max-connections", 256);
    config.maximum_body_bytes = options.count("--max-body-bytes", 16 * 1024 * 1024);
    config.request_timeout =
        std::chrono::seconds(options.durationCount("--timeout-seconds", 30, 1));
    return config;
}
}  // namespace

int main(int argument_count, char** arguments) {
    try {
        CommandLine options(argument_count, arguments);
        options.validateOptions({"--help", "--address", "--port", "--workers", "--queue-size",
                                 "--batch-size", "--batch-wait-ms", "--input-size", "--input-shape",
                                 "--model", "--onnx-threads", "--reuse-buffers", "--http-threads",
                                 "--http-queue-size", "--max-connections", "--max-body-bytes",
                                 "--timeout-seconds"});
        if (options.has("--help")) {
            std::cout << "inference_server [--address 127.0.0.1 --port 8080]\n"
                         "  --workers 4 --queue-size 1024 --batch-size 8 --batch-wait-ms 5\n"
                         "  --input-size 3 --reuse-buffers\n"
                         "  --model models/resnet18.onnx --input-shape 3,224,224 --onnx-threads 1\n"
                         "  --http-threads 4 --http-queue-size 128 --max-connections 256\n"
                         "  --max-body-bytes 16777216 --timeout-seconds 30\n";
            return 0;
        }
        ShutdownController shutdown_controller;
        const auto config = parseServiceConfiguration(options, 3);
        const auto http_config = parseHttpConfiguration(options);
        InferenceService service(config);
        HttpServer server(http_config, service);
        const auto listening_port = server.listeningPort();
        server.startListening();
        std::cout << "Listening on " << http_config.address << ':' << listening_port << std::endl;
        shutdown_controller.waitForShutdownSignal();
        server.requestShutdown();
        service.waitForShutdown();
        server.waitForShutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Server error: " << error.what() << '\n';
        return 1;
    }
}
