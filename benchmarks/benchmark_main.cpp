#include <fstream>
#include <iomanip>
#include <iostream>

#include "benchmark.hpp"
#include "inference_runtime/configuration/command_line.hpp"
#include "inference_runtime/configuration/runtime_options.hpp"

using namespace inference_runtime;
using namespace inference_runtime::benchmark;

namespace {

void printResult(const BenchmarkConfig& config, const BenchmarkResult& result,
                 const std::string& mode) {
    const auto print = [](const char* label, const auto& value) {
        std::cout << std::left << std::setw(30) << label << value << '\n';
    };
    std::cout << "\nConfiguration: " << config.configuration << " (" << mode << ")\n";
    std::cout << std::fixed << std::setprecision(3);
    print("Backend:",
          config.service.engine.backend == InferenceBackend::mock ? "mock" : "onnx-cpu");
    if (config.service.engine.backend == InferenceBackend::mock) {
        print("Synthetic mock delay (us):", config.service.engine.mock_execution_delay.count());
    }
    print("Requests submitted:", config.requests);
    print("Requests completed:", result.completed_requests);
    print("Failed requests:", result.failed_requests);
    print("Client threads:", config.clients);
    print("Worker threads:", config.service.worker_threads);
    print("Maximum queue size:", config.service.maximum_queue_size);
    print("Maximum observed queue depth:", result.maximum_queue_depth);
    print("Maximum batch size:", config.service.batching.maximum_batch_size);
    print("Maximum batch wait (ms):", config.service.batching.maximum_batch_wait.count());
    print("Buffer reuse:", config.service.reuse_input_buffers ? "enabled" : "disabled");
    print("Elapsed time (seconds):", result.elapsed_seconds);
    print("Throughput (requests/sec):", result.completed_requests / result.elapsed_seconds);
    print("Average latency (ms):", result.average_latency_ms);
    print("P50 latency (ms):", result.percentiles.p50_ms);
    print("P95 latency (ms):", result.percentiles.p95_ms);
    print("P99 latency (ms):", result.percentiles.p99_ms);
    print("Average queue wait (ms):", result.average_queue_wait_ms);
    print("Average inference time (ms):", result.average_inference_time_ms);
    print("Average batch size:", result.average_batch_size);
}

void writeCsv(std::ostream& output, const BenchmarkConfig& config, const BenchmarkResult& result,
              const std::string& mode) {
    output << std::setprecision(10) << config.configuration << ',' << config.service.worker_threads
           << ',' << config.service.maximum_queue_size << ','
           << config.service.batching.maximum_batch_size << ','
           << config.service.batching.maximum_batch_wait.count() << ',' << config.requests << ','
           << result.completed_requests / result.elapsed_seconds << ',' << result.average_latency_ms
           << ',' << result.percentiles.p50_ms << ',' << result.percentiles.p95_ms << ','
           << result.percentiles.p99_ms << ',' << result.average_queue_wait_ms << ','
           << result.average_batch_size << ',' << result.average_inference_time_ms << ','
           << result.failed_requests << ','
           << (config.service.reuse_input_buffers ? "true" : "false") << ','
           << (config.service.engine.backend == InferenceBackend::mock ? "mock" : "onnx-cpu") << ','
           << mode << '\n';
}

}  // namespace

int main(int argument_count, char** arguments) {
    try {
        CommandLine options(argument_count, arguments);
        options.validateOptions({"--help", "--requests", "--clients", "--workers", "--queue-size",
                                 "--batch-size", "--batch-wait-ms", "--input-size", "--input-shape",
                                 "--model", "--onnx-threads", "--reuse-buffers", "--compare",
                                 "--output", "--mode", "--warmup", "--outstanding",
                                 "--mock-delay-us"});
        if (options.has("--help")) {
            std::cout
                << "inference_benchmark [--mode runtime|queue|batching|buffer] [--compare]\n"
                   "  --requests 10000 --clients 16 --workers 4 --queue-size 1024\n"
                   "  --batch-size 8 --batch-wait-ms 5 --input-size 128 --outstanding 16\n"
                   "  --warmup 64 --reuse-buffers --mock-delay-us 0 --output results.csv\n"
                   "  --model models/resnet18.onnx --input-shape 3,224,224 --onnx-threads 1\n";
            return 0;
        }
        BenchmarkConfig config;
        config.requests = options.count("--requests", 10000);
        config.clients = options.count("--clients", 16);
        config.outstanding_per_client = options.count("--outstanding", 16);
        config.warmup_requests = options.count("--warmup", 64, 0);
        config.service = parseServiceConfiguration(options, 128);
        config.service.engine.mock_execution_delay =
            std::chrono::microseconds(options.durationCount("--mock-delay-us", 0));
        std::vector<BenchmarkConfig> configurations{config};
        if (options.has("--compare")) {
            if (options.value("--mode", "runtime") != "runtime") {
                throw std::invalid_argument("--compare requires runtime mode");
            }
            configurations.clear();
            auto baseline = config;
            baseline.configuration = "single_worker_no_batching";
            baseline.service.worker_threads = 1;
            baseline.service.batching = {1, std::chrono::milliseconds(0)};
            baseline.service.reuse_input_buffers = false;
            configurations.push_back(baseline);
            baseline.configuration = "multiple_workers_no_batching";
            baseline.service.worker_threads = config.service.worker_threads;
            configurations.push_back(baseline);
            baseline.configuration = "multiple_workers_batching";
            baseline.service.batching = config.service.batching;
            configurations.push_back(baseline);
            baseline.configuration = "multiple_workers_batching_buffer_reuse";
            baseline.service.reuse_input_buffers = true;
            configurations.push_back(baseline);
        }
        std::ofstream csv_output;
        if (options.has("--output")) {
            csv_output.open(options.value("--output"));
            if (!csv_output) {
                throw std::runtime_error("Cannot open CSV output");
            }
            csv_output
                << "configuration,worker_threads,queue_capacity,batch_size,batch_wait_ms,requests,"
                   "throughput_rps,average_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,"
                   "average_queue_wait_ms,average_batch_size,average_inference_time_ms,failed_"
                   "requests,buffer_reuse,backend,mode\n";
        }
        const auto mode = options.value("--mode", "runtime");
        bool all_succeeded = true;
        for (const auto& configuration : configurations) {
            BenchmarkResult result;
            if (mode == "runtime") {
                result = benchmarkRuntime(configuration);
            } else if (mode == "queue") {
                result = benchmarkQueue(configuration);
            } else if (mode == "batching") {
                result = benchmarkBatching(configuration);
            } else if (mode == "buffer") {
                result = benchmarkBuffers(configuration);
            } else {
                throw std::invalid_argument("Unknown benchmark mode: " + mode);
            }
            printResult(configuration, result, mode);
            if (csv_output.is_open()) {
                writeCsv(csv_output, configuration, result, mode);
            }
            all_succeeded &=
                result.failed_requests == 0 && result.completed_requests == configuration.requests;
        }
        if (csv_output.is_open()) {
            csv_output.flush();
            if (!csv_output) {
                throw std::runtime_error("Failed to write CSV output");
            }
        }
        return all_succeeded ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        return 1;
    }
}
