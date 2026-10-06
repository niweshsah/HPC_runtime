# High-Performance Asynchronous Inference Runtime

A C++17 systems project that accepts concurrent inference requests, applies
bounded backpressure, forms dynamic batches, executes them on fixed worker
threads, and delivers results through futures. ONNX Runtime supplies CPU model
execution; a deterministic mock lets the concurrency architecture be built and
tested without any model or Python environment.

The purpose is to make ownership, synchronization, shutdown, and performance
tradeoffs easy to review. This is a production-style reference implementation
with explicit limits and tested failure paths. It is not a claim of deployment
readiness or a universal inference framework.

## Build and run

Requirements: CMake 3.24+, a C++17 compiler, and POSIX threads/signals. Linux and
macOS are supported. Python 3 is optional for HTTP process tests, and separate
Python packages are needed only to export the pretrained model.

```bash
cd high-performance-inference-runtime
mkdir build
cd build
cmake .. \
  -DINFERENCE_RUNTIME_BUILD_TESTS=ON \
  -DINFERENCE_RUNTIME_BUILD_BENCHMARKS=ON \
  -DINFERENCE_RUNTIME_USE_ONNX=OFF
cmake --build . -j
ctest --output-on-failure
./inference_benchmark --requests 10000 --clients 16 --batch-size 8 --batch-wait-ms 5
./inference_server --port 8080 --reuse-buffers
```

For performance measurements, configure `-DCMAKE_BUILD_TYPE=Release`. The default
server listens on `127.0.0.1:8080` with a three-element mock input. Use `--help` on
either executable for its complete options. CMake fails visibly when a required
dependency is unavailable; ONNX builds never silently fall back to mock execution.

Installed dependencies are preferred. Missing GoogleTest 1.15.2, Boost 1.87.0,
and nlohmann/json 3.11.3 sources are fetched from pinned, SHA-256-verified archives.
The large Boost source archive is needed only for the HTTP target. The runtime
library and benchmarks have no Boost or JSON dependency.

Ubuntu 24.04 can build using packages without configuration-time downloads:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake python3 libboost-dev libgtest-dev nlohmann-json3-dev
cmake -S . -B build -DINFERENCE_RUNTIME_FETCH_DEPENDENCIES=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Package-only mode requires Boost >= 1.80, GoogleTest >= 1.14, and JSON >= 3.11.
On macOS, install CMake with `brew install cmake`. A workspace-local CMake
installation is also available in this generated checkout:

```bash
export PATH="$PWD/.tools/bin:$PATH"
```

## Architecture and request lifecycle

```text
 Client threads                         HTTP clients
       |                                     |
       |                         asynchronous Boost.Beast sockets
       |                                     |
       |                         bounded HTTP handler thread pool
       +------------------+------------------+
                          v
               +---------------------+
               | InferenceService    |
               | validation + IDs    |
               +---------------------+
                          |
                          v
               +---------------------+
               | Bounded Request     | <-- producer backpressure
               | Queue               |
               +---------------------+
                          |
                          v
               +---------------------+
               | Batch Scheduler     | <-- size or oldest deadline
               +---------------------+
                          |
                          v
               +---------------------+
               | Bounded Batch Tasks |
               | Fixed Thread Pool   |
               +---------------------+
                          |
                 +--------+---------+
                 v                  v
          input buffer lease   fresh batch allocation
                 +--------+---------+
                          v
               +---------------------+
               | Inference Engine    |
               | ONNX CPU / mock     |
               +---------------------+
                          |
                          v
               PredictionResult + metrics
                          |
                          v
                 promise -> client future
```

1. `submit()` assigns a unique ID and records arrival with `steady_clock`.
2. It validates element count and finite values, then waits for request capacity.
3. The scheduler moves requests into a batch and submits a bounded worker task.
4. A worker packs contiguous input, executes the backend, and builds one result per request.
5. It records completion metrics and fulfills each promise. The caller waits or
   performs independent work through its future.

HTTP-specific types remain outside `inference_runtime`. No ONNX headers appear
in public runtime headers. The service owns its scheduler and worker lifetimes;
the engine owns its environment/session, and a request owns its tensor/promise.
Shared batch ownership exists briefly across dispatch so failed task admission
can still fulfill promises. No threads are detached.

## C++ concepts demonstrated

| Concept | Concrete use |
| --- | --- |
| RAII and composition | Sessions, threads, sockets, signal registrations, buffer leases |
| Templates | Move-aware bounded queue and typed thread-pool submission |
| Futures/promises | Asynchronous predictions and packaged task exception transport |
| Move semantics | Requests, input tensors, task closures, and exclusive buffer leases |
| Smart pointers | Engine implementation ownership and shared pool lifetime state |
| Mutexes/condition variables | Predicate-based queue, buffer, metrics, and join coordination |
| Atomics | Request IDs and an idempotent shutdown request flag |
| STL/chrono | Batch containers, sorting, reductions, tensor packing, monotonic durations |

## Queue, backpressure, and batching

`BlockingQueue<T>` rejects zero capacity and prevents queue copying. Its queue
state, shutdown flag, and high-water depth are protected by one mutex. Separate
condition variables notify consumers about items and producers about space.
Wait predicates handle spurious wakeups. Values move at successful admission
and removal; copyable lvalues are supported through a convenience overload.

Shutdown closes insertion and wakes both sides. Failed insertion preserves a
move-only producer item. Consumers continue draining existing items; `pop()`
returns `nullopt` only after a closed queue empties. `popUntil()` also returns
`nullopt` on a deadline, and `tryPush()` provides nonblocking task admission.
The queue must outlive threads using it; closing a queue does not join them.

The inference service implements **BLOCK_PRODUCER**. A full request queue blocks
the calling thread; invalid input and shutdown return ready failure futures.
Requests are never silently discarded. The batch task queue is bounded as well,
with pending capacity equal to inference worker count. One extra batch may be
held by the scheduler while dispatch is blocked. Caller-owned tensors awaiting
admission remain outside the queue bound, so callers must also bound concurrency.

The scheduler blocks for its first request, then waits until the batch is full
or its oldest request's arrival-based deadline expires. It shortens the deadline
if an earlier-arriving producer enqueues later. Expired requests dispatch promptly;
shutdown flushes partial batches without waiting for their batching deadline.
The deadline controls batch formation, not an inference-start latency guarantee:
saturated workers can delay dispatch. FIFO admission is not a fairness guarantee
between blocked producers, and completion order can differ from request order.

Defaults: four inference workers, 1,024 request slots, eight requests per batch,
and a five-millisecond maximum batch wait. Batch size one disables batching.
Thread-pool tasks must not recursively make blocking submissions into their own
pool. Joining methods are owner-thread operations, not worker-task operations.

## Memory reuse

`--reuse-buffers` preallocates one contiguous batch input buffer per inference
worker. Every buffer can hold `maximum_batch_size * input_element_count` floats.
Workers acquire a move-only lease after their task begins and return it on scope
exit, including exceptions. A partial batch uses only its occupied prefix.

The pool stores fixed buffers and free indices behind a mutex/condition variable.
Returning a lease does not allocate because the free-index container is reserved
at construction. Leases share pool state, so a lease can safely outlive the pool
facade; no acquisition may use a destroyed facade. Input request vectors, output
tensors, and futures still allocate. This is deliberately not a custom allocator
or a claim of an allocation-free pipeline.

## HTTP interface and logging

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":[0.1,0.2,0.3]}'
curl http://127.0.0.1:8080/metrics
```

`/predict` returns request ID, class index, confidence, status, error message, and
timings in milliseconds. It accepts already-preprocessed float32 tensors, not
image uploads. `/metrics` returns JSON; `/health` reports readiness and required
input element count.

Malformed JSON/tensors return 400, oversized bodies 413, unsupported methods
405, unknown routes 404, and backend execution failures 500. A full/closing HTTP
handler queue returns 503. This is adapter admission control, distinct from the
runtime's blocking producer policy. HTTP rejections before runtime submission
are not counted as runtime rejected requests. Health checks bypass the handler
pool and remain responsive during inference backpressure.

There is one asynchronous network thread and a separate fixed handler pool
(default four threads, 128 queued handlers). Connections are capped at 256;
connections exceeding that limit are closed before parsing. Bodies are capped
at 16 MiB, headers at 8 KiB, and each connection has a 30-second deadline. Tune
limits for available memory. Each connection serves one request and closes.
A timed-out/disconnected client does not cancel accepted inference.

Logs use `level=... event=... detail="..."` on stderr for lifecycle events,
batch failures, and HTTP admission rejection. Successful predictions are not
logged individually. TLS, authentication, and multipart image preprocessing are
outside this project's scope.

## Metrics and timing definitions

Counters track accepted, rejected, completed, and failed requests, plus executed
batches. **Completed** means an accepted request received a terminal result;
failed requests are a subset of completed requests. Rejections are never accepted.
Depth/high-water describe the request queue, not pending batches or HTTP sessions.

| Metric | Meaning |
| --- | --- |
| Queue waiting time | Submission entry to worker task start; includes producer blocking, batching, and pending-task wait |
| Inference time | Synchronous backend execution; batch duration is attributed to each request in that batch |
| End-to-end latency | Submission entry to result construction immediately before metrics recording and promise fulfillment |
| Average batch size | Requests in executed batches divided by executed batches |
| Service throughput | Terminal completions / service lifetime; frozen once workers are joined |
| P50/P95/P99 | Nearest-rank percentiles over the most recent 65,536 completions |

Input packing contributes to total latency but not queue wait or backend time.
HTTP parsing, network time, caller input construction/copying before `submit()`,
and client future consumption are outside runtime latency. Per-request inference
averages are batch-size weighted and are not per-batch execution averages.

For nearest rank, sort N samples and select the one-based rank `ceil(p * N)`.
Empty sets yield zero values with sample count zero. The collector keeps a fixed
rolling window while lifetime sums/counters remain cumulative. Benchmarks retain
every measured request's latency and compute exact workload percentiles separately.
Snapshots are synchronized but queue/counter views are not one transaction; under
concurrent submission they can briefly disagree. Final drained counters reconcile.

An average can hide a small group of very slow requests. P95 and P99 expose that
tail, which often determines user-visible responsiveness under contention.

## Graceful shutdown

SIGINT and SIGTERM handlers only store to a lock-free atomic flag, verified by
`static_assert`. An atomic also synchronizes with the main thread when POSIX
delivers the signal to a worker thread.
The main thread sleeps between checks and performs actual C++ shutdown outside
the handler. Signal registrations are restored by RAII; only one process-level
`ShutdownController` should exist at a time.

Shutdown closes inference admission and wakes blocked producers, stops accepting
connections, closes incomplete idle reads, drains accepted request/partial batch
work, closes the task queue, joins scheduler/workers/handlers/network threads,
and then destroys inference resources. Pending HTTP responses can finish while
their deadlines remain valid. Shutdown calls are idempotent, and concurrent
service join callers are serialized. The destructor performs the same drain.

There is no forced cancellation of a hung backend call: safe shutdown requires
backend execution to return. Backend exceptions become explicit failed results
and workers remain available. Catastrophic allocation failure is not a guaranteed
recoverable operating mode.

## ONNX Runtime

See [models/README.md](models/README.md) for export and preprocessing instructions.
Use a CPU SDK with matching headers/library. Tested baseline: 1.22.0.
This generated checkout also contains the downloaded macOS ARM64 SDK at
`.tools/onnxruntime-sdk`; use its absolute path for `ONNXRUNTIME_ROOT` locally.

```bash
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DINFERENCE_RUNTIME_USE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-sdk
cmake --build build-onnx -j
INFERENCE_RUNTIME_TEST_MODEL="$PWD/models/resnet18.onnx" \
  ctest --test-dir build-onnx --output-on-failure
./build-onnx/inference_server --model models/resnet18.onnx --reuse-buffers
./build-onnx/inference_benchmark --model models/resnet18.onnx \
  --requests 64 --clients 4 --workers 2 --batch-size 4 --reuse-buffers
```

Enabling ONNX compiles the capability; pass `--model` to select it at runtime.
Without a model argument, executables still use mock execution. The engine
accepts one float32 input with a dynamic batch and fixed configured sample shape,
and one `[batch, classes]` logits output. It checks model metadata at startup,
packs NCHW inputs, validates returned shapes, and computes stable softmax.
The mock predicts from the sign of mean input and is deterministic across batches.

Workers share a CPU session, with independent call tensors/output values. ONNX
intra-operation threads default to one and can be set with `--onnx-threads`.
Increasing runtime workers and internal model threads simultaneously can
oversubscribe the CPU. [ONNX threading documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
explains those controls. GPU providers, arbitrary outputs, and static batch
models are intentionally unsupported.

## Tests and build options

GoogleTest covers queue blocking/draining, move-only ownership, concurrent
producers/consumers, typed task results/exceptions, partial/deadline batches,
buffer reuse/exclusive ownership/lifetime, overload wakeups, backend failure,
metrics percentiles, HTTP routes, and HTTP saturation with a responsive health
endpoint. The stdlib-only Python process test checks concurrent HTTP requests,
limits, idle connections, SIGINT, and SIGTERM. Concurrency tests use promises and
gates; broad time bounds check blocking rather than exact scheduling intervals.

The real-model test is explicitly skipped unless ONNX is compiled and
`INFERENCE_RUNTIME_TEST_MODEL` points to a model. Missing Python skips registration
of the process test; all C++ tests remain available.

| CMake option | Default |
| --- | --- |
| `INFERENCE_RUNTIME_BUILD_TESTS` | ON |
| `INFERENCE_RUNTIME_BUILD_BENCHMARKS` | ON |
| `INFERENCE_RUNTIME_BUILD_SERVER` | ON |
| `INFERENCE_RUNTIME_USE_ONNX` | OFF |
| `INFERENCE_RUNTIME_FETCH_DEPENDENCIES` | ON |
| `INFERENCE_RUNTIME_WARNINGS_AS_ERRORS` | OFF |
| `INFERENCE_RUNTIME_SANITIZER` | empty; `address` or `thread` |

Project targets use C++17 without compiler extensions and `-Wall -Wextra
-Wpedantic` on GCC/Clang. Third-party source warnings are not promoted to project
errors. Separate targets are `inference_runtime`, `inference_http`,
`inference_server`, `inference_tests`, and `inference_benchmark`.

```bash
cmake -S . -B build-asan -DINFERENCE_RUNTIME_SANITIZER=address \
  -DINFERENCE_RUNTIME_WARNINGS_AS_ERRORS=ON
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure

# Run ThreadSanitizer separately from AddressSanitizer, preferably on Linux.
cmake -S . -B build-tsan -DINFERENCE_RUNTIME_SANITIZER=thread \
  -DINFERENCE_RUNTIME_BUILD_SERVER=OFF -DINFERENCE_RUNTIME_BUILD_BENCHMARKS=OFF
cmake --build build-tsan -j
ctest --test-dir build-tsan --output-on-failure
```

Format C++ with clang-format 19 and the checked-in `.clang-format` file.
With GCC, enabling ThreadSanitizer on Boost.Asio while also enabling `-Werror`
can fail on the third-party `atomic_thread_fence` diagnostic. The server-disabled
command above avoids it. For a full HTTP sanitizer run, use
`-DCMAKE_CXX_FLAGS=-Wno-error=tsan`; the unsupported-fence warning remains visible.

## Benchmarks and interpretation

```bash
./build/inference_benchmark --requests 10000 --clients 16 \
  --workers 4 --queue-size 1024 --batch-size 8 --batch-wait-ms 5 \
  --input-size 128 --outstanding 16 --warmup 64 --output results.csv

# Same workload across the four requested runtime configurations.
scripts/run_benchmark.sh --requests 10000 --clients 16 --output comparison.csv

# Isolate queue operations, batch formation, and allocation/lease costs.
./build/inference_benchmark --mode queue --requests 100000
./build/inference_benchmark --mode batching --requests 100000
./build/inference_benchmark --mode buffer --input-size 150528 --requests 1000
./build/inference_benchmark --mode buffer --input-size 150528 --requests 1000 --reuse-buffers
```

Use `--compare` directly for: single worker without batching, multiple workers
without batching, multiple workers with batching, and the same batching with
buffer reuse. CSV includes configuration, worker/queue/batch settings, request
count, throughput, latency percentiles/averages, failures, reuse, backend, and mode.

Runtime clients call the service directly, each keeping a bounded window of
outstanding futures. `--outstanding 1` provides a closed-loop latency workload;
larger windows generate enough concurrent work to exercise batching. All submitted
results are consumed. Startup and warmup are excluded from measured runtime counts,
latencies, batches, and throughput. Queue high-water remains a lifetime diagnostic
and includes warmup. Throughput ends at the last client's workload completion,
before result aggregation, sorting, printing, or shutdown.

Queue/batching/buffer microbenchmarks include their thread creation/collection
overhead and do not run model inference. In those modes, zero inference time
means no inference was performed. Buffer mode counts buffer operations as requests
and checks all written values through a checksum. Avoid LTO for that microbenchmark.
`--mock-delay-us` adds an explicitly reported synthetic backend delay; default zero.
Mock throughput measures runtime overhead and cannot predict ResNet18 throughput.

### Measured example

Captured on 2026-10-06: Apple M5, macOS 27.0.1 ARM64, Apple Clang 21,
CMake Release (`-O3`), deterministic mock, 128 input floats, zero synthetic delay.
This was a development workstation run with other validation activity; it is not
an isolated hardware study. Each configuration submitted/completed 10,000
requests with zero failures, 16 clients, 16 outstanding requests per client,
64 warmup requests, and request capacity 1,024.

| Configuration | Workers | Batch | Throughput (requests/s) | Average (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_worker_no_batching | 1 | 1 | 73185.8 | 3.420 | 5.092 | 5.496 |
| multiple_workers_no_batching | 4 | 1 | 144541.6 | 1.717 | 2.522 | 3.031 |
| multiple_workers_batching | 4 | 8 | 593931.5 | 0.386 | 0.602 | 0.710 |
| multiple_workers_batching_buffer_reuse | 4 | 8 | 764068.4 | 0.301 | 0.451 | 1.175 |

The final configuration printed this measured output:

```text
Configuration: multiple_workers_batching_buffer_reuse (runtime)
Backend:                      mock
Synthetic mock delay (us):    0
Requests submitted:           10000
Requests completed:           10000
Failed requests:              0
Client threads:               16
Worker threads:               4
Maximum queue size:           1024
Maximum observed queue depth: 256
Maximum batch size:           8
Maximum batch wait (ms):      5
Buffer reuse:                 enabled
Elapsed time (seconds):       0.013
Throughput (requests/sec):    764068.429
Average latency (ms):         0.301
P50 latency (ms):             0.272
P95 latency (ms):             0.451
P99 latency (ms):             1.175
Average queue wait (ms):      0.290
Average inference time (ms):  0.002
Average batch size:           8.000
```

Raw captured output and CSV remain in `build-release/comparison.txt` and
`build-release/comparison.csv` in this checkout. They are ignored build artifacts.

Repeat runs on an otherwise idle machine and report distributions. The example
is one observation, not a performance guarantee or evidence that memory pooling
always wins. Synthetic inputs measure execution, not model accuracy.

## Docker

The multi-stage Ubuntu 24.04 build runs its tests and ships only executables,
required runtime libraries, and a non-root user. The default image uses mock.

```bash
docker build -f docker/Dockerfile -t inference-runtime .
docker run --rm --name inference-runtime -p 8080:8080 inference-runtime
docker stop inference-runtime
docker run --rm --entrypoint inference_benchmark inference-runtime \
  --requests 10000 --clients 16 --compare

docker build -f docker/Dockerfile --build-arg USE_ONNX=ON -t inference-runtime-onnx .
docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" inference-runtime-onnx \
  --address 0.0.0.0 --model /models/resnet18.onnx --reuse-buffers
```

The ONNX image downloads a checksum-pinned CPU SDK for Linux x64 or ARM64.
Models are mounted explicitly. Allow Docker enough shutdown time for the chosen
workload; `docker stop --time 60` permits a longer drain.

## Performance tradeoffs and future work

```text
larger batches -> better amortization and potentially better throughput
               -> potentially worse waiting latency
smaller batches -> lower batch waiting latency
                -> potentially worse throughput
```

Bounded queues cap admitted work and expose overload, but capacity trades memory
and latency against burst tolerance. The single scheduler simplifies ordering
and shutdown; it can become a measured bottleneck. Metrics use locks and sorting
outside the inference path; recording still adds cost. Pool leases reduce batch
input allocation, but synchronization can outweigh allocation savings for small
inputs. Request packing remains a copy. More workers can help concurrent CPU
inference or can increase contention and hurt tail latency.

Potential next steps should follow measurements: admission timeouts/rejection
policy, open-loop HTTP load generation, cancellation/deadlines, per-stage timing,
histogram metrics, preprocessing, model-provider extensions, and deployment
integration. Lock-free queues and custom allocators require evidence that simpler
structures are the bottleneck.

Project code is MIT licensed. Dependencies and pretrained weights retain their
respective licenses.
