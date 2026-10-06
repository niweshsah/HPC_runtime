#pragma once

#include "inference_runtime/concurrency/blocking_queue.hpp"
#include "inference_runtime/inference/inference_request.hpp"

namespace inference_runtime {

struct BatchSchedulerConfig {
    std::size_t maximum_batch_size{8};
    std::chrono::milliseconds maximum_batch_wait{5};
};

class BatchScheduler {
   public:
    BatchScheduler(BlockingQueue<InferenceRequest>& request_queue, BatchSchedulerConfig config);
    RequestBatch buildNextBatch();

   private:
    BlockingQueue<InferenceRequest>& request_queue_;
    const BatchSchedulerConfig config_;
};

}  // namespace inference_runtime
