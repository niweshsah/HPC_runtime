#include "inference_runtime/inference/batch_scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace inference_runtime {



// Batch Scheduler is used to group inference requests into batches for more efficient processing. It waits for a maximum batch wait time or until the maximum batch size is reached, whichever comes first.

// constructor
// sets request_queue_ and config_ member variables
// throws invalid_argument exception if maximum_batch_size is 0 or maximum_batch_wait is negative
BatchScheduler::BatchScheduler(BlockingQueue<InferenceRequest>& request_queue,
                               BatchSchedulerConfig config)
    : request_queue_(request_queue), config_(config) {
        
    if (config.maximum_batch_size == 0 || config.maximum_batch_wait.count() < 0) {
        throw std::invalid_argument("Invalid batch scheduler configuration");
    }
}

RequestBatch BatchScheduler::buildNextBatch() {
    RequestBatch batch; // Create a new RequestBatch object to hold the requests
    batch.reserve(config_.maximum_batch_size);


    auto first_request = request_queue_.pop(); // Pop the first request from the queue, blocking if necessary
    if (!first_request) { // If the queue is empty and shutdown has been requested, return an empty batch
        return batch;
    }

    // Calculate the deadline for the batch based on the first request's received time and the maximum batch wait time
    auto deadline = first_request->request_received_at + config_.maximum_batch_wait;
    batch.push_back(std::move(*first_request));
    while (batch.size() < config_.maximum_batch_size) {
        // After closing, skip waiting but fill batches from accepted queued work.
        if (!request_queue_.isShutdown() && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        auto next_request = request_queue_.popUntil(deadline);
        if (!next_request) {
            break;
        }
        // Concurrent producers can reach the queue in a different order from
        // arrival timestamps, especially after waiting for admission capacity.
        deadline =
            std::min(deadline, next_request->request_received_at + config_.maximum_batch_wait);
        batch.push_back(std::move(*next_request));
    }
    return batch;
}

}  // namespace inference_runtime
