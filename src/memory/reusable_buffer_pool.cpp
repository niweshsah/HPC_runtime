#include "inference_runtime/memory/reusable_buffer_pool.hpp"

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace inference_runtime {

struct ReusableBufferPool::PoolState {
    std::mutex pool_mutex;
    std::condition_variable buffer_available;
    std::vector<std::vector<float>> buffers;
    std::vector<std::size_t> available_indices;
};

ReusableBufferPool::ReusableBufferPool(std::size_t buffer_count, std::size_t elements_per_buffer)
    : state_(std::make_shared<PoolState>()) {
    if (buffer_count == 0 || elements_per_buffer == 0) {
        throw std::invalid_argument("Buffer pool dimensions must be positive");
    }
    state_->buffers.reserve(buffer_count);
    state_->available_indices.reserve(buffer_count);
    for (std::size_t index = 0; index < buffer_count; ++index) {
        state_->buffers.emplace_back(elements_per_buffer);
        state_->available_indices.push_back(index);
    }
}
ReusableBufferPool::~ReusableBufferPool() = default;

ReusableBufferPool::BufferHandle ReusableBufferPool::acquire() {
    std::unique_lock<std::mutex> lock(state_->pool_mutex);
    state_->buffer_available.wait(lock, [this] { return !state_->available_indices.empty(); });
    const auto buffer_index = state_->available_indices.back();
    state_->available_indices.pop_back();
    return BufferHandle(state_, buffer_index);
}

std::size_t ReusableBufferPool::availableBufferCount() const {
    std::lock_guard<std::mutex> lock(state_->pool_mutex);
    return state_->available_indices.size();
}

ReusableBufferPool::BufferHandle::BufferHandle(std::shared_ptr<PoolState> state,
                                               std::size_t buffer_index)
    : state_(std::move(state)), buffer_index_(buffer_index) {}
ReusableBufferPool::BufferHandle::~BufferHandle() { returnBuffer(); }
ReusableBufferPool::BufferHandle::BufferHandle(BufferHandle&& other) noexcept
    : state_(std::move(other.state_)), buffer_index_(other.buffer_index_) {}
ReusableBufferPool::BufferHandle& ReusableBufferPool::BufferHandle::operator=(
    BufferHandle&& other) noexcept {
    if (this != &other) {
        returnBuffer();
        state_ = std::move(other.state_);
        buffer_index_ = other.buffer_index_;
    }
    return *this;
}
void ReusableBufferPool::BufferHandle::returnBuffer() noexcept {
    if (state_) {
        {
            std::lock_guard<std::mutex> lock(state_->pool_mutex);
            // Reserved at construction: returning a slot cannot allocate.
            state_->available_indices.push_back(buffer_index_);
        }
        state_->buffer_available.notify_one();
        state_.reset();
    }
}
float* ReusableBufferPool::BufferHandle::data() noexcept {
    return state_ ? state_->buffers[buffer_index_].data() : nullptr;
}
std::size_t ReusableBufferPool::BufferHandle::size() const noexcept {
    return state_ ? state_->buffers[buffer_index_].size() : 0;
}

}  // namespace inference_runtime
