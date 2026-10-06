#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace inference_runtime {

// Closing prevents insertion, but consumers drain already accepted work.
template <typename T>

class BlockingQueue {
   public:
    explicit BlockingQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("Queue capacity must be positive");
        }
    }

    BlockingQueue(const BlockingQueue&) = delete;
    BlockingQueue& operator=(const BlockingQueue&) = delete;

    bool push(const T& item) {
        T copied_item(item);
        return push(std::move(copied_item));
    }

    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        space_available_.wait(lock,
                              [this] { return shutdown_requested_ || items_.size() < capacity_; });
        // Keep ownership with the producer if admission has closed.
        if (shutdown_requested_) {
            return false;
        }
        insertItem(std::move(item));
        return true;
    }

    bool tryPush(T&& item) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (shutdown_requested_ || items_.size() == capacity_) {
            return false;
        }
        insertItem(std::move(item));
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        item_available_.wait(lock, [this] { return shutdown_requested_ || !items_.empty(); });
        return removeItem();
    }

    std::optional<T> popUntil(std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        item_available_.wait_until(lock, deadline,
                                   [this] { return shutdown_requested_ || !items_.empty(); });
        return removeItem();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_requested_ = true;
        }
        space_available_.notify_all();
        item_available_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return items_.size();
    }
    bool empty() const { return size() == 0; }
    bool isShutdown() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return shutdown_requested_;
    }
    std::size_t maximumObservedSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return maximum_observed_size_;
    }

   private:
    void insertItem(T&& item) {
        items_.push_back(std::move(item));
        if (items_.size() > maximum_observed_size_) {
            maximum_observed_size_ = items_.size();
        }
        item_available_.notify_one();
    }

    std::optional<T> removeItem() {
        // Empty after close means every accepted item has been drained.
        if (items_.empty()) {
            return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        space_available_.notify_one();
        return item;
    }

    const std::size_t capacity_;
    mutable std::mutex queue_mutex_;
    std::condition_variable item_available_;
    std::condition_variable space_available_;
    std::deque<T> items_;
    std::size_t maximum_observed_size_{0};
    bool shutdown_requested_{false};
};

}  // namespace inference_runtime
