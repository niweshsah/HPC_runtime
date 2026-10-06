#include "inference_runtime/shutdown/shutdown_controller.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace {
static_assert(std::atomic<bool>::is_always_lock_free,
              "Signal handling requires a lock-free atomic flag");
std::atomic<bool> shutdown_signal_received{false};
extern "C" void recordShutdownSignal(int) {
    shutdown_signal_received.store(true, std::memory_order_relaxed);
}
}  // namespace

namespace inference_runtime {

struct ShutdownController::SavedHandlers {
    struct sigaction interrupt_handler{};
    struct sigaction termination_handler{};
};

ShutdownController::ShutdownController() : saved_handlers_(std::make_unique<SavedHandlers>()) {
    shutdown_signal_received.store(false, std::memory_order_relaxed);
    struct sigaction handler{};
    handler.sa_handler = recordShutdownSignal;
    sigemptyset(&handler.sa_mask);
    sigaddset(&handler.sa_mask, SIGINT);
    sigaddset(&handler.sa_mask, SIGTERM);
    if (sigaction(SIGINT, &handler, &saved_handlers_->interrupt_handler) != 0) {
        throw std::runtime_error("Cannot install SIGINT handler");
    }
    if (sigaction(SIGTERM, &handler, &saved_handlers_->termination_handler) != 0) {
        sigaction(SIGINT, &saved_handlers_->interrupt_handler, nullptr);
        throw std::runtime_error("Cannot install SIGTERM handler");
    }
}
ShutdownController::~ShutdownController() {
    sigaction(SIGINT, &saved_handlers_->interrupt_handler, nullptr);
    sigaction(SIGTERM, &saved_handlers_->termination_handler, nullptr);
}
bool ShutdownController::shutdownSignalReceived() const noexcept {
    return shutdown_signal_received.load(std::memory_order_relaxed);
}
void ShutdownController::waitForShutdownSignal() const {
    std::mutex wait_mutex;
    std::condition_variable signal_check;
    std::unique_lock<std::mutex> lock(wait_mutex);
    // The signal handler cannot notify a C++ condition variable. Sleep between
    // checks instead of spinning or invoking unsafe runtime code from the handler.
    while (!shutdownSignalReceived()) {
        signal_check.wait_for(lock, std::chrono::milliseconds(100));
    }
}

}  // namespace inference_runtime
