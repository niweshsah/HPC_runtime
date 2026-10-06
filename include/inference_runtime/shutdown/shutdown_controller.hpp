#pragma once

#include <csignal>
#include <memory>

namespace inference_runtime {

class ShutdownController {
   public:
    ShutdownController();
    ~ShutdownController();
    ShutdownController(const ShutdownController&) = delete;
    ShutdownController& operator=(const ShutdownController&) = delete;
    void waitForShutdownSignal() const;
    bool shutdownSignalReceived() const noexcept;

   private:
    struct SavedHandlers;
    std::unique_ptr<SavedHandlers> saved_handlers_;
};

}  // namespace inference_runtime
