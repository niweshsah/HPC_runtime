#pragma once

#include <iostream>
#include <mutex>
#include <string>

namespace inference_runtime {

// Ownership stays with the executable/service; no mutable global logger.
class StructuredLogger {
   public:
    void write(const char* level, const char* event,
               const std::string& detail = {}) const noexcept {
        try {
            std::lock_guard<std::mutex> lock(output_mutex_);
            std::clog << "level=" << level << " event=" << event;
            if (!detail.empty()) {
                std::clog << " detail=\"";
                for (char character : detail) {
                    if (character == '\n' || character == '\r') {
                        std::clog << ' ';
                    } else {
                        if (character == '"' || character == '\\') {
                            std::clog << '\\';
                        }
                        std::clog << character;
                    }
                }
                std::clog << '"';
            }
            std::clog << '\n';
        } catch (...) {
            // Diagnostics must not interfere with fulfilling request promises.
        }
    }

   private:
    mutable std::mutex output_mutex_;
};

}  // namespace inference_runtime
