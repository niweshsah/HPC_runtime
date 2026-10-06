#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace inference_runtime {

struct LatencyPercentiles {
    double p50_ms{0};
    double p95_ms{0};
    double p99_ms{0};
    std::size_t sample_count{0};
};

class LatencyTracker {
   public:
    explicit LatencyTracker(std::size_t sample_capacity = 65536);
    void recordLatency(double latency_ms);
    LatencyPercentiles getPercentiles() const;
    static LatencyPercentiles calculatePercentiles(std::vector<double> samples);

   private:
    mutable std::mutex samples_mutex_;
    std::vector<double> samples_;
    std::size_t next_sample_index_{0};
    std::size_t sample_count_{0};
};

}  // namespace inference_runtime
