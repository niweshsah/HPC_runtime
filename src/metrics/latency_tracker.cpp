#include "inference_runtime/metrics/latency_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace inference_runtime {

LatencyTracker::LatencyTracker(std::size_t sample_capacity) : samples_(sample_capacity) {
    if (sample_capacity == 0) {
        throw std::invalid_argument("Latency capacity must be positive");
    }
}

void LatencyTracker::recordLatency(double latency_ms) {
    if (!std::isfinite(latency_ms) || latency_ms < 0) {
        throw std::invalid_argument("Latency must be finite and nonnegative");
    }
    std::lock_guard<std::mutex> lock(samples_mutex_);
    samples_[next_sample_index_] = latency_ms;
    next_sample_index_ = (next_sample_index_ + 1) % samples_.size();
    sample_count_ = std::min(sample_count_ + 1, samples_.size());
}

LatencyPercentiles LatencyTracker::getPercentiles() const {
    std::vector<double> snapshot;
    {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        snapshot.assign(samples_.begin(), samples_.begin() + sample_count_);
    }
    return calculatePercentiles(std::move(snapshot));
}

LatencyPercentiles LatencyTracker::calculatePercentiles(std::vector<double> samples) {
    if (samples.empty()) {
        return {};
    }
    std::sort(samples.begin(), samples.end());
    // Nearest rank is one-based: ceil(percentile * N), never percentile * (N - 1).
    const auto percentile = [&samples](double fraction) {
        const auto rank = static_cast<std::size_t>(std::ceil(fraction * samples.size()));
        return samples[rank - 1];
    };
    return {percentile(0.50), percentile(0.95), percentile(0.99), samples.size()};
}

}  // namespace inference_runtime
