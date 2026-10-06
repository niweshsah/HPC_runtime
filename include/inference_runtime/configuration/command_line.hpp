#pragma once

#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference_runtime {

inline std::size_t parseUnsignedCount(const std::string& text) {
    std::size_t result{0};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Expected an unsigned integer, received: " + text);
    }
    return result;
}

class CommandLine {
   public:
    CommandLine(int argument_count, char** arguments) {
        for (int index = 1; index < argument_count; ++index) {
            const std::string option = arguments[index];
            if (option.rfind("--", 0) != 0) {
                throw std::invalid_argument("Expected a named option: " + option);
            }
            std::string value = "true";
            if (index + 1 < argument_count &&
                std::string(arguments[index + 1]).rfind("--", 0) != 0) {
                value = arguments[++index];
            }
            if (!options_.emplace(option, value).second) {
                throw std::invalid_argument("Duplicate option: " + option);
            }
        }
    }
    bool has(const std::string& option) const { return options_.count(option) != 0; }
    std::string value(const std::string& option, const std::string& fallback = {}) const {
        const auto found = options_.find(option);
        return found == options_.end() ? fallback : found->second;
    }
    std::size_t count(const std::string& option, std::size_t fallback,
                      std::size_t minimum = 1) const {
        const auto result = has(option) ? parseUnsignedCount(value(option)) : fallback;
        if (result < minimum) {
            throw std::invalid_argument(option + " is below its minimum");
        }
        return result;
    }
    std::int64_t durationCount(const std::string& option, std::size_t fallback,
                               std::size_t minimum = 0) const {
        const auto result = count(option, fallback, minimum);
        if (result >
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / 1000000000) {
            throw std::invalid_argument(option + " exceeds a safe steady-clock duration");
        }
        return static_cast<std::int64_t>(result);
    }
    void validateOptions(std::initializer_list<const char*> supported) const {
        for (const auto& option : options_) {
            bool recognized = false;
            for (const auto* name : supported) {
                if (option.first == name) {
                    recognized = true;
                }
            }
            if (!recognized) {
                throw std::invalid_argument("Unknown option: " + option.first);
            }
        }
    }

   private:
    std::map<std::string, std::string> options_;
};

inline std::vector<std::int64_t> parseTensorShape(const std::string& text) {
    std::vector<std::int64_t> dimensions;
    std::size_t position = 0;
    do {
        const auto separator = text.find(',', position);
        const auto dimension = parseUnsignedCount(text.substr(position, separator - position));
        if (dimension == 0 ||
            dimension > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("Tensor dimensions must be positive int64 values");
        }
        dimensions.push_back(static_cast<std::int64_t>(dimension));
        if (separator == std::string::npos) {
            break;
        }
        position = separator + 1;
    } while (position <= text.size());
    return dimensions;
}

}  // namespace inference_runtime
