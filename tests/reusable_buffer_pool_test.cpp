#include "inference_runtime/memory/reusable_buffer_pool.hpp"

#include <gtest/gtest.h>

#include <future>
#include <optional>
#include <vector>

using inference_runtime::ReusableBufferPool;
using namespace std::chrono_literals;

TEST(ReusableBufferPool, ReturnsSameStorageThroughRAII) {
    ReusableBufferPool buffer_pool(1, 16);
    float* original_storage;
    {
        auto buffer = buffer_pool.acquire();
        original_storage = buffer.data();
        EXPECT_EQ(buffer.size(), 16U);
        EXPECT_EQ(buffer_pool.availableBufferCount(), 0U);
        buffer.data()[0] = 42;
    }
    EXPECT_EQ(buffer_pool.availableBufferCount(), 1U);
    auto reused_buffer = buffer_pool.acquire();
    EXPECT_EQ(reused_buffer.data(), original_storage);
    EXPECT_EQ(reused_buffer.data()[0], 42);
}

TEST(ReusableBufferPool, WaitingAcquisitionWakesOnReturn) {
    ReusableBufferPool buffer_pool(1, 16);
    std::optional<ReusableBufferPool::BufferHandle> buffer(buffer_pool.acquire());
    auto waiting_acquisition =
        std::async(std::launch::async, [&] { return buffer_pool.acquire(); });
    EXPECT_EQ(waiting_acquisition.wait_for(20ms), std::future_status::timeout);
    buffer.reset();
    EXPECT_EQ(waiting_acquisition.get().size(), 16U);
}

TEST(ReusableBufferPool, MoveAssignmentReturnsPreviousLease) {
    ReusableBufferPool buffer_pool(2, 4);
    auto first_buffer = buffer_pool.acquire();
    auto second_buffer = buffer_pool.acquire();
    second_buffer = std::move(first_buffer);
    EXPECT_EQ(first_buffer.data(), nullptr);
    EXPECT_EQ(buffer_pool.availableBufferCount(), 1U);
}

TEST(ReusableBufferPool, LeaseOutlivesPoolFacadeSafely) {
    std::optional<ReusableBufferPool::BufferHandle> buffer;
    {
        ReusableBufferPool buffer_pool(1, 4);
        buffer.emplace(buffer_pool.acquire());
    }
    buffer->data()[0] = 42;
    EXPECT_EQ(buffer->data()[0], 42);
    buffer.reset();
}

TEST(ReusableBufferPool, ConcurrentLeasesHaveExclusiveStorage) {
    ReusableBufferPool buffer_pool(2, 64);
    std::vector<std::future<void>> clients;
    for (int index = 0; index < 8; ++index) {
        clients.push_back(std::async(std::launch::async, [&, index] {
            for (int iteration = 0; iteration < 100; ++iteration) {
                auto buffer = buffer_pool.acquire();
                for (std::size_t element = 0; element < buffer.size(); ++element) {
                    buffer.data()[element] = static_cast<float>(index);
                }
                for (std::size_t element = 0; element < buffer.size(); ++element) {
                    EXPECT_EQ(buffer.data()[element], index);
                }
            }
        }));
    }
    for (auto& client : clients) {
        client.get();
    }
    EXPECT_EQ(buffer_pool.availableBufferCount(), 2U);
}
