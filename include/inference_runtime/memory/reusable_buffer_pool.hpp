#pragma once

#include <cstddef>
#include <memory>

namespace inference_runtime {

class ReusableBufferPool {
    struct PoolState;

   public:
    class BufferHandle {
       public:
        ~BufferHandle();
        BufferHandle(BufferHandle&& other) noexcept;
        BufferHandle& operator=(BufferHandle&& other) noexcept;
        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;
        float* data() noexcept;
        std::size_t size() const noexcept;

       private:
        friend class ReusableBufferPool;
        BufferHandle(std::shared_ptr<PoolState> state, std::size_t buffer_index);
        void returnBuffer() noexcept;
        std::shared_ptr<PoolState> state_;
        std::size_t buffer_index_;
    };

    ReusableBufferPool(std::size_t buffer_count, std::size_t elements_per_buffer);
    ~ReusableBufferPool();
    ReusableBufferPool(const ReusableBufferPool&) = delete;
    ReusableBufferPool& operator=(const ReusableBufferPool&) = delete;
    BufferHandle acquire();
    std::size_t availableBufferCount() const;

   private:
    // Leases share only pool state, not the pool facade. Storage survives the facade
    // until the last lease returns; free slots are exclusively owned by the state.
    std::shared_ptr<PoolState> state_;
};

}  // namespace inference_runtime
