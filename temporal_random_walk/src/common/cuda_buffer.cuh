#ifndef CUDA_BUFFER_CUH
#define CUDA_BUFFER_CUH

#include <cstddef>
#include <utility>

#include "macros.cuh"
#include "memory.cuh"

// Move-only RAII wrapper over a host- or device-allocated buffer.
// Frees with cudaFree when use_gpu is true, with free() otherwise.
// Designed to replace raw T* members paired with manual allocate_memory /
// clear_memory calls and owns_* flags.
template <typename T>
class CudaBuffer {
public:
    HOST CudaBuffer() noexcept = default;

    HOST CudaBuffer(const size_t size, const bool use_gpu)
        : size_(size), use_gpu_(use_gpu) {
        if (size_ > 0) {
            allocate_memory(&ptr_, size_, use_gpu_);
        }
    }

    HOST ~CudaBuffer() { reset(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    HOST CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), use_gpu_(other.use_gpu_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    HOST CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            use_gpu_ = other.use_gpu_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    HOST void reset() noexcept {
        if (ptr_) {
            clear_memory(&ptr_, use_gpu_);
        }
        size_ = 0;
    }

    // Relinquishes ownership; the caller becomes responsible for freeing the
    // returned pointer with the matching allocator (free / cudaFree).
    [[nodiscard]] HOST T* release() noexcept {
        T* released = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return released;
    }

    [[nodiscard]] HOST T* data() const noexcept { return ptr_; }
    [[nodiscard]] HOST T* get() const noexcept { return ptr_; }
    [[nodiscard]] HOST size_t size() const noexcept { return size_; }
    [[nodiscard]] HOST bool use_gpu() const noexcept { return use_gpu_; }
    [[nodiscard]] HOST explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
    bool use_gpu_ = false;
};

#endif // CUDA_BUFFER_CUH
