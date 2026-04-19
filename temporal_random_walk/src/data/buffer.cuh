#ifndef DATA_BUFFER_CUH
#define DATA_BUFFER_CUH

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <new>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"

/**
 * Buffer<T> — RAII owning container for a contiguous array of T, backed by
 * either host memory (malloc/free) or device memory (cudaMalloc/cudaFree)
 * depending on a use_gpu flag fixed at construction.
 *
 * Design goals:
 *   1. Replace the 20+ "T* ptr + size_t _size + owns_data + manual
 *      allocate/clear_memory" groups scattered through the old code.
 *   2. Present a single type to the rest of the codebase so tests can
 *      continue to parameterize on a runtime use_gpu bool (same shape as
 *      today's TYPED_TEST_SUITE pattern).
 *   3. Be move-only: there is exactly one owner of any buffer at any time.
 *      Double-free is prevented by the type system, not by an owns_data
 *      flag.
 *
 * Capacity growth is geometric (doubling), like std::vector, so repeated
 * append calls do not O(n^2) on the allocator.
 *
 * Not yet stream-aware: alloc/free happen on the default stream.
 * Stream-ordered allocation is a later task.
 */
template <typename T>
class Buffer {
public:
    Buffer() noexcept : data_(nullptr), size_(0), capacity_(0), use_gpu_(false) {}

    explicit Buffer(const bool use_gpu) noexcept
        : data_(nullptr), size_(0), capacity_(0), use_gpu_(use_gpu) {}

    Buffer(const size_t n, const bool use_gpu) : Buffer(use_gpu) {
        reserve(n);
        size_ = n;
    }

    ~Buffer() {
        deallocate();
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
          use_gpu_(other.use_gpu_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            use_gpu_ = other.use_gpu_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t   size()     const noexcept { return size_; }
    size_t   capacity() const noexcept { return capacity_; }
    bool     empty()    const noexcept { return size_ == 0; }
    bool     is_gpu()   const noexcept { return use_gpu_; }

    void reserve(size_t n) {
        if (n <= capacity_) return;
        size_t new_cap = capacity_ == 0 ? n : std::max(capacity_ * 2, n);
        T* new_ptr = raw_allocate(new_cap);
        if (data_ && size_ > 0) {
            raw_copy_device_to_device_or_host_to_host(new_ptr, data_, size_);
        }
        deallocate();
        data_ = new_ptr;
        capacity_ = new_cap;
    }

    void resize(size_t n) {
        if (n > capacity_) reserve(n);
        size_ = n;
    }

    void clear() noexcept { size_ = 0; }

    void shrink_to_fit_empty() {
        deallocate();
        size_ = 0;
        capacity_ = 0;
    }

    void append_from_host(const T* src, const size_t n) {
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                data_ + size_, src, n * sizeof(T), cudaMemcpyHostToDevice));
        } else
#endif
        {
            std::memcpy(data_ + size_, src, n * sizeof(T));
        }
        size_ = new_size;
    }

    void append_from_device(const T* src, const size_t n) {
#ifdef HAS_CUDA
        if (!use_gpu_) {
            throw std::runtime_error(
                "Buffer::append_from_device called on host buffer");
        }
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            data_ + size_, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
        size_ = new_size;
#else
        (void)src; (void)n;
        throw std::runtime_error("CUDA not compiled in");
#endif
    }

    void fill(const T& value);

    void drop_front(size_t n) {
        if (n == 0 || data_ == nullptr) return;
        if (n >= size_) { size_ = 0; return; }
        const size_t remaining = size_ - n;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                data_, data_ + n, remaining * sizeof(T),
                cudaMemcpyDeviceToDevice));
        } else
#endif
        {
            std::memmove(data_, data_ + n, remaining * sizeof(T));
        }
        size_ = remaining;
    }

    std::vector<T> to_host_vector() const;

private:
    T*     data_;
    size_t size_;
    size_t capacity_;
    bool   use_gpu_;

    T* raw_allocate(size_t n) {
        if (n == 0) return nullptr;
        T* p = nullptr;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&p, n * sizeof(T)));
        } else
#endif
        {
            p = static_cast<T*>(std::malloc(n * sizeof(T)));
            if (!p) throw std::bad_alloc();
        }
        return p;
    }

    void raw_copy_device_to_device_or_host_to_host(
        T* dst, const T* src, size_t n) {
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
            return;
        }
#endif
        std::memcpy(dst, src, n * sizeof(T));
    }

    void deallocate() noexcept {
        if (!data_) return;
#ifdef HAS_CUDA
        if (use_gpu_) {
            cudaFree(data_);
            cudaGetLastError();
        } else
#endif
        {
            std::free(data_);
        }
        data_ = nullptr;
    }
};

#endif // DATA_BUFFER_CUH
