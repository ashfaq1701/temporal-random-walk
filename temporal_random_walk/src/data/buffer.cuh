#ifndef DATA_BUFFER_CUH
#define DATA_BUFFER_CUH

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

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
 *
 * Pinned host allocation: when pinned_host=true is passed to the
 * constructor AND use_gpu=false AND CUDA is available, host memory is
 * allocated via cudaMallocHost. This roughly doubles effective PCIe
 * bandwidth for subsequent H<->D copies and lets cudaMemcpyAsync truly
 * overlap kernel work. The flag is ignored on CUDA-less builds and on
 * device-resident buffers (use_gpu=true).
 *
 * On cudaMallocHost failure (pinned pool exhausted), the allocator falls
 * back to plain malloc and logs once per process. The caller still gets
 * working memory; copies just lose the bandwidth advantage.
 */

// Process-wide counter of currently-allocated pinned host bytes. Used to
// emit a one-time warning when the footprint exceeds a soft threshold.
inline std::atomic<size_t> g_total_pinned_host_bytes{0};
inline constexpr size_t kPinnedHostWarnThreshold =
    4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4 GiB

template <typename T>
class Buffer {
public:
    Buffer() noexcept
        : data_(nullptr), size_(0), capacity_(0),
          use_gpu_(false), pinned_host_(false) {}

    explicit Buffer(const bool use_gpu) noexcept
        : data_(nullptr), size_(0), capacity_(0),
          use_gpu_(use_gpu), pinned_host_(false) {}

    Buffer(const bool use_gpu, const bool pinned_host) noexcept
        : data_(nullptr), size_(0), capacity_(0),
          use_gpu_(use_gpu),
          // Pinning only applies to host buffers. On device buffers the
          // flag is meaningless, so collapse it to false for clarity.
          pinned_host_(!use_gpu && pinned_host) {}

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
          use_gpu_(other.use_gpu_), pinned_host_(other.pinned_host_) {
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
            pinned_host_ = other.pinned_host_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    HOST DEVICE T*       data()       noexcept { return data_; }
    HOST DEVICE const T* data() const noexcept { return data_; }
    HOST DEVICE size_t   size()     const noexcept { return size_; }
    HOST DEVICE size_t   capacity() const noexcept { return capacity_; }
    HOST DEVICE bool     empty()    const noexcept { return size_ == 0; }
    HOST DEVICE bool     is_gpu()   const noexcept { return use_gpu_; }
    HOST DEVICE bool     is_pinned_host() const noexcept { return pinned_host_; }

    void reserve(size_t n) {
        if (n <= capacity_) return;
        size_t new_cap = capacity_ == 0 ? n : std::max(capacity_ * 2, n);
        // Snapshot allocator state for the OLD buffer before raw_allocate
        // runs: a cudaMallocHost fallback inside raw_allocate flips
        // pinned_host_ to false, which would otherwise mislead deallocate()
        // into calling std::free on a pinned pointer.
        T* const   old_data       = data_;
        const size_t old_capacity = capacity_;
        const bool old_pinned     = pinned_host_;
        const bool old_use_gpu    = use_gpu_;

        T* new_ptr = raw_allocate(new_cap);
        if (old_data && size_ > 0) {
            raw_copy_device_to_device_or_host_to_host(new_ptr, old_data, size_);
        }
        deallocate_ptr(old_data, old_capacity, old_use_gpu, old_pinned);
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

#ifdef HAS_CUDA
    void append_from_host(const T* src, const size_t n, cudaStream_t stream) {
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                data_ + size_, src, n * sizeof(T),
                cudaMemcpyHostToDevice, stream));
        } else {
            std::memcpy(data_ + size_, src, n * sizeof(T));
        }
        size_ = new_size;
    }
#endif

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

    // Async D->H copy (no internal sync). On host buffers it falls back to
    // std::memcpy. Caller must sync the passed stream before reading dst.
    void copy_to_host_async(T* dst, size_t n
#ifdef HAS_CUDA
                            , cudaStream_t stream = 0
#endif
                            ) const {
        if (n == 0) return;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                dst, data_, n * sizeof(T),
                cudaMemcpyDeviceToHost, stream));
            return;
        }
#endif
        std::memcpy(dst, data_, n * sizeof(T));
    }

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
    bool   pinned_host_;

    T* raw_allocate(size_t n) {
        if (n == 0) return nullptr;
        T* p = nullptr;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&p, n * sizeof(T)));
            return p;
        }
        if (pinned_host_) {
            const cudaError_t err = cudaMallocHost(
                reinterpret_cast<void**>(&p), n * sizeof(T));
            if (err == cudaSuccess) {
                const size_t new_total =
                    g_total_pinned_host_bytes.fetch_add(
                        n * sizeof(T), std::memory_order_relaxed)
                    + n * sizeof(T);
                if (new_total > kPinnedHostWarnThreshold) {
                    static std::once_flag warn_once;
                    std::call_once(warn_once, []() {
                        std::cerr << "[Buffer] Total pinned host memory "
                                     "exceeds 4 GiB. Consider reducing "
                                     "batch size or walk count if you see "
                                     "cudaMallocHost failures."
                                  << std::endl;
                    });
                }
                return p;
            }
            // Fallback path. Disable pinning for this buffer's subsequent
            // grows so we don't keep hitting the failing allocator.
            static std::once_flag fallback_once;
            std::call_once(fallback_once, [err]() {
                std::cerr << "[Buffer] cudaMallocHost failed ("
                          << cudaGetErrorString(err)
                          << "); falling back to malloc. H<->D copies "
                             "will be slower." << std::endl;
            });
            pinned_host_ = false;
            (void)cudaGetLastError();
        }
#endif
        p = static_cast<T*>(std::malloc(n * sizeof(T)));
        if (!p) throw std::bad_alloc();
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

    // Free `ptr` using the allocator implied by (use_gpu, pinned_host). Used
    // both by deallocate() (which reads the live members) and by reserve()
    // (which must use a snapshot taken before raw_allocate may have flipped
    // pinned_host_ on a fallback).
    static void deallocate_ptr(
        T* ptr, [[maybe_unused]] size_t cap,
        [[maybe_unused]] bool use_gpu,
        [[maybe_unused]] bool pinned_host) noexcept {
        if (!ptr) return;
#ifdef HAS_CUDA
        if (use_gpu) {
            cudaFree(ptr);
            cudaGetLastError();
            return;
        }
        if (pinned_host) {
            cudaFreeHost(ptr);
            cudaGetLastError();
            g_total_pinned_host_bytes.fetch_sub(
                cap * sizeof(T), std::memory_order_relaxed);
            return;
        }
#endif
        std::free(ptr);
    }

    void deallocate() noexcept {
        deallocate_ptr(data_, capacity_, use_gpu_, pinned_host_);
        data_ = nullptr;
    }
};

#endif // DATA_BUFFER_CUH
