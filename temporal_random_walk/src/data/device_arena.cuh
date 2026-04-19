#ifndef DATA_DEVICE_ARENA_CUH
#define DATA_DEVICE_ARENA_CUH

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <new>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"

/**
 * DeviceArena — bump allocator over a single large device buffer.
 *
 * Rationale: the rebuild pipeline in graph/temporal_graph.cu,
 * graph/edge_data.cu, graph/node_edge_index.cu allocates ~10–15 scratch
 * arrays per batch via thrust::device_vector or raw cudaMalloc. Each of
 * those allocations is a driver call. Replacing them with arena handles
 * eliminates the allocator from the per-batch path; only the occasional
 * arena growth triggers a cudaMalloc.
 *
 * Semantics:
 *   - acquire<T>(n) returns a pointer into the arena, 256-byte aligned
 *     (safe for all T we care about).
 *   - Allocations are NOT individually freed. reset() returns every
 *     outstanding handle's memory at once.
 *   - If a request would exceed the arena's current capacity, the arena
 *     grows to at least max(2*capacity, needed_bytes) and re-allocates.
 *     Growth invalidates any outstanding handles — caller must reset
 *     before growing in practice.
 *   - Move-only. Destructor frees the backing buffer.
 *
 * Not yet stream-aware. Stream-ordered allocation is a later task.
 */
class DeviceArena {
public:
    DeviceArena() noexcept
        : base_(nullptr), capacity_(0), offset_(0), use_gpu_(false) {}

    explicit DeviceArena(const bool use_gpu) noexcept
        : base_(nullptr), capacity_(0), offset_(0), use_gpu_(use_gpu) {}

    DeviceArena(const bool use_gpu, const size_t initial_capacity_bytes)
        : base_(nullptr), capacity_(0), offset_(0), use_gpu_(use_gpu) {
        if (initial_capacity_bytes > 0) {
            grow_to(initial_capacity_bytes);
        }
    }

    ~DeviceArena() {
        destroy();
    }

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    DeviceArena(DeviceArena&& other) noexcept
        : base_(other.base_), capacity_(other.capacity_),
          offset_(other.offset_), use_gpu_(other.use_gpu_) {
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    DeviceArena& operator=(DeviceArena&& other) noexcept {
        if (this != &other) {
            destroy();
            base_ = other.base_;
            capacity_ = other.capacity_;
            offset_ = other.offset_;
            use_gpu_ = other.use_gpu_;
            other.base_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    template <typename T>
    T* acquire(const size_t n) {
        if (n == 0) return nullptr;

        constexpr size_t kAlign = 256;
        const size_t aligned_offset =
            (offset_ + kAlign - 1) & ~(kAlign - 1);
        const size_t bytes_needed = n * sizeof(T);
        const size_t new_offset = aligned_offset + bytes_needed;

        if (new_offset > capacity_) {
            const size_t min_new_cap = new_offset;
            const size_t doubled = capacity_ == 0 ? 0 : capacity_ * 2;
            const size_t target_cap =
                (doubled > min_new_cap) ? doubled : min_new_cap;
            grow_to(target_cap);
            const size_t aligned_offset_new =
                (offset_ + kAlign - 1) & ~(kAlign - 1);
            offset_ = aligned_offset_new + bytes_needed;
            return reinterpret_cast<T*>(base_ + aligned_offset_new);
        }

        offset_ = new_offset;
        return reinterpret_cast<T*>(base_ + aligned_offset);
    }

    void reset() noexcept { offset_ = 0; }

    size_t capacity_bytes() const noexcept { return capacity_; }
    size_t used_bytes()     const noexcept { return offset_; }
    bool   is_gpu()         const noexcept { return use_gpu_; }

private:
    void grow_to(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        char* new_base = nullptr;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&new_base, new_capacity));
        } else
#endif
        {
            new_base = static_cast<char*>(std::malloc(new_capacity));
            if (!new_base) throw std::bad_alloc();
        }

        destroy();
        base_ = new_base;
        capacity_ = new_capacity;
        offset_ = 0;
    }

    void destroy() noexcept {
        if (!base_) return;
#ifdef HAS_CUDA
        if (use_gpu_) {
            cudaFree(base_);
            cudaGetLastError();
        } else
#endif
        {
            std::free(base_);
        }
        base_ = nullptr;
        capacity_ = 0;
        offset_ = 0;
    }

    char*  base_;
    size_t capacity_;
    size_t offset_;
    bool   use_gpu_;
};

#endif // DATA_DEVICE_ARENA_CUH
