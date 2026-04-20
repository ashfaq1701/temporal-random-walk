#ifndef MEMORY_H
#define MEMORY_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>

#include "error_handlers.cuh"
#include "macros.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#endif

namespace fill_memory_detail {

// Predicates for byte-pattern fast paths. Compare bytewise to an all-0
// buffer (for cudaMemsetAsync(0)) or all-0xFF buffer (for
// cudaMemsetAsync(0xFF)). Works for any trivially-copyable T:
//   int(0), size_t(0), double(+0.0) -> all-0 bytes
//   int(-1), size_t::max()          -> all-0xFF bytes
// and cheap enough to run on every fill call.
template <typename T>
inline bool is_zero_value(const T& v) {
    constexpr size_t N = sizeof(T);
    alignas(T) unsigned char zero[N] = {};
    return std::memcmp(&v, zero, N) == 0;
}

template <typename T>
inline bool is_all_0xff_value(const T& v) {
    constexpr size_t N = sizeof(T);
    alignas(T) unsigned char ffs[N];
    std::memset(ffs, 0xFF, N);
    return std::memcmp(&v, ffs, N) == 0;
}

} // namespace fill_memory_detail

/**
 * Read a single T from a pointer whose allocator depends on use_gpu.
 *
 * Safe to call from host for both host and device allocations: on GPU
 * data it does a one-element cudaMemcpy, on host data it does a direct
 * load. Host-only — the device side can always dereference its own
 * pointers and needs no wrapper.
 */
template <typename T>
HOST inline T read_one_host_safe(const T* p, const bool use_gpu) {
#ifdef HAS_CUDA
    if (use_gpu) {
        T out;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&out, p, sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
#else
    (void)use_gpu;
#endif
    return *p;
}

template <typename T>
HOST void allocate_memory(T** data_ptr, const size_t size, const bool use_gpu) {
    if (size == 0) {
        return;
    }

    if (*data_ptr) {
        #ifdef HAS_CUDA
        if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaFree(*data_ptr));
        } else
        #endif
        {
            free(*data_ptr);
        }
        *data_ptr = nullptr;
    }

    #ifdef HAS_CUDA
    if (use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMalloc(data_ptr, size * sizeof(T)));
    }
    else
    #endif
    {
        *data_ptr = static_cast<T *>(malloc(size * sizeof(T)));
    }

    if (!*data_ptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
    }
}

template <typename T>
HOST void resize_memory(T** data_ptr, const size_t size, size_t new_size, bool use_gpu) {
    if (!*data_ptr) {
        allocate_memory(data_ptr, new_size, use_gpu);
        return;
    }

    T* new_ptr = nullptr;

    #ifdef HAS_CUDA
    if (use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&new_ptr, new_size * sizeof(T)));
        if (new_ptr) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(new_ptr, *data_ptr, std::min(size, new_size) * sizeof(T), cudaMemcpyDeviceToDevice));
            CUDA_CHECK_AND_CLEAR(cudaFree(*data_ptr));
        }
    }
    else
    #endif
    {
        new_ptr = static_cast<T *>(realloc(*data_ptr, new_size * sizeof(T)));
    }

    if (!new_ptr) {
        std::cerr << "Memory reallocation failed!" << std::endl;
    } else {
        *data_ptr = new_ptr;
    }
}

template <typename T>
HOST void fill_memory(T* memory, size_t size, T value, bool use_gpu) {
    if (!memory || size == 0) return;

    #ifdef HAS_CUDA
    if (use_gpu) {
        // Fast path: byte-repeatable patterns go through cudaMemsetAsync,
        // avoiding both the hand-rolled fill kernel and the scalar-in-
        // device-memory alloc/copy/free that used to surround it.
        if (fill_memory_detail::is_zero_value(value)) {
            CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
                memory, 0, size * sizeof(T)));
            return;
        }
        if (fill_memory_detail::is_all_0xff_value(value)) {
            CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
                memory, 0xFF, size * sizeof(T)));
            return;
        }
        // General path: thrust dispatches its own kernel; no scratch alloc.
        thrust::fill_n(
            thrust::device,
            thrust::device_pointer_cast(memory),
            size,
            value);
        return;
    }
    #endif
    {
        std::fill(memory, memory + size, value);
    }
}

template <typename T>
HOST void append_memory(T** data_ptr, size_t& size, const T* new_data, const size_t new_size, const bool use_gpu) {
    if (!new_data || new_size == 0) return;  // No data to append

    const size_t total_size = size + new_size;
    T* new_ptr = nullptr;

    #ifdef HAS_CUDA
    if (use_gpu) {
        // Allocate new GPU memory
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&new_ptr, total_size * sizeof(T)));
        if (size > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(new_ptr, *data_ptr, size * sizeof(T), cudaMemcpyDeviceToDevice)); // Copy old data
        }
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(new_ptr + size, new_data, new_size * sizeof(T), cudaMemcpyDeviceToDevice)); // Append new data
        CUDA_CHECK_AND_CLEAR(cudaFree(*data_ptr)); // Free old memory
    }
    else
    #endif
    {
        // CPU allocation
        new_ptr = static_cast<T *>(realloc(*data_ptr, total_size * sizeof(T)));
        if (new_ptr) {
            std::memcpy(new_ptr + size, new_data, new_size * sizeof(T)); // Append new data
        }
    }

    if (new_ptr) {
        *data_ptr = new_ptr;
        size = total_size; // Update the size
    } else {
        std::cerr << "Memory append failed!" << std::endl;
    }
}

template <typename T>
HOST void clear_memory(T** data_ptr, const bool use_gpu) {
    if (!data_ptr) {
        return;
    }

    if (*data_ptr) {
        #ifdef HAS_CUDA
        if (use_gpu) {
            cudaPointerAttributes attributes;
            cudaError_t check_error = cudaPointerGetAttributes(&attributes, *data_ptr);

            if (check_error == cudaSuccess &&
                (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged)) {
                    CUDA_CHECK_AND_CLEAR(cudaFree(*data_ptr));
                } else {
                    clearCudaErrorState();
                }
        }
        else
        #endif
        {
            free(*data_ptr);
        }

        *data_ptr = nullptr;
    }
}

template <typename T>
HOST void copy_memory(T* dst, const T* src, const size_t size, const bool dst_gpu, const bool src_gpu) {
    if (!dst || !src || size == 0) {
        return;  // Nothing to copy
    }

    #ifdef HAS_CUDA
    if (dst_gpu && src_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
    } else if (dst_gpu && !src_gpu) {
        // Host to device
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
    } else if (!dst_gpu && src_gpu) {
        // Device to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    else
    #endif
    {
        // Host to host
        std::memcpy(dst, src, size * sizeof(T));
    }
}

template <typename T>
HOST void remove_first_n_memory(T** data_ptr, size_t& size, size_t n, const bool use_gpu) {
    if (!data_ptr || !*data_ptr || n == 0 || n >= size) {
        return;
    }

    const size_t new_size = size - n;
    T* new_ptr = nullptr;

    #ifdef HAS_CUDA
    if (use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&new_ptr, new_size * sizeof(T)));
        if (new_ptr) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(new_ptr, *data_ptr + n, new_size * sizeof(T), cudaMemcpyDeviceToDevice));
            CUDA_CHECK_AND_CLEAR(cudaFree(*data_ptr));
        }
    }
    else
    #endif
    {
        new_ptr = static_cast<T *>(malloc(new_size * sizeof(T)));
        if (new_ptr) {
            std::memcpy(new_ptr, *data_ptr + n, new_size * sizeof(T));
            free(*data_ptr);
        }
    }

    if (new_ptr) {
        *data_ptr = new_ptr;
        size = new_size;  // Update the size
    } else {
        std::cerr << "Memory reallocation failed in remove_first_n_memory!" << std::endl;
    }
}


#endif // MEMORY_H
