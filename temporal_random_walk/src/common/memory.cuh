#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>
#include <iostream>

#include "macros.cuh"

template <typename T>
__global__ void fill_kernel(T* memory, const size_t size, T* value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        memory[idx] = *value;
    }
}

template <typename T>
HOST void allocate_memory(T** data_ptr, const size_t size, const bool use_gpu) {
    if (*data_ptr) {
        if (use_gpu) {
            cudaFree(*data_ptr);
        } else {
            free(*data_ptr);
        }
        *data_ptr = nullptr;
    }

    if (use_gpu) {
        cudaMalloc(data_ptr, size * sizeof(T));
    } else {
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
    if (use_gpu) {
        cudaMalloc(&new_ptr, new_size * sizeof(T));
        if (new_ptr) {
            cudaMemcpy(new_ptr, *data_ptr, std::min(size, new_size) * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaFree(*data_ptr);
        }
    } else {
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
    if (!memory) {
        std::cerr << "Error: memory is NULL!" << std::endl;
        return;
    }

    if (use_gpu) {
        T* d_value;
        cudaMalloc(&d_value, sizeof(T));
        cudaMemcpy(d_value, &value, sizeof(T), cudaMemcpyHostToDevice);

        constexpr int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        fill_kernel<<<blocksPerGrid, threadsPerBlock>>>(memory, size, d_value);

        cudaFree(d_value);
    } else {
        std::fill(memory, memory + size, value);
    }
}

template <typename T>
HOST void clear_memory(T** data_ptr, const bool use_gpu) {
    if (data_ptr && *data_ptr) {
        if (use_gpu) {
            cudaFree(*data_ptr);
        } else {
            free(*data_ptr);
        }
        *data_ptr = nullptr;
    }
}

#endif // MEMORY_H
