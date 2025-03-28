#ifndef UTILS_H
#define UTILS_H

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#endif

#include <cstddef>

#include "../data/structs.cuh"
#include "../common/macros.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"

HOST inline DataBlock<int> repeat_elements(const DataBlock<int>& arr, int times, const bool use_gpu) {
    const size_t input_size = arr.size;
    const size_t output_size = input_size * times;

    // Allocate memory for the output
    DataBlock<int> repeated_items(output_size, use_gpu);

    #ifdef HAS_CUDA
    if (use_gpu) {
        // Create device pointers
        const int* arr_ptr = arr.data;

        // Use thrust to perform the transformation
        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::counting_iterator(0),
            thrust::counting_iterator<int>(output_size),
            thrust::device_pointer_cast(repeated_items.data),
            [arr_ptr, times] DEVICE (const int idx) {
                const int original_idx = idx / times;
                return arr_ptr[original_idx];
            }
        );

        CUDA_KERNEL_CHECK("After thrust transform in repeat_elements");
    }
    else
    #endif
    {
        // CPU implementation
        for (size_t i = 0; i < input_size; ++i) {
            for (int j = 0; j < times; ++j) {
                repeated_items.data[i * times + j] = arr.data[i];
            }
        }
    }

    return repeated_items;
}

template <typename T>
HOST DividedVector<T> divide_vector(const T* input, size_t input_size, int n, bool use_gpu) {
    return DividedVector<T>(input, input_size, n, use_gpu);
}

HOST inline DataBlock<int> divide_number(const int n, const int i, const bool use_gpu) {
    DataBlock<int> parts(i, use_gpu);

    // Calculate the base division value
    const int base_value = n / i;
    const int remainder = n % i;

    #ifdef HAS_CUDA
    if (use_gpu) {
        // Create a temporary host array
        int* host_parts = new int[i];

        // Fill with base values
        std::fill_n(host_parts, i, base_value);

        // Add remainder to first elements
        for (int j = 0; j < remainder; ++j) {
            host_parts[j]++;
        }

        // Copy to device
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(parts.data, host_parts, i * sizeof(int), cudaMemcpyHostToDevice));
        delete[] host_parts;
    }
    else
    #endif
    {
        // Fill with base values
        std::fill_n(parts.data, i, base_value);

        // Add remainder to first elements
        for (int j = 0; j < remainder; ++j) {
            parts.data[j]++;
        }
    }

    return parts;
}

HOST DEVICE inline int pick_other_number(const int first, const int second, const int picked_number) {
    return (picked_number == first) ? second : first;
}

HOST DEVICE inline size_t next_power_of_two(const size_t n) {
    // If n is already a power of 2, return it
    if ((n & (n - 1)) == 0) {
        return n;
    }

    // Otherwise, find the next power of 2
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }

    return power;
}

#endif // UTILS_H
