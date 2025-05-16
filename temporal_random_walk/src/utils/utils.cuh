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
