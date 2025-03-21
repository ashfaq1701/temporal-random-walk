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
        cudaMemcpy(parts.data, host_parts, i * sizeof(int), cudaMemcpyHostToDevice);
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

/**
* Prime number computation
*/

inline bool is_prime(const int n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    // Check divisibility up to sqrt(n) using 6k ± 1 optimization
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// Finds the next prime number greater than or equal to n
inline int next_prime(int n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;  // Start with an odd number

    while (!is_prime(n)) {
        n += 2;  // Skip even numbers
    }
    return n;
}

#endif // UTILS_H
