#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <map>
#include "../data/structs.cuh"
#include "../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
typename SelectVectorType<int, GPUUsage>::type repeat_elements(
    const typename SelectVectorType<int, GPUUsage>::type& arr,
    int times) {
    #ifdef HAS_CUDA
    if  (GPUUsage == GPUUsageMode::ON_GPU) {
        const size_t input_size = arr.size();
        const size_t output_size = input_size * times;

        const int* arr_ptr = thrust::raw_pointer_cast(arr.data());

        typename SelectVectorType<int, GPUUsage>::type repeated_items;
        repeated_items.resize(output_size);

        thrust::transform(
            thrust::counting_iterator(0),
            thrust::counting_iterator<int>(output_size),
            repeated_items.begin(),
            [arr_ptr, times] __device__ (const int idx) {
                const int original_idx = idx / times;
                return arr_ptr[original_idx];
            }
        );

        return repeated_items;
    }
    else
    #endif
    {
        typename SelectVectorType<int, GPUUsage>::type repeated_items;
        repeated_items.reserve(arr.size() * times);

        for (const auto& item : arr) {
            for (int i = 0; i < times; ++i) {
                repeated_items.push_back(item);
            }
        }

        return repeated_items;
    }
}

template <typename T, GPUUsageMode GPUUsage>
DividedVector<T, GPUUsage> divide_vector(
    const typename SelectVectorType<T, GPUUsage>::type& input,
    int n)
{
    return DividedVector<T, GPUUsage>(input, n);
}

template <GPUUsageMode GPUUsage>
typename SelectVectorType<int, GPUUsage>::type divide_number(int n, int i) {
    typename SelectVectorType<int, GPUUsage>::type parts(i);
    std::fill(parts.begin(), parts.end(), n / i);

    const int remainder = n % i;

    for (int j = 0; j < remainder; ++j) {
        ++parts[j];
    }

    return parts;
}

inline int pick_other_number(const std::tuple<int, int>& number, const int picked_number) {
    const int first = std::get<0>(number);
    const int second = std::get<1>(number);
    return (picked_number == first) ? second : first;
}

#endif //UTILS_H
