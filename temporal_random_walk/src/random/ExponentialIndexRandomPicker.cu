#include "ExponentialIndexRandomPicker.cuh"

#include <cmath>
#include <stdexcept>
#include <utils/rand_utils.cuh>

#include "../utils/utils.h"

// Derivation available in derivations folder
template<GPUUsageMode GPUUsage>
HOST int ExponentialIndexRandomPicker<GPUUsage>::pick_random_host(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    const double u = generate_random_value_host(0.0, 1.0);

    double k;
    if (len_seq < 710) {
        // Inverse CDF formula,
        // k = ln(1 + u * (e^len seq − 1)) − 1
        k = log1p(u * expm1(len_seq)) - 1;
    } else {
        // Inverse CDF approximation for large len_seq,
        // k = len_seq + ln(u) − 1
        k = len_seq + std::log(u) - 1;
    }

    // Due to rounding, the trailing "-1" in the inverse CDF formula causes error.
    // To compensate for this we add 1 with k.
    // And bound the results within limits.
    const int rounded_index = std::max(0, std::min(static_cast<int>(k + 1), len_seq - 1));

    if (prioritize_end) {
        return start + rounded_index;
    } else {
        return start + (len_seq - 1 - rounded_index);
    }
}

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
DEVICE int ExponentialIndexRandomPicker<GPUUsage>::pick_random_device(const int start, const int end, const bool prioritize_end, curandState* rand_state) {
    if (start >= end) {
        return -1;
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    const double u = generate_random_value_device(0.0, 1.0, rand_state);

    double k;
    if (len_seq < 710) {
        // Inverse CDF formula,
        // k = ln(1 + u * (e^len seq − 1)) − 1
        k = log1p(u * expm1(len_seq)) - 1;
    } else {
        // Inverse CDF approximation for large len_seq,
        // k = len_seq + ln(u) − 1
        k = len_seq + std::log(u) - 1;
    }

    // Due to rounding, the trailing "-1" in the inverse CDF formula causes error.
    // To compensate for this we add 1 with k.
    // And bound the results within limits.
    const int rounded_index = std::max(0, std::min(static_cast<int>(k + 1), len_seq - 1));

    if (prioritize_end) {
        return start + rounded_index;
    } else {
        return start + (len_seq - 1 - rounded_index);
    }
}
#endif

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
RandomPicker<GPUUsage>* ExponentialIndexRandomPicker<GPUUsage>::to_device_ptr() {
    // Allocate device memory for the picker
    ExponentialIndexRandomPicker<GPUUsage>* device_picker;
    cudaMalloc(&device_picker, sizeof(ExponentialIndexRandomPicker<GPUUsage>));

    // Copy the object to device
    cudaMemcpy(device_picker, this, sizeof(ExponentialIndexRandomPicker<GPUUsage>), cudaMemcpyHostToDevice);

    return device_picker;
}
#endif

template class ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class ExponentialIndexRandomPicker<GPUUsageMode::ON_GPU>;
#endif
