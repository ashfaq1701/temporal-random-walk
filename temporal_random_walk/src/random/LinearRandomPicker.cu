#include "LinearRandomPicker.cuh"

#include <cmath>
#include <stdexcept>
#include "../utils/rand_utils.cuh"
#include "../utils/utils.h"

// Derivation available in derivations folder
template<GPUUsageMode GPUUsage>
HOST int LinearRandomPicker<GPUUsage>::pick_random_host(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // For a sequence of length n, weights form an arithmetic sequence
    // When prioritizing end: weights are 1, 2, 3, ..., n
    // When prioritizing start: weights are n, n-1, n-2, ..., 1
    // Sum of arithmetic sequence = n(a1 + an)/2 = n(n+1)/2
    const double total_weight = static_cast<double>(len_seq) *
                                   (static_cast<double>(len_seq) + 1.0) / 2.0;

    // Generate random value in [0, total_weight)
    const double random_value = generate_random_value_host(0.0, total_weight);

    // For both cases, we solve quadratic equation i² + i - 2r = 0
    // where r is our random value (or transformed random value)
    // Using quadratic formula: (-1 ± √(1 + 8r))/2
    const double discriminant = 1.0 + 8.0 * random_value;
    const double root = (-1.0 + std::sqrt(discriminant)) / 2.0;
    const int index = static_cast<int>(std::floor(root));

    if (prioritize_end) {
        // For prioritize_end=true, larger indices should have higher probability
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritize_end=false, we reverse the index to give
        // higher probability to smaller indices
        const int revered_index = len_seq - 1 - index;
        return start + std::max(0, revered_index);
    }
}

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
DEVICE int LinearRandomPicker<GPUUsage>::pick_random_device(const int start, const int end, const bool prioritize_end, curandState* rand_state) {
    if (start >= end) {
        return -1;
    }

    const int len_seq = end - start;

    // For a sequence of length n, weights form an arithmetic sequence
    // When prioritizing end: weights are 1, 2, 3, ..., n
    // When prioritizing start: weights are n, n-1, n-2, ..., 1
    // Sum of arithmetic sequence = n(a1 + an)/2 = n(n+1)/2
    const double total_weight = static_cast<double>(len_seq) *
                                   (static_cast<double>(len_seq) + 1.0) / 2.0;

    // Generate random value in [0, total_weight)
    const double random_value = generate_random_value_device(0.0, total_weight, rand_state);

    // For both cases, we solve quadratic equation i² + i - 2r = 0
    // where r is our random value (or transformed random value)
    // Using quadratic formula: (-1 ± √(1 + 8r))/2
    const double discriminant = 1.0 + 8.0 * random_value;
    const double root = (-1.0 + std::sqrt(discriminant)) / 2.0;
    const int index = static_cast<int>(std::floor(root));

    if (prioritize_end) {
        // For prioritize_end=true, larger indices should have higher probability
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritize_end=false, we reverse the index to give
        // higher probability to smaller indices
        const int revered_index = len_seq - 1 - index;
        return start + std::max(0, revered_index);
    }
}
#endif

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
RandomPicker<GPUUsage>* LinearRandomPicker<GPUUsage>::to_device_ptr() {
    // Allocate device memory for the picker
    LinearRandomPicker<GPUUsage>* device_picker;
    cudaMalloc(&device_picker, sizeof(LinearRandomPicker<GPUUsage>));

    // Copy the object to device
    cudaMemcpy(device_picker, this, sizeof(LinearRandomPicker<GPUUsage>), cudaMemcpyHostToDevice);

    return device_picker;
}
#endif

template class LinearRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class LinearRandomPicker<GPUUsageMode::ON_GPU>;
#endif
