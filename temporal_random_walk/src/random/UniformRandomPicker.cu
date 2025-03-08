#include "UniformRandomPicker.cuh"

#include <stdexcept>
#include "../utils/rand_utils.cuh"

template<GPUUsageMode GPUUsage>
HOST int UniformRandomPicker<GPUUsage>::pick_random_host(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    return generate_random_int_host(start, end - 1);
}

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
DEVICE int UniformRandomPicker<GPUUsage>::pick_random_device(const int start, const int end, const bool prioritize_end, curandState* rand_state) {
    if (start >= end) {
        return -1;
    }

    return generate_random_int_device(start, end - 1, rand_state);
}
#endif

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
RandomPicker<GPUUsage>* UniformRandomPicker<GPUUsage>::to_device_ptr() {
    // Allocate device memory for the picker
    UniformRandomPicker<GPUUsage>* device_picker;
    cudaMalloc(&device_picker, sizeof(UniformRandomPicker<GPUUsage>));

    // Copy the object to device
    cudaMemcpy(device_picker, this, sizeof(UniformRandomPicker<GPUUsage>), cudaMemcpyHostToDevice);

    return device_picker;
}
#endif

template class UniformRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class UniformRandomPicker<GPUUsageMode::ON_GPU>;
#endif
