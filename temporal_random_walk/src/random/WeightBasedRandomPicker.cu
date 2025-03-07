#include "WeightBasedRandomPicker.cuh"

#include "../utils/rand_utils.cuh"
#ifdef HAS_CUDA
#include "cuda/std/__algorithm_"
#endif

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
__global__ void pick_random_kernel(
    WeightBasedRandomPicker<GPUUsage>* random_picker,
    const double* cumulative_weights_ptr,
    const size_t weights_size,
    int group_start,
    int group_end,
    int* picked_value,
    curandState* rand_states) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = rand_states[tid];
    *picked_value = random_picker->pick_random_device(
        cumulative_weights_ptr,
        weights_size,
        group_start,
        group_end,
        &localState);
    rand_states[tid] = localState;
}
#endif

template<GPUUsageMode GPUUsage>
int WeightBasedRandomPicker<GPUUsage>::pick_random_host(
    const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
    const int group_start,
    const int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size())) {
        return -1;
    }

    // Get start and end sums
    double start_sum = 0.0;
    if (group_start > 0 && cumulative_weights[group_start] >= cumulative_weights[group_start - 1]) {
        start_sum = cumulative_weights[group_start - 1];
    }
    const double end_sum = cumulative_weights[group_end - 1];

    if (end_sum < start_sum) {
        return -1;
    }

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value_host(start_sum, end_sum);
    return static_cast<int>(std::lower_bound(
            cumulative_weights.begin() + group_start,
            cumulative_weights.begin() + group_end,
            random_val) - cumulative_weights.begin());
}

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
DEVICE int WeightBasedRandomPicker<GPUUsage>::pick_random_device(
    const double* cumulative_weights_ptr,
    const size_t weights_size,
    const int group_start,
    const int group_end,
    curandState* rand_state)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(weights_size)) {
        return -1;
        }

    // Get start and end sums
    double start_sum = 0.0;
    if (group_start > 0 && cumulative_weights_ptr[group_start] >= cumulative_weights_ptr[group_start - 1]) {
        start_sum = cumulative_weights_ptr[group_start - 1];
    }
    const double end_sum = cumulative_weights_ptr[group_end - 1];

    if (end_sum < start_sum) {
        return -1;
    }

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value_device(start_sum, end_sum, rand_state);
    return static_cast<int>(cuda::std::lower_bound(
            cumulative_weights_ptr + group_start,
            cumulative_weights_ptr + group_end,
            random_val) - cumulative_weights_ptr);
}
#endif

template<GPUUsageMode GPUUsage>
HOST int WeightBasedRandomPicker<GPUUsage>::pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        const int group_start,
        const int group_end)
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU)
    {
        int* d_picked_value;
        int h_picked_value = -1;
        curandState* d_rand_states;

        cudaMalloc(&d_picked_value, sizeof(int));
        cudaMalloc(&d_rand_states, sizeof(curandState));

        setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));
        pick_random_kernel<<<1, 1>>>(
            this,
            thrust::raw_pointer_cast(cumulative_weights.data()),
            cumulative_weights.size(),
            group_start,
            group_start,
            d_picked_value,
            d_rand_states);

        cudaMemcpy(&h_picked_value, d_picked_value, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_picked_value);
        cudaFree(d_rand_states);

        return h_picked_value;
    }
    else
    #endif
    {
        return this->pick_random_host(cumulative_weights, group_start, group_end);
    }
};

template class WeightBasedRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class WeightBasedRandomPicker<GPUUsageMode::ON_GPU>;
#endif
