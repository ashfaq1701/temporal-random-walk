#include "IndexBasedRandomPicker.cuh"
#include "../cuda_common/setup.cuh"

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
__global__ void pick_random_kernel(
    IndexBasedRandomPicker<GPUUsage>* random_picker,
    int start,
    int end,
    bool prioritize_end,
    int* picked_value, curandState* rand_states) {
    curandState localState = rand_states[threadIdx.x];
    *picked_value = random_picker->pick_random_device(start, end, prioritize_end, &localState);
    rand_states[threadIdx.x] = localState;  // Store back the updated state
}
#endif

template<GPUUsageMode GPUUsage>
int IndexBasedRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end)
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
        pick_random_kernel<<<1, 1>>>(this, start, end, prioritize_end, d_picked_value, d_rand_states);

        cudaMemcpy(&h_picked_value, d_picked_value, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_picked_value);
        cudaFree(d_rand_states);

        return h_picked_value;
    }
    else
    #endif
    {
        return this->pick_random_host(start, end, prioritize_end);
    }
}

template class IndexBasedRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class IndexBasedRandomPicker<GPUUsageMode::ON_GPU>;
#endif
