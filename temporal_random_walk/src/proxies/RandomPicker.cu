#include "RandomPicker.cuh"

#include "../src/random/pickers.cuh"
#include "../src/common/setup.cuh"

#ifdef HAS_CUDA

__global__ void pick_exponential_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    const bool prioritize_end,
    curandState* rand_states) {
    // Each thread picks a random number
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only the first thread computes the result
    if (idx == 0) {
        *result = random_pickers::pick_random_exponential_index_device(start, end, prioritize_end, &rand_states[idx]);
    }
}

#endif

ExponentialIndexRandomPicker::ExponentialIndexRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int ExponentialIndexRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) const {
    #ifdef HAS_CUDA
    if (use_gpu) {
        // Initialize CUDA random states (1 thread is enough since we only need 1 random number)
        curandState* rand_states = get_cuda_rand_states(1, 1);

        // Allocate device memory for the result
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        // Launch kernel with a single thread
        pick_exponential_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_states);

        // Copy result back to host
        int h_result;
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_result);
        cudaFree(rand_states);

        return h_result;
    }
    else
    #endif
    {
        return random_pickers::pick_random_exponential_index_host(start, end, prioritize_end);
    }
}

#ifdef HAS_CUDA

__global__ void pick_linear_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    const bool prioritize_end,
    curandState* rand_states) {
    // Each thread picks a random number
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only the first thread computes the result
    if (idx == 0) {
        *result = random_pickers::pick_random_linear_device(start, end, prioritize_end, &rand_states[idx]);
    }
}

#endif

LinearRandomPicker::LinearRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int LinearRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) const {
    #ifdef HAS_CUDA
    if (use_gpu) {
        // Initialize CUDA random states (1 thread is enough since we only need 1 random number)
        curandState* rand_states = get_cuda_rand_states(1, 1);

        // Allocate device memory for the result
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        // Launch kernel with a single thread
        pick_linear_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_states);

        // Copy result back to host
        int h_result;
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_result);
        cudaFree(rand_states);

        return h_result;
    }
    else
    #endif
    {
        return random_pickers::pick_random_linear_host(start, end, prioritize_end);
    }
}

#ifdef HAS_CUDA
__global__ void pick_uniform_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    curandState* rand_states) {
    // Each thread picks a random number
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only the first thread computes the result
    if (idx == 0) {
        *result = random_pickers::pick_random_uniform_device(start, end, &rand_states[idx]);
    }
}
#endif

UniformRandomPicker::UniformRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int UniformRandomPicker::pick_random(const int start, const int end, const bool /* prioritize_end */) const {
    #ifdef HAS_CUDA
    if (use_gpu) {
        // Initialize CUDA random states (1 thread is enough since we only need 1 random number)
        curandState* rand_states = get_cuda_rand_states(1, 1);

        // Allocate device memory for the result
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        // Launch kernel with a single thread
        pick_uniform_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, rand_states);

        // Copy result back to host
        int h_result;
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_result);
        cudaFree(rand_states);

        return h_result;
    }
    else
    #endif
    {
        return random_pickers::pick_random_uniform_host(start, end);
    }
}

#ifdef HAS_CUDA
__global__ void pick_weighted_random_number_cuda_kernel(
    int* result,
    double* weights,
    const size_t weights_size,
    const size_t group_start,
    const size_t group_end,
    curandState* rand_states) {
    // Each thread picks a random number
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only the first thread computes the result
    if (idx == 0) {
        *result = random_pickers::pick_random_exponential_weights_device(weights, weights_size, group_start, group_end, &rand_states[idx]);
    }
}
#endif

WeightBasedRandomPicker::WeightBasedRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int WeightBasedRandomPicker::pick_random(const double* weights, const size_t weights_size, const size_t group_start, const size_t group_end) const {
    #ifdef HAS_CUDA
    if (use_gpu) {
        // Initialize CUDA random states
        curandState* rand_states = get_cuda_rand_states(1, 1);

        // Allocate device memory for weights
        double* d_weights;
        cudaMalloc(&d_weights, weights_size * sizeof(double));

        // Copy weights to device
        cudaMemcpy(d_weights, weights, weights_size * sizeof(double), cudaMemcpyHostToDevice);

        // Allocate device memory for the result
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        // Launch kernel with a single thread
        pick_weighted_random_number_cuda_kernel<<<1, 1>>>(d_result, d_weights, weights_size, group_start, group_end, rand_states);

        // Copy result back to host
        int h_result;
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_result);
        cudaFree(d_weights);
        cudaFree(rand_states);

        return h_result;
    }
    else
    #endif
    {
        return random_pickers::pick_random_exponential_weights_host(
            const_cast<double*>(weights), weights_size, group_start, group_end);
    }
}

int WeightBasedRandomPicker::pick_random(const std::vector<double>& cumulative_weights, const int group_start, const int group_end) const {
    return pick_random(cumulative_weights.data(), cumulative_weights.size(),
                     static_cast<size_t>(group_start), static_cast<size_t>(group_end));
}
