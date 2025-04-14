#include "setup.cuh"

#include <random>

#include "cuda_config.cuh"
#include "error_handlers.cuh"

unsigned long get_random_seed() {
    std::random_device rd;
    const unsigned long seed = rd() ^ (time(nullptr) << 1);
    return seed;
}

#ifdef HAS_CUDA

__global__ void setup_curand_states(curandState* rand_states, const unsigned long seed) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &rand_states[tid]);
}

HOST curandState* get_cuda_rand_states(size_t grid_dim, size_t block_dim) {
    const size_t total_threads = grid_dim * block_dim;

    curandState* rand_states;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&rand_states, total_threads * sizeof(curandState)));

    setup_curand_states<<<grid_dim, block_dim>>>(rand_states, get_random_seed());
    CUDA_KERNEL_CHECK("After curand states initialization");

    return rand_states;
}

HOST std::pair<size_t, size_t> get_optimal_launch_params(const size_t data_size, const cudaDeviceProp* device_prop) {
    size_t block_dim = BLOCK_DIM;
    size_t grid_dim = (data_size + block_dim - 1) / block_dim;
    const size_t min_grid_size = 2 * device_prop->multiProcessorCount;
    grid_dim = std::max(grid_dim, min_grid_size);
    return {grid_dim, block_dim};
}

#endif
