#ifndef CUDA_COMMON_SETUP_H
#define CUDA_COMMON_SETUP_H

#ifdef HAS_CUDA

#include <curand_kernel.h>
#include "../cuda_common/macros.cuh"

__global__ void setup_curand_states(curandState* rand_states, unsigned long seed);

HOST curandState* get_cuda_rand_states(size_t grid_dim, size_t block_dim);

std::pair<size_t, size_t> get_optimal_launch_params(int data_size, const cudaDeviceProp* device_prop);

#endif

#endif //CUDA_COMMON_SETUP_H
