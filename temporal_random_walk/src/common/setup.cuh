#ifndef CUDA_SETUP_H
#define CUDA_SETUP_H

#include <cstddef>
#include <curand_kernel.h>
#include "macros.cuh"

__global__ void setup_curand_states(curandState* rand_states, unsigned long seed);

unsigned long get_random_seed();

HOST curandState* get_cuda_rand_states(size_t grid_dim, size_t block_dim);

std::pair<size_t, size_t> get_optimal_launch_params(size_t data_size, const cudaDeviceProp* device_prop);

#endif //CUDA_SETUP_H
