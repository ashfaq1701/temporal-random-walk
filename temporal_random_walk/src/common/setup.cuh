#ifndef CUDA_SETUP_H
#define CUDA_SETUP_H

#include <cstddef>

#ifdef HAS_CUDA
#include <curand_kernel.h>
#include "cuda_config.cuh"
#endif

#include "macros.cuh"

#ifdef HAS_CUDA

HOST std::pair<size_t, size_t> get_optimal_launch_params(size_t data_size, const cudaDeviceProp* device_prop, size_t block_dim=BLOCK_DIM);

#endif

#endif //CUDA_SETUP_H
