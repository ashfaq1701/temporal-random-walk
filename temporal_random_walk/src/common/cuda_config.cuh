#ifndef CUDA_CONST_H
#define CUDA_CONST_H

#ifdef HAS_CUDA

#include <thrust/execution_policy.h>

constexpr auto DEVICE_EXECUTION_POLICY = thrust::device;
constexpr size_t BLOCK_DIM = 512;

#endif

#endif // CUDA_CONST_H
