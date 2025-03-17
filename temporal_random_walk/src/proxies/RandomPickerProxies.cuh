#ifndef RANDOM_PICKER_PROXIES_H
#define RANDOM_PICKER_PROXIES_H

#include <vector>
#include "../src/common/setup.cuh"

#ifdef HAS_CUDA

__global__ void pick_exponential_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    curandState* rand_states);

#endif

class ExponentialIndexRandomPickerProxy {
    bool use_gpu;

public:

    explicit ExponentialIndexRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;
};

#ifdef HAS_CUDA

__global__ void pick_linear_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    curandState* rand_states);

#endif

class LinearRandomPickerProxy {

    bool use_gpu;

public:

    explicit LinearRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;
};

#ifdef HAS_CUDA

__global__ void pick_uniform_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    curandState* rand_states);

#endif

class UniformRandomPickerProxy {
    bool use_gpu;

public:
    explicit UniformRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool /* prioritize_end */) const;
};

#ifdef HAS_CUDA

__global__ void pick_weighted_random_number_cuda_kernel(
    int* result,
    double* weights,
    size_t weights_size,
    size_t group_start,
    size_t group_end,
    curandState* rand_states);

#endif

class WeightBasedRandomPickerProxy {
    bool use_gpu;

public:
    explicit WeightBasedRandomPickerProxy(bool use_gpu);

    int pick_random(const double* weights, size_t weights_size, size_t group_start, size_t group_end) const;

    int pick_random(const std::vector<double>& cumulative_weights, int group_start, int group_end) const;
};

#endif // RANDOM_PICKER_PROXIES_H
