#ifndef RANDOM_PICKER_PROXIES_H
#define RANDOM_PICKER_PROXIES_H

#include "../src/common/setup.cuh"

__global__ inline void pick_exponential_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    curandState* rand_states);

class ExponentialIndexRandomPickerProxy {
    bool use_gpu;

public:

    explicit ExponentialIndexRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;
};

__global__ inline void pick_linear_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    curandState* rand_states);

class LinearRandomPickerProxy {

    bool use_gpu;

public:

    explicit LinearRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;
};

__global__ inline void pick_uniform_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    curandState* rand_states);

class UniformRandomPickerProxy {
    bool use_gpu;

public:
    explicit UniformRandomPickerProxy(bool use_gpu);

    int pick_random(int start, int end, bool /* prioritize_end */) const;
};

__global__ inline void pick_weighted_random_number_cuda_kernel(
    int* result,
    double* weights,
    size_t weights_size,
    size_t group_start,
    size_t group_end,
    curandState* rand_states);

class WeightBasedRandomPickerProxy {
    bool use_gpu;

public:
    explicit WeightBasedRandomPickerProxy(bool use_gpu);

    int pick_random(const double* weights, size_t weights_size, size_t group_start, size_t group_end) const;

    int pick_random(const std::vector<double>& cumulative_weights, int group_start, int group_end) const;
};

#endif // RANDOM_PICKER_PROXIES_H
