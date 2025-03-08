#ifndef RAND_UTILS_H
#define RAND_UTILS_H

#include <random>
#include "../cuda_common/types.cuh"
#include "../data/enums.h"
#ifdef HAS_CUDA
#include <curand_kernel.h>
#include "../cuda_common/setup.cuh"
#endif

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

template <typename T>
T generate_random_value_host(T start, T end) {
    std::uniform_real_distribution<T> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_int_host(const int start, const int end) {
    std::uniform_int_distribution<> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_number_bounded_by_host(const int max_bound) {
    return generate_random_int_host(0, max_bound - 1);
}

inline bool generate_random_boolean_host() {
    return generate_random_int_host(0, 1) == 1;
}

inline int pick_random_number_host(const int a, const int b) {
    return generate_random_boolean_host() ? a : b;
}

template <typename T, GPUUsageMode GPUUsage>
void shuffle_vector_host(typename SelectVectorType<T, GPUUsage>::type& vec) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec.begin(), vec.end(), rng);
}

#ifdef HAS_CUDA

template <typename T>
DEVICE T generate_random_value_device(T start, T end, curandState* state) {
    return start + (end - start) * curand_uniform(state);
}

DEVICE inline int generate_random_int_device(int start, int end, curandState* state) {
    return start + (curand(state) % (end - start + 1));
}

DEVICE inline int generate_random_number_bounded_by_device(int max_bound, curandState* state) {
    return generate_random_int_device(0, max_bound - 1, state);
}

DEVICE inline bool generate_random_boolean_device(curandState* state) {
    return curand_uniform(state) > 0.5f;
}

template <typename T>
__global__ void shuffle_kernel(T* vec, const int size, curandState* states) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    // Fisher-Yates shuffle (Parallelized)
    for (int i = size - 1; i > 0; --i) {
        int j = curand(&states[idx]) % (i + 1);
        if (idx == i || idx == j) {
            T temp = vec[i];
            vec[i] = vec[j];
            vec[j] = temp;
        }
    }
}

template <typename T>
void shuffle_vector(SelectVectorType<T, ON_GPU>& vec) {
    int size = vec.size();
    shuffle_kernel<<<CudaRandomStates::get_grid_dim(), CudaRandomStates::get_block_dim()>>>(
        thrust::raw_pointer_cast(vec.data()), size, CudaRandomStates::get_states());
}

#endif

#endif // RAND_UTILS_H
