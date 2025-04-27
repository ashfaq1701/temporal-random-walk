#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <random>
#include <ctime>
#include <algorithm>
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#endif

#include "../common/macros.cuh"
#include "../common/cuda_config.cuh"

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

HOST DEVICE inline int generate_random_number_bounded_by(const int max_bound, const double rand_number) {
    return static_cast<int>(rand_number * max_bound);
}

HOST DEVICE inline bool generate_random_boolean(const int rand_number) {
    return rand_number >= 0.5;
}

HOST DEVICE inline int pick_random_number(const int a, const int b, const int rand_number) {
    return generate_random_boolean(rand_number) ? a : b;
}

template <typename T>
HOST void shuffle_vector_host(T* vec, size_t size) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec, vec + size, rng);
}

#ifdef HAS_CUDA

template <typename T>
HOST void shuffle_vector_device(T* data, size_t size) {
    thrust::device_ptr<T> d_data(data);
    thrust::default_random_engine random_engine(static_cast<unsigned int>(std::time(nullptr)));
    thrust::shuffle(DEVICE_EXECUTION_POLICY, d_data, d_data + size, random_engine);
    CUDA_KERNEL_CHECK("After thrust shuffle in shuffle_vector_device");
}

#endif

#endif // UTILS_RANDOM_H
