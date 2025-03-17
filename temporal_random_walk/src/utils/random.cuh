#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <random>
#include <thrust/device_ptr.h>
#include <thrust/shuffle.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <ctime>

#include "../common/macros.cuh"
#include "../common/cuda_config.cuh"

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

template <typename T>
HOST T generate_random_value_host(T start, T end) {
    std::uniform_real_distribution<T> dist(start, end);
    return dist(thread_local_gen);
}

HOST inline int generate_random_int_host(const int start, const int end) {
    std::uniform_int_distribution<> dist(start, end);
    return dist(thread_local_gen);
}

HOST inline int generate_random_number_bounded_by_host(const int max_bound) {
    return generate_random_int_host(0, max_bound - 1);
}

HOST inline bool generate_random_boolean_host() {
    return generate_random_int_host(0, 1) == 1;
}

HOST inline int pick_random_number_host(const int a, const int b) {
    return generate_random_boolean_host() ? a : b;
}

template <typename T>
HOST void shuffle_vector_host(T* vec, size_t size) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec, vec + size, rng);
}

template <typename T>
DEVICE T generate_random_value_device(T start, T end, curandState* state) {
    return start + (end - start) * curand_uniform(state);
}

DEVICE inline int generate_random_int_device(const int start, const int end, curandState* state) {
    const int rand_val = start + static_cast<int>(curand_uniform(state) * (end - start + 1));
    return rand_val;
}

DEVICE inline int generate_random_number_bounded_by_device(const int max_bound, curandState* state) {
    return generate_random_int_device(0, max_bound - 1, state);
}

DEVICE inline bool generate_random_boolean_device(curandState* state) {
    return curand_uniform(state) > 0.5f;
}

DEVICE inline int pick_random_number_device(const int a, const int b, curandState* state) {
    return generate_random_boolean_device(state) ? a : b;
}

template <typename T>
HOST void shuffle_vector_device(T* data, size_t size) {
    thrust::device_ptr<T> d_data(data);
    thrust::default_random_engine random_engine(static_cast<unsigned int>(std::time(nullptr)));
    thrust::shuffle(DEVICE_EXECUTION_POLICY, d_data, d_data + size, random_engine);
}



#endif // UTILS_RANDOM_H
