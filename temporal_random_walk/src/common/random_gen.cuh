#ifndef RANDOM_GEN_H
#define RANDOM_GEN_H

#ifdef HAS_CUDA
#include <curand.h>
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <omp.h>
#include <random>

#include "error_handlers.cuh"
#include "memory.cuh"

inline double* generate_n_random_numbers_cpu(const size_t n) {
    double* random_numbers = nullptr;
    allocate_memory(&random_numbers, n, false);

    std::random_device rd;  // uses hardware entropy

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        #pragma omp for
        for (size_t i = 0; i < n; ++i) {
            random_numbers[i] = dis(gen);
        }
    }

    return random_numbers;
}

#ifdef HAS_CUDA
inline double* generate_n_random_numbers_gpu(const size_t n) {
    double* d_random_numbers = nullptr;
    allocate_memory(&d_random_numbers, n, true);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    // Generate a random seed internally
    const unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    CHECK_CURAND(curandGenerateUniformDouble(gen, d_random_numbers, n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    return d_random_numbers;
}
#endif

inline double* generate_n_random_numbers(const size_t n, const bool use_gpu) {
    #ifdef HAS_CUDA
    if (use_gpu) {
        return generate_n_random_numbers_gpu(n);
    }
    else
    #endif
    {
        return generate_n_random_numbers_cpu(n);
    }
}

#endif // RANDOM_GEN_H
