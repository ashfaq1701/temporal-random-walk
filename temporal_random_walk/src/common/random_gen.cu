#include "random_gen.cuh"

#include <omp.h>
#include <iostream>
#include <random>

#ifdef HAS_CUDA
#include <curand.h>
#include <cuda_runtime.h>
#endif

#include "error_handlers.cuh"
#include "memory.cuh"

double* generate_n_random_numbers_cpu(const size_t n, const unsigned long long seed) {

    double* random_numbers = nullptr;
    allocate_memory(&random_numbers, n, false);

    // Parallelize with OpenMP
    #pragma omp parallel
    {
        // Each thread needs its own random generator to avoid contention
        const int thread_id = omp_get_thread_num();
        const unsigned long long thread_seed = seed + thread_id; // Different seed per thread

        std::mt19937 gen(thread_seed);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        // Divide work among threads
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            random_numbers[i] = dis(gen);
        }
    }

    return random_numbers;
}

#ifdef HAS_CUDA
double* generate_n_random_numbers_gpu(const size_t n) {
    double* d_random_numbers = nullptr;
    allocate_memory(&d_random_numbers, n, true);

    // Create and configure the cuRAND generator
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // Generate uniform random numbers in [0, 1)
    CHECK_CURAND(curandGenerateUniformDouble(gen, d_random_numbers, n));

    // Clean up the generator
    CHECK_CURAND(curandDestroyGenerator(gen));

    return d_random_numbers;
}
#endif

double* generate_n_random_numbers(const size_t n, const bool use_gpu) {
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
