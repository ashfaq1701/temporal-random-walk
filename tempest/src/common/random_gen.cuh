#ifndef RANDOM_GEN_H
#define RANDOM_GEN_H

#ifdef HAS_CUDA
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#endif

#include <chrono>
#include <omp.h>
#include <random>

#include "error_handlers.cuh"
#include "const.cuh"
#include "../data/buffer.cuh"

inline uint64_t secure_random_seed() {
    std::random_device rd;
    uint64_t seed = 0;

    seed |= static_cast<uint64_t>(rd()) << 32;
    seed |= static_cast<uint64_t>(rd());

    return seed;
}

// Counter-based mixer (splitmix64). Draws are a pure function of (seed, index),
// so the produced stream is identical regardless of OpenMP thread count or
// scheduling — a prerequisite for reproducible walks, since each walk reads a
// fixed rand_nums slice indexed by walk_idx.
HOST inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Resolve an explicit seed if one was supplied, otherwise draw a fresh one.
inline uint64_t resolve_random_seed(const uint64_t base_seed) {
    return (base_seed != EMPTY_GLOBAL_SEED) ? base_seed : secure_random_seed();
}

inline Buffer<double> generate_n_random_numbers_cpu(const size_t n, const uint64_t base_seed) {
    Buffer<double> random_numbers(n, false);

    double* out = random_numbers.data();
    const uint64_t seed = resolve_random_seed(base_seed);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // Take the top 53 bits -> a uniform double in [0, 1).
        const uint64_t bits = splitmix64(seed ^ (static_cast<uint64_t>(i) + 0x9E3779B97F4A7C15ULL));
        out[i] = static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
    }

    return random_numbers;
}

#ifdef HAS_CUDA

// init once per thread; draw_u01_philox steps the counter per draw.
using PhiloxState = curandStatePhilox4_32_10_t;

DEVICE __forceinline__ void init_philox_state(
    PhiloxState& state,
    const uint64_t base_seed,
    const uint64_t walk_idx,
    const uint64_t offset = 0ULL) {
    // subsequence=walk_idx gives each walk its own stream; offset avoids
    // counter reuse across successive step-kernel launches.
    curand_init(base_seed, walk_idx, offset, &state);
}

DEVICE __forceinline__ double draw_u01_philox(PhiloxState& state) {
    return curand_uniform_double(&state);
}

inline Buffer<double> generate_n_random_numbers_gpu(const size_t n, const uint64_t base_seed) {
    Buffer<double> d_random_numbers(n, true);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    const auto seed = resolve_random_seed(base_seed);
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    CHECK_CURAND(curandGenerateUniformDouble(gen, d_random_numbers.data(), n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    return d_random_numbers;
}

#endif

inline Buffer<double> generate_n_random_numbers(
        const size_t n, const bool use_gpu, const uint64_t base_seed = EMPTY_GLOBAL_SEED) {
    #ifdef HAS_CUDA
    if (use_gpu) {
        return generate_n_random_numbers_gpu(n, base_seed);
    }
    else
    #endif
    {
        return generate_n_random_numbers_cpu(n, base_seed);
    }
}

#endif // RANDOM_GEN_H
