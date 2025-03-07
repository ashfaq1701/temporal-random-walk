#ifndef CUDA_COMMON_SETUP_H
#define CUDA_COMMON_SETUP_H

#ifdef HAS_CUDA

#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void setup_curand_states(curandState* rand_states, const unsigned long seed);

class CudaRandomStates {
public:
    // Initialize random states (should be called at startup)
    static void initialize();

    // Access the device random states
    static curandState* get_states();

    // Get device properties
    static unsigned int get_thread_count();
    static dim3 get_grid_dim();
    static dim3 get_block_dim();

    // Cleanup memory (called at program exit)
    static void cleanup();

private:
    static void init_device_properties();
    static void allocate_states();

    static curandState* d_states;
    static unsigned int num_blocks;
    static unsigned int num_threads;
    static unsigned int total_threads;
    static bool initialized;
};

#endif

#endif //CUDA_COMMON_SETUP_H
