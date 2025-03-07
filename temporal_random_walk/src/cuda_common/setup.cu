#include "setup.cuh"

#ifdef HAS_CUDA

__global__ void setup_curand_states(curandState* rand_states, const unsigned long seed) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &rand_states[tid]);
}

// Static member initialization
curandState* CudaRandomStates::d_states = nullptr;
unsigned int CudaRandomStates::num_blocks = 0;
unsigned int CudaRandomStates::num_threads = 0;
unsigned int CudaRandomStates::total_threads = 0;
bool CudaRandomStates::initialized = false;

// CUDA Kernel to initialize curand states
__global__ void init_random_states_kernel(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
    if (idx < gridDim.x * blockDim.x) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Initialize device properties
void CudaRandomStates::init_device_properties() {
    if (initialized) return;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    num_threads = prop.maxThreadsPerBlock / 2;
    num_blocks = (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) / num_threads;
    total_threads = num_threads * num_blocks;

    initialized = true;
}

// Allocate memory and launch random state initialization
void CudaRandomStates::allocate_states() {
    if (d_states) return;

    cudaMalloc(&d_states, total_threads * sizeof(curandState));

    init_random_states_kernel<<<num_blocks, num_threads>>>(
        d_states, static_cast<unsigned long>(time(nullptr))
    );
}

// Public method to initialize states
void CudaRandomStates::initialize() {
    init_device_properties();
    allocate_states();
}

// Cleanup method to free memory
void CudaRandomStates::cleanup() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}

// Accessors
curandState* CudaRandomStates::get_states() { return d_states; }
unsigned int CudaRandomStates::get_thread_count() { return total_threads; }
dim3 CudaRandomStates::get_grid_dim() { return {num_blocks, 1, 1}; }
dim3 CudaRandomStates::get_block_dim() { return {num_threads, 1, 1}; }

#endif
