#ifndef CUDA_ERROR_HANDLERS
#define CUDA_ERROR_HANDLERS

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <string>

/**
 * Macro that checks for CUDA errors and clears the error state, then exits.
 * This prevents errors from "sticking" around and causing false positives later
 */
#define CUDA_CHECK_AND_CLEAR(call) do { \
    /* Clear any prior errors */ \
    cudaGetLastError(); \
    \
    /* Make the call */ \
    cudaError_t err = call; \
    \
    /* Check for errors from this call */ \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::cerr << "  Code: " << err << " (" << cudaGetErrorString(err) << ")\n"; \
        std::cerr << "  Call: " << #call << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_LOG_ERROR_AND_CONTINUE(call) do { \
    cudaError_t err = call; \
    \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::cerr << "  Code: " << static_cast<int>(err) << " (" << cudaGetErrorString(err) << ")\n"; \
        std::cerr << "  Call: " << #call << "\n"; \
        cudaGetLastError(); /* Clear sticky error */ \
    } \
} while(0)

/**
 * Macro to check for errors after asynchronous operations like kernel launches
 * Clears error state before and after checking
 */
#define CUDA_KERNEL_CHECK(msg) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::cerr << "  Code: " << err << " (" << cudaGetErrorString(err) << ")\n"; \
        std::cerr << "  Message: " << msg << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
    /* Synchronize to catch asynchronous errors */ \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA synchronization error in " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::cerr << "  Code: " << err << " (" << cudaGetErrorString(err) << ")\n"; \
        std::cerr << "  Message: " << msg << " (during synchronization)\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * Function to reset CUDA error state
 * Call this if you want to explicitly clear error state without checking
 */
inline void clearCudaErrorState() {
    cudaGetLastError(); // This clears the error state
}

#endif

#endif // CUDA_ERROR_HANDLERS
