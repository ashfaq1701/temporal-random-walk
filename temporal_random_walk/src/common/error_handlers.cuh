#ifndef CUDA_ERROR_HANDLERS
#define CUDA_ERROR_HANDLERS

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

/**
 * Checks a synchronous CUDA runtime call. Throws std::runtime_error on failure.
 * Also clears any sticky prior error before the call so a failure here cannot
 * be confused with a failure from earlier code.
 */
#define CUDA_CHECK_AND_CLEAR(call) do { \
    cudaGetLastError(); /* clear any prior error */ \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

/**
 * Asynchronous kernel-launch check.
 *
 * Only calls cudaGetLastError(), which catches launch-configuration errors
 * (bad grid/block, too much shared memory, invalid args, etc.) without
 * stalling the device. This is the production check: it lets kernels queue up
 * on the stream without a synchronization barrier between them.
 *
 * Asynchronous errors (out-of-bounds accesses, kernel aborts) will not be
 * caught here; they surface at the next synchronizing call
 * (cudaMemcpy, cudaStreamSynchronize, etc.) or via CUDA_KERNEL_CHECK_SYNC.
 */
#define CUDA_KERNEL_CHECK_ASYNC(msg) do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA launch error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg); \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

/**
 * Synchronous kernel check. Checks the launch error AND synchronizes the
 * device to surface any asynchronous kernel errors.
 *
 * Use this sparingly and deliberately — never in the hot path of a streaming
 * workload. Appropriate places: right before copying results to host, at the
 * end of a self-contained benchmark region, or in tests.
 */
#define CUDA_KERNEL_CHECK_SYNC(msg) do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA launch error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg); \
        throw std::runtime_error(_oss.str()); \
    } \
    _err = cudaDeviceSynchronize(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA sync error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg) << " (during synchronization)"; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

/**
 * Backward-compatible alias kept so the 84 existing call sites do not need to
 * change. In debug builds this preserves the old synchronous behavior so that
 * async errors surface at the offending kernel. In release builds it becomes
 * the async-only check, which is what production should have been doing.
 *
 * New code should prefer CUDA_KERNEL_CHECK_ASYNC explicitly and only reach
 * for CUDA_KERNEL_CHECK_SYNC when a sync point is genuinely intended.
 */
#ifdef NDEBUG
    #define CUDA_KERNEL_CHECK(msg) CUDA_KERNEL_CHECK_ASYNC(msg)
#else
    #define CUDA_KERNEL_CHECK(msg) CUDA_KERNEL_CHECK_SYNC(msg)
#endif

/**
 * cuRAND error check. Throws on failure.
 */
#define CHECK_CURAND(call) do { \
    curandStatus_t _status = (call); \
    if (_status != CURAND_STATUS_SUCCESS) { \
        std::ostringstream _oss; \
        _oss << "cuRAND error in " << __FILE__ << ":" << __LINE__ \
             << " status=" << static_cast<int>(_status) << " " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

/**
 * CUB error check. Throws on failure.
 */
#define CUB_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUB error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

/**
 * Explicitly clear the sticky CUDA error state. Use when you want to discard
 * a known-prior error without reacting to it (e.g., the intentional
 * cudaPointerGetAttributes probe in clear_memory).
 */
inline void clearCudaErrorState() {
    cudaGetLastError();
}

#endif // HAS_CUDA

#endif // CUDA_ERROR_HANDLERS
