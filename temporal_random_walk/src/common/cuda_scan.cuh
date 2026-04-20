#ifndef CUDA_SCAN_CUH
#define CUDA_SCAN_CUH

#ifdef HAS_CUDA

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "error_handlers.cuh"
#include "../data/buffer.cuh"

/**
 * CUB-backed inclusive-sum scan. Drop-in replacement for
 * thrust::inclusive_scan when input and output iterators reference the same
 * primitive type (size_t, double, etc.). CUB specializes its scan kernels
 * per-architecture and on primitives is typically 20-30% faster than
 * thrust's generic implementation.
 *
 * Two-call convention: first call queries required temp storage (fast, no
 * device work), second call does the scan. Scratch is held in a
 * scope-local Buffer<uint8_t> so no manual cudaFree is needed.
 *
 * Stream-aware: all CUB + scratch ops run on the given stream. Defaults to
 * 0 (legacy default stream) so existing graph-layer callers that are not
 * yet stream-threaded keep working.
 */
template <typename InputIteratorT, typename OutputIteratorT>
inline void cub_inclusive_sum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_bytes, d_in, d_out, num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceScan::InclusiveSum(
        temp.data(), temp_bytes, d_in, d_out, num_items, stream));
}

#endif  // HAS_CUDA

#endif  // CUDA_SCAN_CUH
