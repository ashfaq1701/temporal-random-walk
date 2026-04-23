#ifndef CUDA_SORT_CUH
#define CUDA_SORT_CUH

#ifdef HAS_CUDA

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include "error_handlers.cuh"
#include "../data/buffer.cuh"
#include "../data/device_arena.cuh"

/**
 * Legacy sort-by-keys that returns values re-ordered in place. No streams,
 * no pluggable scratch — retained for callers that haven't been moved to the
 * Buffer/stream-aware helpers yet. Prefer cub_sort_pairs below for new work.
 */
template <typename KeyType, typename ValueType>
void cub_radix_sort_values_by_keys(
    const KeyType* d_keys,
    ValueType* d_values,
    size_t num_items)
{
    KeyType* d_keys_out = nullptr;
    ValueType* d_values_out = nullptr;
    cudaMalloc(&d_keys_out, sizeof(KeyType) * num_items);
    cudaMalloc(&d_values_out, sizeof(ValueType) * num_items);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_keys_out, d_values, d_values_out, num_items);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_keys_out, d_values, d_values_out, num_items);

    cudaMemcpy(d_values, d_values_out, sizeof(ValueType) * num_items,
               cudaMemcpyDeviceToDevice);

    cudaFree(d_temp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
}

/**
 * CUB-backed sort-by-key producing separate out buffers for keys and values.
 * Stream-aware; scratch held in a scope-local Buffer<uint8_t>. Two-call
 * convention: query temp bytes, then sort. `d_keys_in` is treated as const
 * by CUB — the in-buffer is not mutated.
 *
 * Same signature pattern as cub_exclusive_sum etc. in cuda_scan.cuh.
 */
template <typename KeyType, typename ValueType>
inline void cub_sort_pairs(
    const KeyType* d_keys_in,
    KeyType* d_keys_out,
    const ValueType* d_values_in,
    ValueType* d_values_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.data(), temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));
}

// Arena-backed overload. Scratch is bump-allocated from a caller-owned
// DeviceArena, replacing the per-call cudaMalloc/cudaFree inside the
// Buffer<uint8_t>-backed variant above. Both alloc and free there fence
// the default stream, which is the dominant wall-time cost when the
// scheduler calls this every intermediate step. See cuda_scan.cuh for
// the parallel rationale.
template <typename KeyType, typename ValueType>
inline void cub_sort_pairs(
    const KeyType* d_keys_in,
    KeyType* d_keys_out,
    const ValueType* d_values_in,
    ValueType* d_values_out,
    const size_t num_items,
    DeviceArena& arena,
    const cudaStream_t stream) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));

    uint8_t* temp = arena.acquire<uint8_t>(temp_bytes);
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));
}

#endif  // HAS_CUDA

#endif  // CUDA_SORT_CUH
