#ifndef CUDA_SCAN_CUH
#define CUDA_SCAN_CUH

#ifdef HAS_CUDA

#include <cub/device/device_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_partition.cuh>
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

/**
 * CUB-backed exclusive-sum scan. Same two-call convention as the inclusive
 * variant. Used for boundary-flag -> scatter-index conversion where we need
 * exclusive prefixes (so flag_scan[i] is the output slot for the i'th entry).
 */
template <typename InputIteratorT, typename OutputIteratorT>
inline void cub_exclusive_sum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, d_in, d_out, num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceScan::ExclusiveSum(
        temp.data(), temp_bytes, d_in, d_out, num_items, stream));
}

/**
 * CUB-backed even-binned histogram. Each sample in [lower_level, upper_level)
 * is placed into one of num_buckets bins; bin[i] counts samples in
 * [lower_level + i * width, lower_level + (i+1) * width) where
 * width = (upper_level - lower_level) / num_buckets.
 *
 * For integer samples with upper_level - lower_level == num_buckets, each
 * sample lands exactly in the bucket equal to (sample - lower_level).
 *
 * Replaces thrust::for_each + atomicAdd patterns for per-bucket counting.
 * Per-block shared-memory local histograms avoid the hot-node atomic
 * contention that serialized the atomic version on skewed graphs.
 *
 * Counter buffer must be pre-zeroed by the caller (CUB overwrites, but
 * only the buckets it hits in range).
 */
template <typename SampleT, typename CounterT>
inline void cub_histogram_even(
    const SampleT* d_samples,
    CounterT* d_histogram,
    const int num_buckets,
    const SampleT lower_level,
    const SampleT upper_level,
    const size_t num_samples,
    const cudaStream_t stream = 0) {

    if (num_samples == 0 || num_buckets <= 0) return;

    const int num_output_levels = num_buckets + 1;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceHistogram::HistogramEven(
        nullptr, temp_bytes,
        d_samples, d_histogram,
        num_output_levels, lower_level, upper_level,
        num_samples, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceHistogram::HistogramEven(
        temp.data(), temp_bytes,
        d_samples, d_histogram,
        num_output_levels, lower_level, upper_level,
        num_samples, stream));
}

/**
 * CUB-backed run-length encode over a sorted key sequence. Writes unique
 * keys, run lengths, and *d_num_runs_out (single device counter). Same
 * two-call Buffer<uint8_t> scratch convention as the other cub_* helpers.
 */
template <typename KeyType, typename LengthType, typename NumRunsType>
inline void cub_run_length_encode(
    const KeyType* d_keys_in,
    KeyType* d_unique_out,
    LengthType* d_counts_out,
    NumRunsType* d_num_runs_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_bytes,
        d_keys_in, d_unique_out, d_counts_out, d_num_runs_out,
        num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceRunLengthEncode::Encode(
        temp.data(), temp_bytes,
        d_keys_in, d_unique_out, d_counts_out, d_num_runs_out,
        num_items, stream));
}

/**
 * CUB-backed flagged stream compaction. For each i in [0, num_items), copies
 * d_in[i] to the output iff d_flags[i] is truthy. *d_num_selected_out
 * receives the number of selected elements. Used to drop terminated walks
 * before sort-and-group in the node-grouped intermediate-step path: we flag
 * alive walks and compact their original walk indices so downstream kernels
 * never touch the dead ones.
 */
template <typename InputT, typename FlagT, typename NumSelectedT>
inline void cub_partition_flagged(
    const InputT* d_in,
    const FlagT* d_flags,
    InputT* d_out,
    NumSelectedT* d_num_selected_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DevicePartition::Flagged(
        nullptr, temp_bytes,
        d_in, d_flags, d_out, d_num_selected_out,
        num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DevicePartition::Flagged(
        temp.data(), temp_bytes,
        d_in, d_flags, d_out, d_num_selected_out,
        num_items, stream));
}

#endif  // HAS_CUDA

#endif  // CUDA_SCAN_CUH
