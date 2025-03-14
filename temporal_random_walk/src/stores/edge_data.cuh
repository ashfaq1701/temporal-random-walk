#ifndef EDGE_DATA_H
#define EDGE_DATA_H

#include <cstddef>
#include "../common/macros.cuh"
#include "../data/structs.cuh"

struct EdgeData {
    bool use_gpu;

    int* sources = nullptr;
    size_t sources_size = 0;

    int* targets = nullptr;
    size_t targets_size = 0;

    int64_t* timestamps = nullptr;
    size_t timestamps_size = 0;

    size_t* timestamp_group_offsets = nullptr;
    size_t timestamp_group_offsets_size = 0;

    int64_t* unique_timestamps = nullptr;
    size_t unique_timestamps_size = 0;

    double* forward_cumulative_weights_exponential = nullptr;
    size_t forward_cumulative_weights_exponential_size = 0;

    double* backward_cumulative_weights_exponential = nullptr;
    size_t backward_cumulative_weights_exponential_size = 0;

    explicit EdgeData(const bool use_gpu): use_gpu(use_gpu) {}

    ~EdgeData() {
        if (use_gpu) {
            if (sources) cudaFree(sources);
            if (targets) cudaFree(targets);
            if (timestamps) cudaFree(timestamps);
            if (timestamp_group_offsets) cudaFree(timestamp_group_offsets);
            if (unique_timestamps) cudaFree(unique_timestamps);
            if (forward_cumulative_weights_exponential) cudaFree(forward_cumulative_weights_exponential);
            if (backward_cumulative_weights_exponential) cudaFree(backward_cumulative_weights_exponential);
        } else {
            delete[] sources;
            delete[] targets;
            delete[] timestamps;
            delete[] timestamp_group_offsets;
            delete[] unique_timestamps;
            delete[] forward_cumulative_weights_exponential;
            delete[] backward_cumulative_weights_exponential;
        }
    }
};

namespace edge_data {
    /**
     * Common Functions
     */
    HOST void reserve(EdgeData *edge_data, size_t size);

    HOST void clear(EdgeData *edge_data);

    HOST DEVICE size_t size(const EdgeData *edge_data);

    HOST void set_size(EdgeData* edge_data, size_t size);

    HOST bool empty(const EdgeData *edge_data);

    HOST DEVICE void add_edges(EdgeData *edge_data, const int *sources, const int *targets, const int64_t *timestamps, size_t size);

    HOST DEVICE DataBlock<Edge> get_edges(const EdgeData *edge_data);

    HOST DEVICE SizeRange get_timestamp_group_range(const EdgeData *edge_data, size_t group_idx);

    HOST DEVICE size_t get_timestamp_group_count(const EdgeData *edge_data);

    HOST size_t find_group_after_timestamp(const EdgeData *edge_data, int64_t timestamp);

    HOST size_t find_group_before_timestamp(const EdgeData *edge_data, int64_t timestamp);

    /**
     * Std implementations
     */
    HOST void update_timestamp_groups_std(EdgeData *edge_data);

    HOST void update_temporal_weights_std(EdgeData *edge_data, double timescale_bound);

    /**
     * CUDA implementations
     */
    HOST void update_timestamp_groups_cuda(EdgeData *edge_data);

    HOST void update_temporal_weights_cuda(EdgeData *edge_data, double timescale_bound);

    /**
     * Device functions
     */
    DEVICE size_t find_group_after_timestamp_device(const EdgeData *edge_data, int64_t timestamp);

    DEVICE size_t find_group_before_timestamp_device(const EdgeData *edge_data, int64_t timestamp);

    HOST EdgeData* to_device_ptr(const EdgeData* edge_data);
}

#endif // EDGE_DATA_H
