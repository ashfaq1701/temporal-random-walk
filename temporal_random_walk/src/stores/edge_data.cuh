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
};

namespace edge_data {
    /**
     * Common Functions
     */
    HOST void reserve(EdgeData* edge_data, size_t size);

    HOST void clear (EdgeData* edge_data);

    HOST size_t size(const EdgeData* edge_data);

    HOST bool empty(const EdgeData* edge_data);

    HOST bool add_edges(int* sources, int* targets, int64_t* timestamps, size_t size);

    HOST DataBlock<Edge> get_edges(const EdgeData* edge_data);

    HOST SizeRange get_timestamp_group_range(const EdgeData* edge_data);

    HOST size_t get_timestamp_group_count(const EdgeData* edge_data);

    /**
     * Std implementations
     */
    HOST void update_timestamp_groups_std(const EdgeData* edge_data);

    HOST void update_temporal_weights_std(const EdgeData* edge_data);

    /**
     * CUDA implementations
     */
    HOST void update_temporal_weights_cuda(const EdgeData* edge_data);

    HOST void update_temporal_weights_cuda(const EdgeData* edge_data);

    /**
     * Device functions
     */
    DEVICE size_t get_timestamp_group_count_device(const EdgeData* edge_data);

    DEVICE size_t find_group_after_timestamp_device(const EdgeData* edge_data, int64_t timestamp);

    DEVICE size_t find_group_before_timestamp_device(const EdgeData* edge_data, int64_t timestamp);
}

#endif // EDGE_DATA_H
