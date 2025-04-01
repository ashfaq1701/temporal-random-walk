#ifndef EDGE_DATA_STORE_H
#define EDGE_DATA_STORE_H

#include <cstddef>
#include "../common/macros.cuh"
#include "../data/structs.cuh"

struct EdgeDataStore {
    bool use_gpu;
    bool owns_data;

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

    explicit EdgeDataStore(const bool use_gpu): use_gpu(use_gpu), owns_data(true) {}

    ~EdgeDataStore() {
        if (owns_data) {
            clear_memory(&sources, use_gpu);
            clear_memory(&targets, use_gpu);
            clear_memory(&timestamps, use_gpu);
            clear_memory(&timestamp_group_offsets, use_gpu);
            clear_memory(&unique_timestamps, use_gpu);
            clear_memory(&forward_cumulative_weights_exponential, use_gpu);
            clear_memory(&backward_cumulative_weights_exponential, use_gpu);
        } else {
            sources = nullptr;
            targets = nullptr;
            timestamps = nullptr;
            timestamp_group_offsets = nullptr;
            unique_timestamps = nullptr;
            forward_cumulative_weights_exponential = nullptr;
            backward_cumulative_weights_exponential = nullptr;
        }
    }
};

namespace edge_data {
    /**
     * Common Functions
     */
    HOST void resize(EdgeDataStore *edge_data, size_t size);

    HOST void clear(EdgeDataStore *edge_data);

    HOST DEVICE size_t size(const EdgeDataStore *edge_data);

    HOST void set_size(EdgeDataStore* edge_data, size_t size);

    HOST DEVICE bool empty(const EdgeDataStore *edge_data);

    HOST void add_edges(EdgeDataStore *edge_data, const int *sources, const int *targets, const int64_t *timestamps, size_t size);

    HOST DataBlock<Edge> get_edges(const EdgeDataStore *edge_data);

    HOST DEVICE SizeRange get_timestamp_group_range(const EdgeDataStore *edge_data, size_t group_idx);

    HOST DEVICE size_t get_timestamp_group_count(const EdgeDataStore *edge_data);

    HOST size_t find_group_after_timestamp(const EdgeDataStore *edge_data, int64_t timestamp);

    HOST size_t find_group_before_timestamp(const EdgeDataStore *edge_data, int64_t timestamp);

    /**
     * Std implementations
     */
    HOST void update_timestamp_groups_std(EdgeDataStore *edge_data);

    HOST void update_temporal_weights_std(EdgeDataStore *edge_data, double timescale_bound);

    #ifdef HAS_CUDA

    /**
     * CUDA implementations
     */
    HOST void update_timestamp_groups_cuda(EdgeDataStore *edge_data);

    HOST void update_temporal_weights_cuda(EdgeDataStore *edge_data, double timescale_bound);

    /**
     * Device functions
     */
    DEVICE size_t find_group_after_timestamp_device(const EdgeDataStore *edge_data, int64_t timestamp);

    DEVICE size_t find_group_before_timestamp_device(const EdgeDataStore *edge_data, int64_t timestamp);

    HOST EdgeDataStore* to_device_ptr(const EdgeDataStore* edge_data);

    #endif

}

#endif // EDGE_DATA_STORE_H
