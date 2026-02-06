#ifndef EDGE_DATA_STORE_H
#define EDGE_DATA_STORE_H

#include <cstddef>
#include <algorithm>

#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#endif

#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../common/error_handlers.cuh"
#include "../common/memory.cuh"
#include "../common/cuda_config.cuh"

struct EdgeDataStore {
    bool use_gpu;
    bool owns_data;
    bool enable_weight_computation = false;
    bool enable_temporal_node2vec = false;

    int* sources = nullptr;
    size_t sources_size = 0;

    int* targets = nullptr;
    size_t targets_size = 0;

    int64_t* timestamps = nullptr;
    size_t timestamps_size = 0;

    int* active_node_ids = nullptr;
    size_t active_node_ids_size = 0;

    // Node adjacency CSR (for temporal node2vec)
    size_t *node_adj_offsets = nullptr;
    size_t node_adj_offsets_size = 0;

    int *node_adj_neighbors = nullptr;
    size_t node_adj_neighbors_size = 0;

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
            clear_memory(&active_node_ids, use_gpu);
            clear_memory(&node_adj_offsets, use_gpu);
            clear_memory(&node_adj_neighbors, use_gpu);
            clear_memory(&timestamp_group_offsets, use_gpu);
            clear_memory(&unique_timestamps, use_gpu);
            clear_memory(&forward_cumulative_weights_exponential, use_gpu);
            clear_memory(&backward_cumulative_weights_exponential, use_gpu);
        } else {
            sources = nullptr;
            targets = nullptr;
            timestamps = nullptr;
            active_node_ids = nullptr;
            node_adj_offsets = nullptr;
            node_adj_neighbors = nullptr;
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

    HOST DEVICE size_t size(const EdgeDataStore *edge_data);

    HOST void set_size(EdgeDataStore* edge_data, size_t size);

    HOST DEVICE inline bool empty(const EdgeDataStore *edge_data) {
        return edge_data->timestamps_size == 0;
    }

    HOST void add_edges(EdgeDataStore *edge_data, const int *sources, const int *targets, const int64_t *timestamps, size_t size);

    HOST DataBlock<Edge> get_edges(const EdgeDataStore *edge_data);

    HOST DEVICE inline SizeRange get_timestamp_group_range(const EdgeDataStore *edge_data, const size_t group_idx) {
        if (group_idx >= edge_data->unique_timestamps_size) {
            return SizeRange{0, 0};
        }

        return SizeRange{edge_data->timestamp_group_offsets[group_idx], edge_data->timestamp_group_offsets[group_idx + 1]};
    }

    HOST DEVICE inline size_t get_timestamp_group_count(const EdgeDataStore *edge_data) {
        return edge_data->unique_timestamps_size;
    }

    HOST inline size_t find_group_after_timestamp(const EdgeDataStore *edge_data, const int64_t timestamp) {
        if (edge_data->unique_timestamps_size == 0) return 0;

        // Get raw pointer to data and use std::upper_bound directly
        const int64_t* begin = edge_data->unique_timestamps;
        const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

        const auto it = std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    HOST inline size_t find_group_before_timestamp(const EdgeDataStore *edge_data, const int64_t timestamp) {
        if (edge_data->unique_timestamps_size == 0) return 0;

        // Get raw pointer to data and use std::lower_bound directly
        const int64_t* begin = edge_data->unique_timestamps;
        const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

        const auto it = std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    HOST inline bool is_node_active_host(const EdgeDataStore* edge_data, const int node_id) {
        // Check bounds first to avoid out-of-bounds access
        if (node_id < 0 || node_id >= edge_data->active_node_ids_size) {
            return false;
        }

        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            // For GPU implementation, need to copy value from device to host
            int is_active;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&is_active, edge_data->active_node_ids + node_id, sizeof(int), cudaMemcpyDeviceToHost));
            return is_active == 1;
        }
        else
        #endif
        {
            // For CPU implementation, direct access is sufficient
            return edge_data->active_node_ids[node_id] == 1;
        }
    }

    HOST DataBlock<int> get_active_node_ids(const EdgeDataStore* edge_data);

    HOST size_t active_node_count(const EdgeDataStore* edge_data);

    /**
     * Std implementations
     */

    HOST void populate_active_nodes_std(EdgeDataStore* edge_data);

    HOST void build_node_adjacency_csr(EdgeDataStore *edge_data);

    HOST void update_timestamp_groups_std(EdgeDataStore *edge_data);

    HOST void update_temporal_weights_std(EdgeDataStore *edge_data, double timescale_bound);

    #ifdef HAS_CUDA

    /**
     * CUDA implementations
     */

    HOST void populate_active_nodes_cuda(EdgeDataStore* edge_data);

    HOST void build_node_adjacency_csr_cuda(EdgeDataStore *edge_data);

    HOST void update_timestamp_groups_cuda(EdgeDataStore *edge_data);

    HOST void update_temporal_weights_cuda(EdgeDataStore *edge_data, double timescale_bound);

    /**
     * Device functions
     */

    DEVICE inline size_t find_group_after_timestamp_device(const EdgeDataStore *edge_data, const int64_t timestamp) {
        if (edge_data->unique_timestamps_size == 0) return 0;

        const int64_t* begin = edge_data->unique_timestamps;
        const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

        const auto it = cuda::std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    DEVICE inline size_t find_group_before_timestamp_device(const EdgeDataStore *edge_data, const int64_t timestamp) {
        if (edge_data->unique_timestamps_size == 0) return 0;

        const int64_t* begin = edge_data->unique_timestamps;
        const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

        const auto it = cuda::std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    DEVICE inline bool is_node_active_device(const EdgeDataStore* edge_data, const int node_id) {
        // Check bounds first to avoid out-of-bounds access
        if (node_id < 0 || node_id >= edge_data->active_node_ids_size) {
            return false;
        }

        return edge_data->active_node_ids[node_id] == 1;
    }

    HOST EdgeDataStore *to_device_ptr(const EdgeDataStore *edge_data);

    #endif

    HOST size_t get_memory_used(EdgeDataStore* edge_data);

}

#endif // EDGE_DATA_STORE_H
