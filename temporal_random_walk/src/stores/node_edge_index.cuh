#ifndef NODE_EDGE_INDEX_STORE_H
#define NODE_EDGE_INDEX_STORE_H

#include <cstddef>

#include "edge_data.cuh"
#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../common/error_handlers.cuh"

#include "../common/memory.cuh"

struct NodeEdgeIndexStore {
    bool use_gpu;
    bool owns_data;

    size_t* node_group_outbound_offsets = nullptr;
    size_t node_group_outbound_offsets_size = 0;

    size_t* node_group_inbound_offsets = nullptr;
    size_t node_group_inbound_offsets_size = 0;

    size_t* node_ts_sorted_outbound_indices = nullptr;
    size_t node_ts_sorted_outbound_indices_size = 0;

    size_t* node_ts_sorted_inbound_indices = nullptr;
    size_t node_ts_sorted_inbound_indices_size = 0;

    size_t* count_ts_group_per_node_outbound = nullptr;
    size_t count_ts_group_per_node_outbound_size = 0;

    size_t* count_ts_group_per_node_inbound = nullptr;
    size_t count_ts_group_per_node_inbound_size = 0;

    size_t* node_ts_group_outbound_offsets = nullptr;
    size_t node_ts_group_outbound_offsets_size = 0;

    size_t* node_ts_group_inbound_offsets = nullptr;
    size_t node_ts_group_inbound_offsets_size = 0;

    double* outbound_forward_cumulative_weights_exponential = nullptr;
    size_t outbound_forward_cumulative_weights_exponential_size = 0;

    double* outbound_backward_cumulative_weights_exponential = nullptr;
    size_t outbound_backward_cumulative_weights_exponential_size = 0;

    double* inbound_backward_cumulative_weights_exponential = nullptr;
    size_t inbound_backward_cumulative_weights_exponential_size = 0;

    explicit NodeEdgeIndexStore(const bool use_gpu): use_gpu(use_gpu), owns_data(true) {}

    ~NodeEdgeIndexStore() {
        if (owns_data) {
            clear_memory(&node_group_outbound_offsets, use_gpu);
            clear_memory(&node_group_inbound_offsets, use_gpu);
            clear_memory(&node_ts_sorted_outbound_indices, use_gpu);
            clear_memory(&node_ts_sorted_inbound_indices, use_gpu);
            clear_memory(&count_ts_group_per_node_outbound, use_gpu);
            clear_memory(&count_ts_group_per_node_inbound, use_gpu);
            clear_memory(&node_ts_group_outbound_offsets, use_gpu);
            clear_memory(&node_ts_group_inbound_offsets, use_gpu);
            clear_memory(&outbound_forward_cumulative_weights_exponential, use_gpu);
            clear_memory(&outbound_backward_cumulative_weights_exponential, use_gpu);
            clear_memory(&inbound_backward_cumulative_weights_exponential, use_gpu);
        } else {
            node_group_outbound_offsets = nullptr;
            node_group_inbound_offsets = nullptr;
            node_ts_sorted_outbound_indices = nullptr;
            node_ts_sorted_inbound_indices = nullptr;
            count_ts_group_per_node_outbound = nullptr;
            count_ts_group_per_node_inbound = nullptr;
            node_ts_group_outbound_offsets = nullptr;
            node_ts_group_inbound_offsets = nullptr;
            outbound_forward_cumulative_weights_exponential = nullptr;
            outbound_backward_cumulative_weights_exponential = nullptr;
            inbound_backward_cumulative_weights_exponential = nullptr;
        }
    }
};

namespace node_edge_index {

    /**
     * Common Functions
     */
    HOST void clear(NodeEdgeIndexStore* node_edge_index);

    HOST DEVICE SizeRange get_edge_range(const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

    HOST DEVICE SizeRange get_timestamp_group_range(const NodeEdgeIndexStore* node_edge_index, int dense_node_id, size_t group_idx, bool forward, bool is_directed);

    HOST DEVICE MemoryView<size_t> get_timestamp_offset_vector(const NodeEdgeIndexStore* node_edge_index, bool forward, bool is_directed);

    HOST DEVICE size_t get_timestamp_group_count(const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

    /**
     * Rebuild related functions
     */

    HOST void allocate_node_group_offsets(NodeEdgeIndexStore* node_edge_index, size_t node_index_capacity, bool is_directed);

    HOST void allocate_node_ts_sorted_indices(NodeEdgeIndexStore* node_edge_index, bool is_directed);

    /**
     * Std implementations
     */
    HOST void compute_node_group_offsets_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        bool is_directed
    );

    HOST void compute_node_ts_sorted_indices_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        bool is_directed,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids
    );

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        size_t node_count,
        bool is_directed,
        const int* outbound_node_ids,
        const int* inbound_node_ids
    );

    HOST void update_temporal_weights_std(NodeEdgeIndexStore *node_edge_index, const EdgeDataStore *edge_data,
                                                 double timescale_bound);

    /**
     * Cuda implementations
     */
    #ifdef HAS_CUDA

    HOST void compute_node_group_offsets_cuda(
        NodeEdgeIndexStore *node_edge_index,
        const EdgeDataStore *edge_data,
        bool is_directed
    );

    HOST void compute_node_ts_sorted_indices_cuda(
        NodeEdgeIndexStore *node_edge_index,
        const EdgeDataStore *edge_data,
        bool is_directed,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids
    );

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
        NodeEdgeIndexStore *node_edge_index,
        const EdgeDataStore *edge_data,
        size_t node_count,
        bool is_directed,
        const int* outbound_node_ids,
        const int* inbound_node_ids);

    HOST void update_temporal_weights_cuda(
        NodeEdgeIndexStore *node_edge_index,
        const EdgeDataStore *edge_data,
        double timescale_bound
    );

    HOST NodeEdgeIndexStore* to_device_ptr(const NodeEdgeIndexStore *node_edge_index);

    #endif

    HOST void rebuild(NodeEdgeIndexStore *node_edge_index, const EdgeDataStore *edge_data, bool is_directed);

    HOST size_t get_memory_used(NodeEdgeIndexStore* node_edge_index);

}

#endif // NODE_EDGE_INDEX_STORE_H
