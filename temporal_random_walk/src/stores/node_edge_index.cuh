#ifndef NODE_EDGE_INDEX_STORE_H
#define NODE_EDGE_INDEX_STORE_H

#include <cstddef>

#include "edge_data.cuh"
#include "node_mapping.cuh"
#include "../common/macros.cuh"
#include "../data/structs.cuh"

struct NodeEdgeIndexStore {
    bool use_gpu;

    size_t* outbound_offsets = nullptr;
    size_t outbound_offsets_size = 0;

    size_t* inbound_offsets = nullptr;
    size_t inbound_offsets_size = 0;

    size_t* outbound_indices = nullptr;
    size_t outbound_indices_size = 0;

    size_t* inbound_indices = nullptr;
    size_t inbound_indices_size = 0;

    size_t* outbound_timestamp_group_offsets = nullptr;
    size_t outbound_timestamp_group_offsets_size = 0;

    size_t* inbound_timestamp_group_offsets = nullptr;
    size_t inbound_timestamp_group_offsets_size = 0;

    size_t* outbound_timestamp_group_indices = nullptr;
    size_t outbound_timestamp_group_indices_size = 0;

    size_t* inbound_timestamp_group_indices = nullptr;
    size_t inbound_timestamp_group_indices_size = 0;

    double* outbound_forward_cumulative_weights_exponential = nullptr;
    size_t outbound_forward_cumulative_weights_exponential_size = 0;

    double* outbound_backward_cumulative_weights_exponential = nullptr;
    size_t outbound_backward_cumulative_weights_exponential_size = 0;

    double* inbound_backward_cumulative_weights_exponential = nullptr;
    size_t inbound_backward_cumulative_weights_exponential_size = 0;

    explicit NodeEdgeIndexStore(const bool use_gpu): use_gpu(use_gpu) {}

    ~NodeEdgeIndexStore() {
        #ifdef HAS_CUDA
        if (use_gpu) {
            if (outbound_offsets) cudaFree(outbound_offsets);
            if (inbound_offsets) cudaFree(inbound_offsets);
            if (outbound_indices) cudaFree(outbound_indices);
            if (inbound_indices) cudaFree(inbound_indices);
            if (outbound_timestamp_group_offsets) cudaFree(outbound_timestamp_group_offsets);
            if (inbound_timestamp_group_offsets) cudaFree(inbound_timestamp_group_offsets);
            if (outbound_timestamp_group_indices) cudaFree(outbound_timestamp_group_indices);
            if (inbound_timestamp_group_indices) cudaFree(inbound_timestamp_group_indices);
            if (outbound_forward_cumulative_weights_exponential) cudaFree(outbound_forward_cumulative_weights_exponential);
            if (outbound_backward_cumulative_weights_exponential) cudaFree(outbound_backward_cumulative_weights_exponential);
            if (inbound_backward_cumulative_weights_exponential) cudaFree(inbound_backward_cumulative_weights_exponential);
        }
        else
        #endif
        {
            delete[] outbound_offsets;
            delete[] inbound_offsets;
            delete[] outbound_indices;
            delete[] inbound_indices;
            delete[] outbound_timestamp_group_offsets;
            delete[] inbound_timestamp_group_offsets;
            delete[] outbound_timestamp_group_indices;
            delete[] inbound_timestamp_group_indices;
            delete[] outbound_forward_cumulative_weights_exponential;
            delete[] outbound_backward_cumulative_weights_exponential;
            delete[] inbound_backward_cumulative_weights_exponential;
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

    HOST DEVICE size_t get_timestamp_group_count(const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

    HOST DEVICE MemoryView<size_t> get_timestamp_offset_vector(const NodeEdgeIndexStore* node_edge_index, bool forward, bool is_directed);

    /**
     * Rebuild related functions
     */

    HOST void allocate_node_edge_offsets(NodeEdgeIndexStore* node_edge_index, size_t num_nodes, bool is_directed);

    HOST void allocate_node_edge_indices(NodeEdgeIndexStore* node_edge_index, bool is_directed);

    HOST void allocate_node_timestamp_indices(NodeEdgeIndexStore* node_edge_index, bool is_directed);

    /**
     * Std implementations
     */

    HOST void populate_dense_ids_std(
        EdgeDataStore* edge_data,
        NodeMappingStore* node_mapping,
        int* dense_sources,
        int* dense_targets
    );

    HOST void compute_node_edge_offsets_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        const int* dense_sources,
        const int* dense_targets,
        bool is_directed
    );

    HOST void compute_node_edge_indices_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        const int* dense_sources,
        const int* dense_targets,
        EdgeWithEndpointType* outbound_edge_indices_buffer,
        bool is_directed
    );

    HOST void compute_node_timestamp_offsets_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        size_t num_nodes,
        bool is_directed
    );

    HOST void compute_node_timestamp_indices_std(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        size_t num_nodes,
        bool is_directed
    );

    HOST void update_temporal_weights_std(NodeEdgeIndexStore* node_edge_index, const EdgeDataStore* edge_data, double timescale_bound);

    /**
     * Cuda implementations
     */

    #ifdef HAS_CUDA
    HOST void populate_dense_ids_cuda(
        const EdgeDataStore* edge_data,
        const NodeMappingStore* node_mapping,
        int* dense_sources,
        int* dense_targets
    );

    HOST void compute_node_edge_offsets_cuda(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        int* dense_sources,
        int* dense_targets,
        bool is_directed
    );

    HOST void compute_node_edge_indices_cuda(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        const int* dense_sources,
        const int* dense_targets,
        EdgeWithEndpointType* outbound_edge_indices_buffer,
        bool is_directed
    );

    HOST void compute_node_timestamp_offsets_cuda(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        size_t num_nodes,
        bool is_directed
    );

    HOST void compute_node_timestamp_indices_cuda(
        NodeEdgeIndexStore* node_edge_index,
        const EdgeDataStore* edge_data,
        size_t num_nodes,
        bool is_directed
    );

    HOST void update_temporal_weights_cuda(NodeEdgeIndexStore* node_edge_index, const EdgeDataStore* edge_data, double timescale_bound);

    HOST NodeEdgeIndexStore* to_device_ptr(const NodeEdgeIndexStore* node_edge_index);
    #endif

    HOST void rebuild(NodeEdgeIndexStore* node_edge_index, EdgeDataStore* edge_data, NodeMappingStore* node_mapping, bool is_directed);
}

#endif // NODE_EDGE_INDEX_STORE_H
