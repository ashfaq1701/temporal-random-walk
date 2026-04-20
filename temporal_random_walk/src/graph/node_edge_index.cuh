#ifndef NODE_EDGE_INDEX_CUH
#define NODE_EDGE_INDEX_CUH

#include <cstddef>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/buffer.cuh"

// STAGING FILE for task 5b. Not in CMake. Swapped in by task 5g.

namespace node_edge_index {

    /**
     * Common Functions
     */
    HOST void clear(TemporalGraphData& data);

    HOST DEVICE SizeRange get_edge_range(
        const TemporalGraphData& data,
        int dense_node_id,
        bool forward);

    HOST DEVICE SizeRange get_timestamp_group_range(
        const TemporalGraphData& data,
        int dense_node_id,
        size_t group_idx,
        bool forward);

    HOST DEVICE MemoryView<size_t> get_timestamp_offset_vector(
        const TemporalGraphData& data,
        bool forward);

    HOST DEVICE size_t get_timestamp_group_count(
        const TemporalGraphData& data,
        int dense_node_id,
        bool forward);

    /**
     * Rebuild allocation
     */
    HOST void allocate_node_group_offsets(
        TemporalGraphData& data,
        size_t node_index_capacity);

    HOST void allocate_node_ts_sorted_indices(TemporalGraphData& data);

    /**
     * Std rebuild compute functions
     */
    HOST void compute_node_group_offsets_std(TemporalGraphData& data);

    HOST void compute_node_ts_sorted_indices_std(
        TemporalGraphData& data,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids);

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_std(
        TemporalGraphData& data,
        size_t node_count,
        const int* outbound_node_ids,
        const int* inbound_node_ids);

    HOST void update_temporal_weights_std(
        TemporalGraphData& data,
        double timescale_bound);

    /**
     * CUDA rebuild compute functions
     */
    #ifdef HAS_CUDA

    HOST void compute_node_group_offsets_cuda(TemporalGraphData& data);

    HOST void compute_node_ts_sorted_indices_cuda(
        TemporalGraphData& data,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids);

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
        TemporalGraphData& data,
        size_t node_count,
        const int* outbound_node_ids,
        const int* inbound_node_ids);

    HOST void update_temporal_weights_cuda(
        TemporalGraphData& data,
        double timescale_bound);

    #endif

    /**
     * Top-level
     */
    HOST void rebuild(TemporalGraphData& data);

    HOST size_t get_memory_used(const TemporalGraphData& data);

}

#endif // NODE_EDGE_INDEX_CUH
