#ifndef NODE_EDGE_INDEX_CUH
#define NODE_EDGE_INDEX_CUH

#include <cstddef>
#include <vector>

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

    // Host-safe queries: branch on data.use_gpu internally. For GPU data,
    // they cudaMemcpy the few size_t values they need. Safe to call from
    // host regardless of where the buffers live. Device-side equivalents
    // for these same queries operate on TemporalGraphView and live in
    // edge_selectors.cuh / temporal_node2vec_helpers.cuh.
    HOST SizeRange get_edge_range(
        const TemporalGraphData& data,
        int dense_node_id,
        bool forward);

    HOST SizeRange get_timestamp_group_range(
        const TemporalGraphData& data,
        int dense_node_id,
        size_t group_idx,
        bool forward);

    HOST MemoryView<size_t> get_timestamp_offset_vector(
        const TemporalGraphData& data,
        bool forward);

    HOST size_t get_timestamp_group_count(
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

    // ============================================================
    // Test / debug helpers. See edge_data::snapshot for usage notes.
    // ============================================================

    struct NodeEdgeIndexSnapshot {
        std::vector<size_t> node_group_outbound_offsets;
        std::vector<size_t> node_group_inbound_offsets;
        std::vector<size_t> node_ts_sorted_outbound_indices;
        std::vector<size_t> node_ts_sorted_inbound_indices;
        std::vector<size_t> count_ts_group_per_node_outbound;
        std::vector<size_t> count_ts_group_per_node_inbound;
        std::vector<size_t> node_ts_group_outbound_offsets;
        std::vector<size_t> node_ts_group_inbound_offsets;
        std::vector<double> outbound_forward_cumulative_weights_exponential;
        std::vector<double> outbound_backward_cumulative_weights_exponential;
        std::vector<double> inbound_backward_cumulative_weights_exponential;
    };

    HOST inline NodeEdgeIndexSnapshot snapshot(const TemporalGraphData& data) {
        return NodeEdgeIndexSnapshot{
            data.node_group_outbound_offsets.to_host_vector(),
            data.node_group_inbound_offsets.to_host_vector(),
            data.node_ts_sorted_outbound_indices.to_host_vector(),
            data.node_ts_sorted_inbound_indices.to_host_vector(),
            data.count_ts_group_per_node_outbound.to_host_vector(),
            data.count_ts_group_per_node_inbound.to_host_vector(),
            data.node_ts_group_outbound_offsets.to_host_vector(),
            data.node_ts_group_inbound_offsets.to_host_vector(),
            data.outbound_forward_cumulative_weights_exponential.to_host_vector(),
            data.outbound_backward_cumulative_weights_exponential.to_host_vector(),
            data.inbound_backward_cumulative_weights_exponential.to_host_vector(),
        };
    }

}

#endif // NODE_EDGE_INDEX_CUH
