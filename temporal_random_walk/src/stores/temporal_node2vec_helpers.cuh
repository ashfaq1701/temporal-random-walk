#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include "temporal_graph.cuh"

namespace temporal_graph {

    HOST DEVICE inline bool is_node_adjacent_to_host_or_device(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int candidate_node) {
        if (graph == nullptr || graph->node_edge_index == nullptr || graph->edge_data == nullptr) {
            return false;
        }

        const SizeRange outbound_range = node_edge_index::get_edge_range(
            graph->node_edge_index,
            prev_node,
            true,
            graph->is_directed);

        for (size_t i = outbound_range.from; i < outbound_range.to; ++i) {
            const size_t edge_idx = graph->node_edge_index->node_ts_sorted_outbound_indices[i];
            if (graph->edge_data->targets[edge_idx] == candidate_node) {
                return true;
            }
        }

        if (!graph->is_directed) {
            return false;
        }

        const SizeRange inbound_range = node_edge_index::get_edge_range(
            graph->node_edge_index,
            prev_node,
            false,
            true);

        for (size_t i = inbound_range.from; i < inbound_range.to; ++i) {
            const size_t edge_idx = graph->node_edge_index->node_ts_sorted_inbound_indices[i];
            if (graph->edge_data->sources[edge_idx] == candidate_node) {
                return true;
            }
        }

        return false;
    }

    HOST inline double compute_node2vec_beta_host(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return graph->inv_p;
        }

        if (is_node_adjacent_to_host_or_device(graph, prev_node, w)) {
            return 1.0;
        }

        return graph->inv_q;
    }

    #ifdef HAS_CUDA

    DEVICE inline double compute_node2vec_beta_device(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return graph->inv_p;
        }

        if (is_node_adjacent_to_host_or_device(graph, prev_node, w)) {
            return 1.0;
        }

        return graph->inv_q;
    }

    #endif

}

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
