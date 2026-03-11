#ifndef DATA_HELPERS_CUH
#define DATA_HELPERS_CUH

namespace temporal_graph {

    HOST DEVICE inline double get_group_exponential_weight_from_cumulative(
        const double *weights,
        const size_t current_group_pos,
        const size_t node_group_begin) {
        if (current_group_pos == node_group_begin) {
            return weights[current_group_pos];
        }

        return weights[current_group_pos] - weights[current_group_pos - 1];
    }

    template<bool Forward, bool IsDirected>
    HOST DEVICE size_t get_node_group_edge_end(
        const TemporalGraphStore *graph,
        const int node_id,
        const size_t *node_ts_groups_offsets,
        const size_t group_pos,
        const size_t node_group_end) {

        if (group_pos + 1 < node_group_end) {
            return node_ts_groups_offsets[group_pos + 1];
        }

        if constexpr (Forward) {
            return graph->node_edge_index->node_group_outbound_offsets[node_id + 1];
        }

        return IsDirected
            ? graph->node_edge_index->node_group_inbound_offsets[node_id + 1]
            : graph->node_edge_index->node_group_outbound_offsets[node_id + 1];
    }

    template<bool Forward, bool IsDirected>
    HOST DEVICE int get_candidate_node(
        const TemporalGraphStore *graph,
        const int node_id,
        const size_t edge_idx) {

        const EdgeDataStore *const edge_data = graph->edge_data;
        const int src = edge_data->sources[edge_idx];
        const int dst = edge_data->targets[edge_idx];

        if constexpr (IsDirected) {
            return Forward ? dst : src;
        }

        return (src == node_id) ? dst : src;
    }

}


#endif //DATA_HELPERS_CUH
