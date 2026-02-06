#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include "temporal_graph.cuh"

namespace temporal_graph {

    #ifdef HAS_CUDA
    #define NODE2VEC_FORCE_INLINE __forceinline__
    #else
    #define NODE2VEC_FORCE_INLINE inline
    #endif

    // -------------------------------------------------------------------------
    // Node2Vec beta primitives
    // -------------------------------------------------------------------------

    HOST DEVICE NODE2VEC_FORCE_INLINE bool is_node_adjacent_to(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int candidate_node) {
        if (graph == nullptr || graph->node_edge_index == nullptr || graph->edge_data == nullptr) {
            return false;
        }

        const NodeEdgeIndexStore *const node_edge_index = graph->node_edge_index;
        const EdgeDataStore *const edge_data = graph->edge_data;
        const bool is_directed = graph->is_directed;

        const SizeRange outbound_range = node_edge_index::get_edge_range(
            node_edge_index,
            prev_node,
            true,
            is_directed);

        if (!is_directed) {
            for (size_t i = outbound_range.from; i < outbound_range.to; ++i) {
                const size_t edge_idx = node_edge_index->node_ts_sorted_outbound_indices[i];
                if (edge_data->targets[edge_idx] == candidate_node) {
                    return true;
                }
            }

            return false;
        }

        const SizeRange inbound_range = node_edge_index::get_edge_range(
            node_edge_index,
            prev_node,
            false,
            true);

        const size_t outbound_size = outbound_range.to - outbound_range.from;
        const size_t inbound_size = inbound_range.to - inbound_range.from;

        if (outbound_size <= inbound_size) {
            for (size_t i = outbound_range.from; i < outbound_range.to; ++i) {
                const size_t edge_idx = node_edge_index->node_ts_sorted_outbound_indices[i];
                if (edge_data->targets[edge_idx] == candidate_node) {
                    return true;
                }
            }

            for (size_t i = inbound_range.from; i < inbound_range.to; ++i) {
                const size_t edge_idx = node_edge_index->node_ts_sorted_inbound_indices[i];
                if (edge_data->sources[edge_idx] == candidate_node) {
                    return true;
                }
            }
            return false;
        }

        for (size_t i = inbound_range.from; i < inbound_range.to; ++i) {
            const size_t edge_idx = node_edge_index->node_ts_sorted_inbound_indices[i];
            if (edge_data->sources[edge_idx] == candidate_node) {
                return true;
            }
        }

        for (size_t i = outbound_range.from; i < outbound_range.to; ++i) {
            const size_t edge_idx = node_edge_index->node_ts_sorted_outbound_indices[i];
            if (edge_data->targets[edge_idx] == candidate_node) {
                return true;
            }
        }

        return false;
    }

    HOST NODE2VEC_FORCE_INLINE double compute_node2vec_beta_host(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        const double inv_p = graph->inv_p;
        const double inv_q = graph->inv_q;

        if (w == prev_node) {
            return inv_p;
        }

        if (is_node_adjacent_to(graph, prev_node, w)) {
            return 1.0;
        }

        return inv_q;
    }

    #ifdef HAS_CUDA

    DEVICE NODE2VEC_FORCE_INLINE double compute_node2vec_beta_device(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        const double inv_p = graph->inv_p;
        const double inv_q = graph->inv_q;

        if (w == prev_node) {
            return inv_p;
        }

        if (is_node_adjacent_to(graph, prev_node, w)) {
            return 1.0;
        }

        return inv_q;
    }

    #endif

    // -------------------------------------------------------------------------
    // Temporal-node2vec group utilities
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST DEVICE NODE2VEC_FORCE_INLINE size_t get_node_group_edge_end(
        const TemporalGraphStore *graph,
        const int node_id,
        const size_t *node_ts_groups_offsets,
        const size_t group_pos,
        const size_t group_end_offset) {
        if (group_pos + 1 < group_end_offset) {
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
    HOST DEVICE NODE2VEC_FORCE_INLINE int get_node2vec_candidate_node(
        const TemporalGraphStore *graph,
        const int node_id,
        const size_t edge_idx) {
        const EdgeDataStore *const edge_data = graph->edge_data;
        const int src = edge_data->sources[edge_idx];
        const int dst = edge_data->targets[edge_idx];

        if constexpr (IsDirected) {
            return Forward ? dst : src;
        }

        return src == node_id ? dst : src;
    }

    HOST DEVICE NODE2VEC_FORCE_INLINE double get_group_exponential_weight_from_cumulative(
        const double *weights,
        const size_t current_group_pos,
        const size_t range_start) {
        if (current_group_pos == range_start) {
            return range_start > 0
                       ? (weights[current_group_pos] - weights[range_start - 1])
                       : weights[current_group_pos];
        }

        return weights[current_group_pos] - weights[current_group_pos - 1];
    }

    // -------------------------------------------------------------------------
    // Temporal-node2vec edge selection inside selected timestamp group
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST inline long pick_random_temporal_node2vec_edge_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        const size_t * __restrict__ node_ts_sorted_indices,
        const double edge_selector_rand_num) {
        if (prev_node == -1 || edge_start >= edge_end) {
            return -1;
        }

        if (edge_end - edge_start == 1) {
            return static_cast<long>(node_ts_sorted_indices[edge_start]);
        }

        double beta_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
            const double beta = compute_node2vec_beta_host(graph, prev_node, w);
            beta_sum += beta;
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
            const double beta = compute_node2vec_beta_host(graph, prev_node, w);
            running_sum += beta;
            if (running_sum >= target) {
                return static_cast<long>(edge_idx);
            }
        }

        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }

    // -------------------------------------------------------------------------
    // Temporal-node2vec timestamp-group selection
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST inline int pick_random_temporal_node2vec_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t range_start,
        const size_t range_end,
        const size_t group_end_offset,
        const size_t * __restrict__ node_ts_groups_offsets,
        const size_t * __restrict__ node_ts_sorted_indices,
        const double * __restrict__ weights,
        const double group_selector_rand_num) {
        if (range_start >= range_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;
        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group_pos,
                group_end_offset);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                const double beta = compute_node2vec_beta_host(graph, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, range_start);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(range_end - 1);

        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group_pos,
                group_end_offset);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                const double beta = compute_node2vec_beta_host(graph, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, range_start);
            running_sum += exp_weight * beta_sum;
            if (running_sum >= target) {
                selected_group = static_cast<int>(group_pos);
                break;
            }
        }

        return selected_group;
    }

    #ifdef HAS_CUDA

    template<bool Forward, bool IsDirected>
    DEVICE inline int pick_random_temporal_node2vec_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t range_start,
        const size_t range_end,
        const size_t group_end_offset,
        const size_t * __restrict__ node_ts_groups_offsets,
        const size_t * __restrict__ node_ts_sorted_indices,
        const double * __restrict__ weights,
        const double group_selector_rand_num) {
        if (range_start >= range_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;
        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group_pos,
                group_end_offset);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                const double beta = compute_node2vec_beta_device(graph, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, range_start);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(range_end - 1);

        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group_pos,
                group_end_offset);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                const double beta = compute_node2vec_beta_device(graph, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, range_start);
            running_sum += exp_weight * beta_sum;
            if (running_sum >= target) {
                selected_group = static_cast<int>(group_pos);
                break;
            }
        }

        return selected_group;
    }

    template<bool Forward, bool IsDirected>
    DEVICE inline long pick_random_temporal_node2vec_edge_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        const size_t * __restrict__ node_ts_sorted_indices,
        const double edge_selector_rand_num) {
        if (prev_node == -1 || edge_start >= edge_end) {
            return -1;
        }

        if (edge_end - edge_start == 1) {
            return static_cast<long>(node_ts_sorted_indices[edge_start]);
        }

        double beta_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
            const double beta = compute_node2vec_beta_device(graph, prev_node, w);
            beta_sum += beta;
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
            const double beta = compute_node2vec_beta_device(graph, prev_node, w);
            running_sum += beta;
            if (running_sum >= target) {
                return static_cast<long>(edge_idx);
            }
        }

        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }

    #endif

}

#undef NODE2VEC_FORCE_INLINE

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
