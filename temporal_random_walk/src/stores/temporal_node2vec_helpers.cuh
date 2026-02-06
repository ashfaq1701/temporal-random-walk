#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include <algorithm>
#include "temporal_graph.cuh"

namespace temporal_graph {

    // -------------------------------------------------------------------------
    // Node2Vec beta primitives
    // -------------------------------------------------------------------------

    HOST inline bool is_node_adjacent_to_host(
    const TemporalGraphStore *graph,
    const int prev_node,
    const int candidate_node)
    {
        const EdgeDataStore *edge_data = graph->edge_data;

        if (!edge_data->enable_temporal_node2vec ||
            edge_data->node_adj_offsets == nullptr) {
            return false;
            }

        const size_t start = edge_data->node_adj_offsets[prev_node];
        const size_t end   = edge_data->node_adj_offsets[prev_node + 1];

        return std::binary_search(
            edge_data->node_adj_neighbors + start,
            edge_data->node_adj_neighbors + end,
            candidate_node
        );
    }

    HOST inline double compute_node2vec_beta_host(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return graph->inv_p;
        }

        if (is_node_adjacent_to_host(graph, prev_node, w)) {
            return 1.0;
        }

        return graph->inv_q;
    }

    // -------------------------------------------------------------------------
    // Temporal-node2vec group utilities
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST DEVICE size_t get_node_group_edge_end(
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
    HOST DEVICE inline int get_node2vec_candidate_node(
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

    HOST DEVICE inline double get_group_exponential_weight_from_cumulative(
        const double *weights,
        const size_t current_group_pos,
        const size_t range_start) {
        if (current_group_pos == range_start) {
            return (range_start > 0)
                ? (weights[current_group_pos] - weights[range_start - 1])
                : weights[current_group_pos];
        }
        return weights[current_group_pos] - weights[current_group_pos - 1];
    }

    // -------------------------------------------------------------------------
    // Temporal-node2vec edge selection inside selected timestamp group
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST long pick_random_temporal_node2vec_edge_host(
    const TemporalGraphStore *graph,
    const int node_id,
    const int prev_node,
    const size_t edge_start,
    const size_t edge_end,
    const size_t * node_ts_sorted_indices,
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
    HOST int pick_random_temporal_node2vec_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t range_start,
        const size_t range_end,
        const size_t group_end_offset,
        const size_t * node_ts_groups_offsets,
        const size_t * node_ts_sorted_indices,
        const double * weights,
        const double group_selector_rand_num) {

        if (range_start >= range_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;

        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph, node_id, node_ts_groups_offsets, group_pos, group_end_offset);

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
                graph, node_id, node_ts_groups_offsets, group_pos, group_end_offset);

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


    template<bool Forward, bool IsDirected>
    HOST int pick_random_temporal_node2vec_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t range_start,
        const size_t range_end,
        const size_t group_end_offset,
        size_t *node_ts_groups_offsets,
        const size_t *node_ts_sorted_indices,
        double *weights,
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

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                beta_sum += compute_node2vec_beta_host(graph, prev_node, w);
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

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_node2vec_candidate_node<Forward, IsDirected>(graph, node_id, edge_idx);
                beta_sum += compute_node2vec_beta_host(graph, prev_node, w);
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

    DEVICE inline bool is_node_adjacent_to_device(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int candidate_node)
    {
        const EdgeDataStore *edge_data = graph->edge_data;

        if (!edge_data->enable_temporal_node2vec ||
            edge_data->node_adj_offsets == nullptr) {
            return false;
            }

        const size_t start = edge_data->node_adj_offsets[prev_node];
        const size_t end   = edge_data->node_adj_offsets[prev_node + 1];

        return cuda::std::binary_search(
            edge_data->node_adj_neighbors + start,
            edge_data->node_adj_neighbors + end,
            candidate_node
        );
    }

    DEVICE inline double compute_node2vec_beta_device(
        const TemporalGraphStore *graph,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return graph->inv_p;
        }

        if (is_node_adjacent_to_device(graph, prev_node, w)) {
            return 1.0;
        }

        return graph->inv_q;
    }

    template<bool Forward, bool IsDirected>
    DEVICE int pick_random_temporal_node2vec_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t range_start,
        const size_t range_end,
        const size_t group_end_offset,
        const size_t * node_ts_groups_offsets,
        const size_t * node_ts_sorted_indices,
        const double * weights,
        const double group_selector_rand_num) {

        if (range_start >= range_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;

        for (size_t group_pos = range_start; group_pos < range_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph, node_id, node_ts_groups_offsets, group_pos, group_end_offset);

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
                graph, node_id, node_ts_groups_offsets, group_pos, group_end_offset);

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
    DEVICE long pick_random_temporal_node2vec_edge_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        const size_t * node_ts_sorted_indices,
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

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
