#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include <algorithm>
#include "../common/macros.cuh"
#include "../data/temporal_graph_view.cuh"
#include "walk_step_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#endif

namespace temporal_graph {

    // ==================== HOST ====================

    HOST inline bool is_node_adjacent_to_host(
        const TemporalGraphView& view,
        const int prev_node,
        const int candidate_node) {

        if (!view.enable_temporal_node2vec || view.node_adj_offsets == nullptr) {
            return false;
        }

        if (prev_node < 0
            || static_cast<size_t>(prev_node) + 1 >= view.node_adj_offsets_size) {
            return false;
        }

        const size_t start = view.node_adj_offsets[prev_node];
        const size_t end   = view.node_adj_offsets[prev_node + 1];

        if (start >= end) {
            return false;
        }

        return std::binary_search(
            view.node_adj_neighbors + start,
            view.node_adj_neighbors + end,
            candidate_node
        );
    }

    HOST inline double compute_node2vec_beta_host(
        const TemporalGraphView& view,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return view.inv_p;
        }

        if (is_node_adjacent_to_host(view, prev_node, w)) {
            return 1.0;
        }

        return view.inv_q;
    }

    // -------------------------------------------------------------------------
    // Temporal-node2vec edge selection inside selected timestamp group
    // -------------------------------------------------------------------------

    template<bool Forward, bool IsDirected>
    HOST long pick_random_temporal_node2vec_edge_host(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        const size_t* node_ts_sorted_indices,
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
            const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            const double beta = compute_node2vec_beta_host(view, prev_node, w);
            beta_sum += beta;
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            const double beta = compute_node2vec_beta_host(view, prev_node, w);
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
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t* node_ts_groups_offsets,
        const size_t* node_ts_sorted_indices,
        const double* weights,
        const double group_selector_rand_num) {

        if (valid_node_ts_group_begin >= valid_node_ts_group_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;

        for (size_t group_pos = valid_node_ts_group_begin; group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_host(view, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, node_group_begin);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        for (size_t group_pos = valid_node_ts_group_begin; group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_host(view, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, node_group_begin);
            running_sum += exp_weight * beta_sum;

            if (running_sum >= target) {
                selected_group = static_cast<int>(group_pos);
                break;
            }
        }

        return selected_group;
    }

    // ==================== DEVICE ====================

    #ifdef HAS_CUDA

    DEVICE inline bool is_node_adjacent_to_device(
        const TemporalGraphView& view,
        const int prev_node,
        const int candidate_node) {
        if (!view.enable_temporal_node2vec || view.node_adj_offsets == nullptr) {
            return false;
        }

        if (prev_node < 0
            || static_cast<size_t>(prev_node) + 1 >= view.node_adj_offsets_size) {
            return false;
        }

        const size_t start = view.node_adj_offsets[prev_node];
        const size_t end   = view.node_adj_offsets[prev_node + 1];

        if (start >= end) {
            return false;
        }

        const int* begin  = view.node_adj_neighbors + start;
        const int* finish = view.node_adj_neighbors + end;

        const int* it = cuda::std::lower_bound(begin, finish, candidate_node);
        return (it != finish && *it == candidate_node);
    }

    DEVICE inline double compute_node2vec_beta_device(
        const TemporalGraphView& view,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return view.inv_p;
        }

        if (is_node_adjacent_to_device(view, prev_node, w)) {
            return 1.0;
        }

        return view.inv_q;
    }

    template<bool Forward, bool IsDirected>
    DEVICE int pick_random_temporal_node2vec_device(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t* node_ts_groups_offsets,
        const size_t* node_ts_sorted_indices,
        const double* weights,
        const double group_selector_rand_num) {

        if (valid_node_ts_group_begin >= valid_node_ts_group_end || prev_node == -1) {
            return -1;
        }

        double total_weight = 0.0;

        for (size_t group_pos = valid_node_ts_group_begin; group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_device(view, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, node_group_begin);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        for (size_t group_pos = valid_node_ts_group_begin; group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);

            if (edge_start == edge_end) {
                continue;
            }

            double beta_sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_device(view, prev_node, w);
                beta_sum += beta;
            }

            const double exp_weight = get_group_exponential_weight_from_cumulative(weights, group_pos, node_group_begin);
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
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        const size_t* node_ts_sorted_indices,
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
            const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            const double beta = compute_node2vec_beta_device(view, prev_node, w);
            beta_sum += beta;
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            const double beta = compute_node2vec_beta_device(view, prev_node, w);
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
