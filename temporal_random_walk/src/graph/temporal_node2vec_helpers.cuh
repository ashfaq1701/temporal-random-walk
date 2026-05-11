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

    // ------------------------------------------------------------------------
    // Per-hop β-sum cache bounds.
    //
    // The two TN2V pickers each do a two-pass ITS: pass 1 sums β to learn
    // total_weight, pass 2 re-scans accumulating until the random target is
    // crossed.  Pass 2's β computations are pure recompute of pass 1's
    // values — a cache of bounded size lets pass 2 read from the stack
    // instead of redoing an O(log D) binary-search per edge.
    //
    // Fall back to two-pass recompute when the bound is exceeded.  Bounds
    // chosen to stay within a few cache lines (64 doubles = 512 B per
    // picker call) and register-friendly on GPU; the typical TN2V workload
    // sits well under these limits.
    constexpr size_t kNode2VecGroupCacheCap = 64;
    constexpr size_t kNode2VecEdgeCacheCap  = 64;

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

        const size_t group_size = edge_end - edge_start;
        const bool   use_cache  = group_size <= kNode2VecEdgeCacheCap;

        double beta_sum = 0.0;
        double cached_beta[kNode2VecEdgeCacheCap];

        if (use_cache) {
            // Pass 1: compute β-sum and cache per-edge β for pass 2.
            for (size_t i = 0; i < group_size; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[edge_start + i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_host(view, prev_node, w);
                cached_beta[i] = beta;
                beta_sum += beta;
            }
        } else {
            // Pass 1 (recompute path, group too large to cache).
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                beta_sum += compute_node2vec_beta_host(view, prev_node, w);
            }
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        if (use_cache) {
            // Pass 2: ITS using cached β — no recompute.
            for (size_t i = 0; i < group_size; ++i) {
                running_sum += cached_beta[i];
                if (running_sum >= target) {
                    return static_cast<long>(
                        node_ts_sorted_indices[edge_start + i]);
                }
            }
        } else {
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                running_sum += compute_node2vec_beta_host(view, prev_node, w);
                if (running_sum >= target) {
                    return static_cast<long>(edge_idx);
                }
            }
        }

        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }

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

        const size_t group_span = valid_node_ts_group_end - valid_node_ts_group_begin;
        const bool   use_cache  = group_span <= kNode2VecGroupCacheCap;

        double total_weight = 0.0;
        double cached_combined[kNode2VecGroupCacheCap];  // exp_weight * beta_sum
        size_t cached_groups[kNode2VecGroupCacheCap];     // group_pos of non-empty entries
        size_t cache_count = 0;

        // ----- Pass 1: total_weight (and cache per-group combined weight)
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
            const double combined   = exp_weight * beta_sum;

            if (use_cache) {
                cached_combined[cache_count] = combined;
                cached_groups[cache_count]   = group_pos;
                ++cache_count;
            }
            total_weight += combined;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        // ----- Pass 2: ITS pick. Cached path is O(non-empty groups) with no β recompute.
        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        if (use_cache) {
            for (size_t i = 0; i < cache_count; ++i) {
                running_sum += cached_combined[i];
                if (running_sum >= target) {
                    selected_group = static_cast<int>(cached_groups[i]);
                    return selected_group;
                }
            }
            // Numerical drift: fall through to "last non-empty" if no crossover.
            if (cache_count > 0) {
                selected_group = static_cast<int>(cached_groups[cache_count - 1]);
            }
            return selected_group;
        }

        // Recompute path (group_span > cache cap).
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
                beta_sum += compute_node2vec_beta_host(view, prev_node, w);
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

        const size_t group_span = valid_node_ts_group_end - valid_node_ts_group_begin;
        const bool   use_cache  = group_span <= kNode2VecGroupCacheCap;

        double total_weight = 0.0;
        double cached_combined[kNode2VecGroupCacheCap];  // exp_weight * beta_sum
        size_t cached_groups[kNode2VecGroupCacheCap];     // group_pos of non-empty
        size_t cache_count = 0;

        // ----- Pass 1
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
            const double combined   = exp_weight * beta_sum;

            if (use_cache) {
                cached_combined[cache_count] = combined;
                cached_groups[cache_count]   = group_pos;
                ++cache_count;
            }
            total_weight += combined;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        // ----- Pass 2
        const double target = group_selector_rand_num * total_weight;
        double running_sum = 0.0;
        int selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        if (use_cache) {
            for (size_t i = 0; i < cache_count; ++i) {
                running_sum += cached_combined[i];
                if (running_sum >= target) {
                    selected_group = static_cast<int>(cached_groups[i]);
                    return selected_group;
                }
            }
            if (cache_count > 0) {
                selected_group = static_cast<int>(cached_groups[cache_count - 1]);
            }
            return selected_group;
        }

        // Recompute path (group_span > cache cap).
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
                beta_sum += compute_node2vec_beta_device(view, prev_node, w);
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

        const size_t group_size = edge_end - edge_start;
        const bool   use_cache  = group_size <= kNode2VecEdgeCacheCap;

        double beta_sum = 0.0;
        double cached_beta[kNode2VecEdgeCacheCap];

        if (use_cache) {
            for (size_t i = 0; i < group_size; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[edge_start + i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                const double beta = compute_node2vec_beta_device(view, prev_node, w);
                cached_beta[i] = beta;
                beta_sum += beta;
            }
        } else {
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                beta_sum += compute_node2vec_beta_device(view, prev_node, w);
            }
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;

        if (use_cache) {
            for (size_t i = 0; i < group_size; ++i) {
                running_sum += cached_beta[i];
                if (running_sum >= target) {
                    return static_cast<long>(
                        node_ts_sorted_indices[edge_start + i]);
                }
            }
        } else {
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int w = get_candidate_node<Forward, IsDirected>(
                    view, node_id, edge_idx);
                running_sum += compute_node2vec_beta_device(view, prev_node, w);
                if (running_sum >= target) {
                    return static_cast<long>(edge_idx);
                }
            }
        }

        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }

    #endif

}

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
