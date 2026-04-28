#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../common/const.cuh"
#include "../common/macros.cuh"
#include "../data/temporal_graph_view.cuh"
#include "walk_step_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#endif

namespace temporal_graph {

    // -------------------------------------------------------------------------
    // K-candidate sampling helpers (HOST + DEVICE).
    //
    // SplitMix64 derives K well-distributed pseudo-random uint64 values from
    // a single seed (the per-call rand-num the caller already had). We use
    // these to draw K uniform offsets into the candidate group and one
    // uniform [0,1) draw for the final β-weighted pick. Quality is far
    // above what a K=64 Monte Carlo estimator needs.
    //
    // Weighted selection inside the K-sample uses A-Res (Algorithm-A with
    // Reservoir, Efraimidis & Spirakis). Item with weight β gets key
    // log(u)/β; we keep the running argmax. Single pass over K, O(1) state,
    // selection probability ∝ β. (Standard derivation: argmax of u^(1/β)
    // produces the desired distribution; we work in log-space to avoid
    // pow.)
    // -------------------------------------------------------------------------

    HOST DEVICE inline uint64_t splitmix64_step(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // r ∈ [0, 1) → uint64. Uses the full mantissa.
    HOST DEVICE inline uint64_t splitmix_seed_from_u01(const double r) {
        return static_cast<uint64_t>(r * 9007199254740992.0);  // 2^53
    }

    // uint64 → [0, 1). Top 53 bits → double mantissa.
    HOST DEVICE inline double u01_from_uint64(const uint64_t x) {
        return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    }

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
        const size_t n     = end - start;

        if (n == 0) {
            return false;
        }

        const int* arr = view.node_adj_neighbors + start;

        // Linear scan beats binary search at small N (cache-friendly, no
        // dependent-load chain). Exact either way.
        if (n <= static_cast<size_t>(N2V_ADJ_LINEAR_SCAN_THRESHOLD)) {
            for (size_t i = 0; i < n; ++i) {
                if (arr[i] == candidate_node) return true;
            }
            return false;
        }

        return std::binary_search(arr, arr + n, candidate_node);
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
    // K-sample β-sum estimator. Returns an unbiased estimate of
    // Σ_{i=edge_start}^{edge_end} β(prev, candidate(i)) using K uniform
    // samples scaled up to group size. Caller seeds `state` so the same
    // K offsets can be re-derived in a second pass when needed.
    // -------------------------------------------------------------------------
    template<bool Forward, bool IsDirected>
    HOST inline double n2v_kapped_beta_sum_host(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t group_size,
        const size_t* node_ts_sorted_indices,
        uint64_t state) {

        double sum_K = 0.0;
        for (int k = 0; k < K_NODE2VEC; ++k) {
            state = splitmix64_step(state);
            const size_t off = static_cast<size_t>(state % group_size);
            const size_t edge_idx = node_ts_sorted_indices[edge_start + off];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            sum_K += compute_node2vec_beta_host(view, prev_node, w);
        }
        return (sum_K / static_cast<double>(K_NODE2VEC))
               * static_cast<double>(group_size);
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

        const size_t group_size = edge_end - edge_start;

        if (group_size == 1) {
            return static_cast<long>(node_ts_sorted_indices[edge_start]);
        }

        // Bounded-degree path: A-Res over K uniform samples.
        if (group_size > static_cast<size_t>(K_NODE2VEC)) {
            uint64_t state = splitmix_seed_from_u01(edge_selector_rand_num);

            double max_key      = -1e300;
            long   winner_edge  = -1;
            double beta_sum     = 0.0;

            for (int k = 0; k < K_NODE2VEC; ++k) {
                state = splitmix64_step(state);
                const size_t off      = static_cast<size_t>(state % group_size);
                const size_t edge_idx = node_ts_sorted_indices[edge_start + off];
                const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta     = compute_node2vec_beta_host(view, prev_node, w);
                beta_sum += beta;

                state = splitmix64_step(state);
                const double u   = u01_from_uint64(state);
                // A-Res key = log(u) / β. β > 0 always for valid Node2Vec.
                const double key = (beta > 0.0)
                                   ? std::log(u + 1e-300) / beta
                                   : -1e300;
                if (key > max_key) {
                    max_key     = key;
                    winner_edge = static_cast<long>(edge_idx);
                }
            }

            return (beta_sum > 0.0) ? winner_edge : -1;
        }

        // Exact path for small groups (n <= K).
        double beta_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            beta_sum += compute_node2vec_beta_host(view, prev_node, w);
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum  = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            running_sum += compute_node2vec_beta_host(view, prev_node, w);
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

        // Per-group seed mixes the call seed with group_pos so different
        // groups don't sample the same offsets, and so the second pass
        // can re-derive each group's K samples deterministically.
        const uint64_t base_seed = splitmix_seed_from_u01(group_selector_rand_num);

        auto group_beta_sum = [&](const size_t group_pos,
                                  const size_t edge_start,
                                  const size_t edge_end) {
            const size_t group_size = edge_end - edge_start;
            if (group_size > static_cast<size_t>(K_NODE2VEC)) {
                const uint64_t group_seed =
                    splitmix64_step(base_seed ^ static_cast<uint64_t>(group_pos));
                return n2v_kapped_beta_sum_host<Forward, IsDirected>(
                    view, node_id, prev_node,
                    edge_start, group_size,
                    node_ts_sorted_indices, group_seed);
            }
            double sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                sum += compute_node2vec_beta_host(view, prev_node, w);
            }
            return sum;
        };

        double total_weight = 0.0;
        for (size_t group_pos = valid_node_ts_group_begin;
             group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end   = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);
            if (edge_start == edge_end) continue;

            const double beta_sum   = group_beta_sum(group_pos, edge_start, edge_end);
            const double exp_weight =
                get_group_exponential_weight_from_cumulative(
                    weights, group_pos, node_group_begin);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum  = 0.0;
        int    selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        for (size_t group_pos = valid_node_ts_group_begin;
             group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end   = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);
            if (edge_start == edge_end) continue;

            const double beta_sum   = group_beta_sum(group_pos, edge_start, edge_end);
            const double exp_weight =
                get_group_exponential_weight_from_cumulative(
                    weights, group_pos, node_group_begin);
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
        const size_t n     = end - start;

        if (n == 0) {
            return false;
        }

        const int* arr = view.node_adj_neighbors + start;

        // Linear scan beats binary search at small N: the n loads can
        // overlap (ILP) instead of serializing on a dependent-load chain.
        // Exact either way — pure Tier-0.
        if (n <= static_cast<size_t>(N2V_ADJ_LINEAR_SCAN_THRESHOLD)) {
            #pragma unroll 8
            for (size_t i = 0; i < n; ++i) {
                if (arr[i] == candidate_node) return true;
            }
            return false;
        }

        const int* it = cuda::std::lower_bound(arr, arr + n, candidate_node);
        return (it != arr + n && *it == candidate_node);
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
    DEVICE inline double n2v_kapped_beta_sum_device(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t group_size,
        const size_t* node_ts_sorted_indices,
        uint64_t state) {

        double sum_K = 0.0;
        for (int k = 0; k < K_NODE2VEC; ++k) {
            state = splitmix64_step(state);
            const size_t off      = static_cast<size_t>(state % group_size);
            const size_t edge_idx = node_ts_sorted_indices[edge_start + off];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            sum_K += compute_node2vec_beta_device(view, prev_node, w);
        }
        return (sum_K / static_cast<double>(K_NODE2VEC))
               * static_cast<double>(group_size);
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

        const uint64_t base_seed = splitmix_seed_from_u01(group_selector_rand_num);

        auto group_beta_sum = [&](const size_t group_pos,
                                  const size_t edge_start,
                                  const size_t edge_end) {
            const size_t group_size = edge_end - edge_start;
            if (group_size > static_cast<size_t>(K_NODE2VEC)) {
                const uint64_t group_seed =
                    splitmix64_step(base_seed ^ static_cast<uint64_t>(group_pos));
                return n2v_kapped_beta_sum_device<Forward, IsDirected>(
                    view, node_id, prev_node,
                    edge_start, group_size,
                    node_ts_sorted_indices, group_seed);
            }
            double sum = 0.0;
            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];
                const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                sum += compute_node2vec_beta_device(view, prev_node, w);
            }
            return sum;
        };

        double total_weight = 0.0;
        for (size_t group_pos = valid_node_ts_group_begin;
             group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end   = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);
            if (edge_start == edge_end) continue;

            const double beta_sum   = group_beta_sum(group_pos, edge_start, edge_end);
            const double exp_weight =
                get_group_exponential_weight_from_cumulative(
                    weights, group_pos, node_group_begin);
            total_weight += exp_weight * beta_sum;
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = group_selector_rand_num * total_weight;
        double running_sum  = 0.0;
        int    selected_group = static_cast<int>(valid_node_ts_group_end - 1);

        for (size_t group_pos = valid_node_ts_group_begin;
             group_pos < valid_node_ts_group_end; ++group_pos) {
            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end   = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets, group_pos, node_group_end);
            if (edge_start == edge_end) continue;

            const double beta_sum   = group_beta_sum(group_pos, edge_start, edge_end);
            const double exp_weight =
                get_group_exponential_weight_from_cumulative(
                    weights, group_pos, node_group_begin);
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

        const size_t group_size = edge_end - edge_start;

        if (group_size == 1) {
            return static_cast<long>(node_ts_sorted_indices[edge_start]);
        }

        // Bounded-degree path: A-Res over K uniform samples.
        if (group_size > static_cast<size_t>(K_NODE2VEC)) {
            uint64_t state = splitmix_seed_from_u01(edge_selector_rand_num);

            double max_key      = -1e300;
            long   winner_edge  = -1;
            double beta_sum     = 0.0;

            for (int k = 0; k < K_NODE2VEC; ++k) {
                state = splitmix64_step(state);
                const size_t off      = static_cast<size_t>(state % group_size);
                const size_t edge_idx = node_ts_sorted_indices[edge_start + off];
                const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
                const double beta     = compute_node2vec_beta_device(view, prev_node, w);
                beta_sum += beta;

                state = splitmix64_step(state);
                const double u   = u01_from_uint64(state);
                // A-Res key = log(u) / β. β > 0 always for valid Node2Vec.
                const double key = (beta > 0.0)
                                   ? log(u + 1e-300) / beta
                                   : -1e300;
                if (key > max_key) {
                    max_key     = key;
                    winner_edge = static_cast<long>(edge_idx);
                }
            }

            return (beta_sum > 0.0) ? winner_edge : -1;
        }

        // Exact path for small groups (n <= K).
        double beta_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            beta_sum += compute_node2vec_beta_device(view, prev_node, w);
        }

        if (beta_sum <= 0.0) {
            return -1;
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum  = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int    w        = get_candidate_node<Forward, IsDirected>(view, node_id, edge_idx);
            running_sum += compute_node2vec_beta_device(view, prev_node, w);
            if (running_sum >= target) {
                return static_cast<long>(edge_idx);
            }
        }
        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }

    #endif

}

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
