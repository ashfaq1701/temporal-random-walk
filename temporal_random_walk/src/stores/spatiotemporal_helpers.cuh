#ifndef SPATIOTEMPORAL_HELPERS_CUH
#define SPATIOTEMPORAL_HELPERS_CUH

#include <cmath>
#include <algorithm>
#include "temporal_graph.cuh"
#include "store_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

namespace temporal_graph {
    // -------------------------------------------------------------------------
    // Common helpers
    // -------------------------------------------------------------------------

    HOST DEVICE inline int count_node_visits(
        const int *walk_nodes,
        const int walk_len,
        const int node) {
        int count = 0;
        for (int i = 0; i < walk_len; ++i) {
            if (walk_nodes[i] == node) {
                ++count;
            }
        }
        return count;
    }

    template<bool Forward>
    HOST size_t count_node_temporal_degree_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t *node_edge_offsets,
        const size_t *node_ts_groups_offsets,
        const size_t *node_ts_sorted_indices) {
        if (node_group_begin >= node_group_end) {
            return 0;
        }

        const size_t node_edge_begin = node_edge_offsets[node_id];
        const size_t node_edge_end = node_edge_offsets[node_id + 1];

        const int64_t *timestamps = graph->edge_data->timestamps;

        const size_t *begin =
                node_ts_groups_offsets + static_cast<std::ptrdiff_t>(node_group_begin);

        const size_t *end =
                node_ts_groups_offsets + static_cast<std::ptrdiff_t>(node_group_end);

        if constexpr (Forward) {
            const size_t *it = std::upper_bound(
                begin,
                end,
                timestamp,
                [timestamps, node_ts_sorted_indices]
        (const int64_t ts, const size_t group_pos) {
                    return ts < timestamps[node_ts_sorted_indices[group_pos]];
                });

            if (it == end) {
                return 0;
            }

            const auto first_group =
                    static_cast<size_t>(it - node_ts_groups_offsets);

            const size_t first_edge = node_ts_groups_offsets[first_group];
            return node_edge_end - first_edge;
        } else {
            const size_t *it = std::lower_bound(
                begin,
                end,
                timestamp,
                [timestamps, node_ts_sorted_indices]
        (const size_t group_pos, const int64_t ts) {
                    return timestamps[node_ts_sorted_indices[group_pos]] < ts;
                });

            if (it == begin) {
                return 0;
            }

            const auto first_group =
                    static_cast<size_t>(it - node_ts_groups_offsets);

            const size_t first_edge = node_ts_groups_offsets[first_group];
            return first_edge - node_edge_begin;
        }
    }

    template<bool Forward, bool IsDirected>
    HOST long pick_random_spatiotemporal_edge_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,

        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,

        const size_t node_group_begin,
        const size_t node_group_end,

        const size_t *count_ts_group_per_node,
        const size_t *node_edge_offsets,
        const size_t *node_ts_groups_offsets,
        const size_t *node_ts_sorted_indices,
        const double *weights,

        const int *walk_nodes,
        const int walk_len,

        const double selector_rand_num) {
        if (valid_node_ts_group_begin >= valid_node_ts_group_end) {
            return -1;
        }

        // ---- Cache graph parameters locally ----
        const double beta = graph->spatiotemporal_beta;
        const double gamma = graph->spatiotemporal_gamma;

        // ---- Node cache (degree + spatial) ----
        int last_node = -1;
        size_t last_degree = 0;
        double last_spatial = 0.0;

        // ---- Visit cache ----
        int last_visit_node = -1;
        int last_visit_count = 0;

        double total_weight = 0.0;

        // PASS 1: compute total weight
        for (size_t group = valid_node_ts_group_begin; group < valid_node_ts_group_end; ++group) {
            const double temporal_weight = get_group_exponential_weight_from_cumulative(
                weights,
                group,
                node_group_begin);

            const size_t edge_start = node_ts_groups_offsets[group];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group,
                node_group_end);

            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];

                const int candidate_node = get_candidate_node<Forward, IsDirected>(
                    graph,
                    node_id,
                    edge_idx);

                // ---- DEGREE + SPATIAL CACHE ----
                if (candidate_node != last_node) {
                    const size_t candidate_node_group_begin = count_ts_group_per_node[candidate_node];

                    const size_t candidate_node_group_end = count_ts_group_per_node[candidate_node + 1];

                    last_degree = count_node_temporal_degree_host<Forward>(
                        graph,
                        candidate_node,
                        timestamp,
                        candidate_node_group_begin,
                        candidate_node_group_end,
                        node_edge_offsets,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices);

                    last_spatial = last_degree > 0 ? std::exp(beta / static_cast<double>(last_degree)) : 0.0;

                    last_node = candidate_node;
                }

                // ---- VISIT CACHE ----
                if (candidate_node != last_visit_node) {
                    last_visit_count = count_node_visits(
                        walk_nodes,
                        walk_len,
                        candidate_node);

                    last_visit_node = candidate_node;
                }

                const double exploration = std::exp(-gamma * static_cast<double>(last_visit_count));

                total_weight += temporal_weight + last_spatial + exploration;
            }
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = selector_rand_num * total_weight;
        double running_sum = 0.0;

        long last_edge = -1;

        // PASS 2: sample edge
        for (size_t group = valid_node_ts_group_begin; group < valid_node_ts_group_end; ++group) {
            const double temporal_weight = get_group_exponential_weight_from_cumulative(
                weights,
                group,
                node_group_begin);

            const size_t edge_start = node_ts_groups_offsets[group];

            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group,
                node_group_end);

            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];

                const int candidate_node = get_candidate_node<Forward, IsDirected>(
                    graph,
                    node_id,
                    edge_idx);

                // ---- DEGREE + SPATIAL CACHE ----
                if (candidate_node != last_node) {
                    const size_t candidate_node_group_begin = count_ts_group_per_node[candidate_node];

                    const size_t candidate_node_group_end = count_ts_group_per_node[candidate_node + 1];

                    last_degree = count_node_temporal_degree_host<Forward>(
                        graph,
                        candidate_node,
                        timestamp,
                        candidate_node_group_begin,
                        candidate_node_group_end,
                        node_edge_offsets,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices);

                    last_spatial = last_degree > 0 ? std::exp(beta / static_cast<double>(last_degree)) : 0.0;
                    last_node = candidate_node;
                }

                // ---- VISIT CACHE ----
                if (candidate_node != last_visit_node) {
                    last_visit_count = count_node_visits(walk_nodes, walk_len, candidate_node);
                    last_visit_node = candidate_node;
                }

                const double exploration = std::exp(-gamma * static_cast<double>(last_visit_count));

                const double weight = temporal_weight + last_spatial + exploration;

                if (weight <= 0.0) {
                    continue;
                }

                last_edge = static_cast<long>(edge_idx);
                running_sum += weight;

                if (running_sum >= target) {
                    return static_cast<long>(edge_idx);
                }
            }
        }

        return last_edge;
    }

    #ifdef HAS_CUDA

    template<bool Forward>
    DEVICE size_t count_node_temporal_degree_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t *node_edge_offsets,
        const size_t *node_ts_groups_offsets,
        const size_t *node_ts_sorted_indices) {
        if (node_group_begin >= node_group_end) {
            return 0;
        }

        const size_t node_edge_begin = node_edge_offsets[node_id];
        const size_t node_edge_end = node_edge_offsets[node_id + 1];

        const int64_t *timestamps = graph->edge_data->timestamps;

        const size_t *begin =
                node_ts_groups_offsets + static_cast<std::ptrdiff_t>(node_group_begin);

        const size_t *end =
                node_ts_groups_offsets + static_cast<std::ptrdiff_t>(node_group_end);

        if constexpr (Forward) {
            const size_t *it = cuda::std::upper_bound(
                begin,
                end,
                timestamp,
                [timestamps, node_ts_sorted_indices]
                (const int64_t ts, const size_t group_pos) {
                    return ts < timestamps[node_ts_sorted_indices[group_pos]];
                });

            if (it == end) {
                return 0;
            }

            const auto first_group =
                    static_cast<size_t>(it - node_ts_groups_offsets);

            const size_t first_edge = node_ts_groups_offsets[first_group];
            return node_edge_end - first_edge;
        } else {
            const size_t *it = cuda::std::lower_bound(
                begin,
                end,
                timestamp,
                [timestamps, node_ts_sorted_indices]
                (const size_t group_pos, const int64_t ts) {
                    return timestamps[node_ts_sorted_indices[group_pos]] < ts;
                });

            if (it == begin) {
                return 0;
            }

            const auto first_group =
                    static_cast<size_t>(it - node_ts_groups_offsets);

            const size_t first_edge = node_ts_groups_offsets[first_group];
            return first_edge - node_edge_begin;
        }
    }

    template<bool Forward, bool IsDirected>
    HOST long pick_random_spatiotemporal_edge_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,

        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,

        const size_t node_group_begin,
        const size_t node_group_end,

        const size_t *count_ts_group_per_node,
        const size_t *node_edge_offsets,
        const size_t *node_ts_groups_offsets,
        const size_t *node_ts_sorted_indices,
        const double *weights,

        const int *walk_nodes,
        const int walk_len,

        const double selector_rand_num) {
        if (valid_node_ts_group_begin >= valid_node_ts_group_end) {
            return -1;
        }

        // ---- Cache graph parameters locally ----
        const double beta = graph->spatiotemporal_beta;
        const double gamma = graph->spatiotemporal_gamma;

        // ---- Node cache (degree + spatial) ----
        int last_node = -1;
        size_t last_degree = 0;
        double last_spatial = 0.0;

        // ---- Visit cache ----
        int last_visit_node = -1;
        int last_visit_count = 0;

        double total_weight = 0.0;

        // PASS 1: compute total weight
        for (size_t group = valid_node_ts_group_begin; group < valid_node_ts_group_end; ++group) {
            const double temporal_weight = get_group_exponential_weight_from_cumulative(
                weights,
                group,
                node_group_begin);

            const size_t edge_start = node_ts_groups_offsets[group];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group,
                node_group_end);

            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];

                const int candidate_node = get_candidate_node<Forward, IsDirected>(
                    graph,
                    node_id,
                    edge_idx);

                // ---- DEGREE + SPATIAL CACHE ----
                if (candidate_node != last_node) {
                    const size_t candidate_node_group_begin = count_ts_group_per_node[candidate_node];

                    const size_t candidate_node_group_end = count_ts_group_per_node[candidate_node + 1];

                    last_degree = count_node_temporal_degree_device<Forward>(
                        graph,
                        candidate_node,
                        timestamp,
                        candidate_node_group_begin,
                        candidate_node_group_end,
                        node_edge_offsets,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices);

                    last_spatial = last_degree > 0 ? std::exp(beta / static_cast<double>(last_degree)) : 0.0;

                    last_node = candidate_node;
                }

                // ---- VISIT CACHE ----
                if (candidate_node != last_visit_node) {
                    last_visit_count = count_node_visits(
                        walk_nodes,
                        walk_len,
                        candidate_node);

                    last_visit_node = candidate_node;
                }

                const double exploration = std::exp(-gamma * static_cast<double>(last_visit_count));

                total_weight += temporal_weight + last_spatial + exploration;
            }
        }

        if (total_weight <= 0.0) {
            return -1;
        }

        const double target = selector_rand_num * total_weight;
        double running_sum = 0.0;

        long last_edge = -1;

        // PASS 2: sample edge
        for (size_t group = valid_node_ts_group_begin; group < valid_node_ts_group_end; ++group) {
            const double temporal_weight = get_group_exponential_weight_from_cumulative(
                weights,
                group,
                node_group_begin);

            const size_t edge_start = node_ts_groups_offsets[group];

            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                graph,
                node_id,
                node_ts_groups_offsets,
                group,
                node_group_end);

            for (size_t i = edge_start; i < edge_end; ++i) {
                const size_t edge_idx = node_ts_sorted_indices[i];

                const int candidate_node = get_candidate_node<Forward, IsDirected>(
                    graph,
                    node_id,
                    edge_idx);

                // ---- DEGREE + SPATIAL CACHE ----
                if (candidate_node != last_node) {
                    const size_t candidate_node_group_begin = count_ts_group_per_node[candidate_node];

                    const size_t candidate_node_group_end = count_ts_group_per_node[candidate_node + 1];

                    last_degree = count_node_temporal_degree_device<Forward>(
                        graph,
                        candidate_node,
                        timestamp,
                        candidate_node_group_begin,
                        candidate_node_group_end,
                        node_edge_offsets,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices);

                    last_spatial = last_degree > 0 ? std::exp(beta / static_cast<double>(last_degree)) : 0.0;
                    last_node = candidate_node;
                }

                // ---- VISIT CACHE ----
                if (candidate_node != last_visit_node) {
                    last_visit_count = count_node_visits(walk_nodes, walk_len, candidate_node);
                    last_visit_node = candidate_node;
                }

                const double exploration = std::exp(-gamma * static_cast<double>(last_visit_count));

                const double weight = temporal_weight + last_spatial + exploration;

                if (weight <= 0.0) {
                    continue;
                }

                last_edge = static_cast<long>(edge_idx);
                running_sum += weight;

                if (running_sum >= target) {
                    return static_cast<long>(edge_idx);
                }
            }
        }

        return last_edge;
    }

    #endif
}

#endif // SPATIOTEMPORAL_HELPERS_CUH
