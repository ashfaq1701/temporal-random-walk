#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include "temporal_graph.cuh"
#include "temporal_node2vec_helpers.cuh"
#include "spatiotemporal_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

#include "../random/pickers.cuh"
#include "../utils/random.cuh"

namespace temporal_graph {

    /**
     * Host functions
     */

    template<bool Forward, RandomPickerType PickerType>
    HOST InternalEdge get_edge_at_host(
        const TemporalGraphStore *graph,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (edge_data::empty(graph->edge_data)) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = edge_data::find_group_after_timestamp(graph->edge_data, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = edge_data::find_group_before_timestamp(graph->edge_data, timestamp);
                if (last_group == static_cast<size_t>(-1)) return InternalEdge{-1, -1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        // Get selected group's boundaries
        const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            graph->edge_data->sources[random_idx],
            graph->edge_data->targets[random_idx],
            graph->edge_data->timestamps[random_idx],
            static_cast<int64_t>(random_idx)
        };
    }


    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    HOST InternalEdge get_node_edge_at_host(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num,
        const int* walk_nodes,
        const int walk_len) {
        if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Get appropriate node indices based on direction and graph type
        const size_t *count_ts_group_per_node = Forward
                                                    ? graph->node_edge_index->count_ts_group_per_node_outbound
                                                    : (IsDirected
                                                           ? graph->node_edge_index->count_ts_group_per_node_inbound
                                                           : graph->node_edge_index->count_ts_group_per_node_outbound);

        const size_t *node_ts_groups_offsets = Forward
                                              ? graph->node_edge_index->node_ts_group_outbound_offsets
                                              : (IsDirected
                                                     ? graph->node_edge_index->node_ts_group_inbound_offsets
                                                     : graph->node_edge_index->node_ts_group_outbound_offsets);

        const size_t *node_ts_sorted_indices = Forward
                                   ? graph->node_edge_index->node_ts_sorted_outbound_indices
                                   : (IsDirected
                                          ? graph->node_edge_index->node_ts_sorted_inbound_indices
                                          : graph->node_edge_index->node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size = Forward
                                   ? graph->node_edge_index->node_ts_sorted_outbound_indices_size
                                   : (IsDirected
                                          ? graph->node_edge_index->node_ts_sorted_inbound_indices_size
                                          : graph->node_edge_index->node_ts_sorted_outbound_indices_size);

        // Node edge boundaries
        const size_t *node_edge_offsets = Forward
                                    ? graph->node_edge_index->node_group_outbound_offsets
                                    : (IsDirected
                                        ? graph->node_edge_index->node_group_inbound_offsets
                                        : graph->node_edge_index->node_group_outbound_offsets);

        // Temporal exponential cumulative weights
        const double *weights = Forward
                                    ? graph->node_edge_index->outbound_forward_cumulative_weights_exponential
                                    : (IsDirected
                                        ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                        : graph->node_edge_index->outbound_backward_cumulative_weights_exponential);

        const size_t weights_size = Forward
                                    ? graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size
                                    : (IsDirected
                                        ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                                        : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size);

        // Get node's group range
        const size_t node_group_start_offset = count_ts_group_per_node[node_id];
        const size_t node_group_end_offset = count_ts_group_per_node[node_id + 1];
        if (node_group_start_offset == node_group_end_offset) return InternalEdge{-1, -1, -1, -1};

        long group_pos;
        if (timestamp != -1) {
            if constexpr (Forward) {
                // Find first group after timestamp
                const auto it = std::upper_bound(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset),
                    timestamp,
                    [graph, node_ts_sorted_indices](const int64_t ts, const size_t pos) {
                        return ts < graph->edge_data->timestamps[node_ts_sorted_indices[pos]];
                    });

                // Count available groups after timestamp
                const size_t available = std::distance(
                    it,
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset));
                if (available == 0) return InternalEdge{-1, -1, -1, -1};

                const size_t valid_node_ts_slice_start_pos = it - node_ts_groups_offsets;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available), false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available) return InternalEdge{-1, -1, -1, -1};
                    group_pos = static_cast<long>(valid_node_ts_slice_start_pos) + index;
                } else {
                    if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                        if (prev_node == -1) {
                            group_pos = random_pickers::pick_using_weight_based_picker_host(
                                RandomPickerType::ExponentialWeight,
                                weights,
                                weights_size,
                                valid_node_ts_slice_start_pos, node_group_end_offset, group_selector_rand_num);
                        } else {
                            group_pos = pick_random_temporal_node2vec_host<Forward, IsDirected>(
                                graph,
                                node_id,
                                prev_node,
                                valid_node_ts_slice_start_pos,
                                node_group_end_offset,
                                node_group_start_offset,
                                node_group_end_offset,
                                node_ts_groups_offsets,
                                node_ts_sorted_indices,
                                weights,
                                group_selector_rand_num);
                        }
                    } else if (PickerType == RandomPickerType::SpatioTemporal) {
                        group_pos = pick_random_spatiotemporal_edge_host<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            valid_node_ts_slice_start_pos,
                            node_group_end_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                    } else {
                        group_pos = random_pickers::pick_using_weight_based_picker_host(
                            PickerType,
                            weights,
                            weights_size,
                            valid_node_ts_slice_start_pos, node_group_end_offset, group_selector_rand_num);
                    }
                    if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                // Find first group >= timestamp
                auto it = std::lower_bound(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset),
                    timestamp,
                    [graph, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                        return graph->edge_data->timestamps[node_ts_sorted_indices[pos]] < ts;
                    });

                const size_t available = std::distance(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    it);
                if (available == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available) return InternalEdge{-1, -1, -1, -1};
                    group_pos = static_cast<long>((it - node_ts_groups_offsets) - 1 - (available - index - 1));
                } else {
                    const auto valid_node_ts_slice_end_pos = static_cast<size_t>(it - node_ts_groups_offsets);
                    if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                        if (prev_node == -1) {
                            group_pos = random_pickers::pick_using_weight_based_picker_host(
                                RandomPickerType::ExponentialWeight,
                                weights,
                                weights_size,
                                node_group_start_offset,
                                valid_node_ts_slice_end_pos,
                                group_selector_rand_num
                            );
                        } else {
                            group_pos = pick_random_temporal_node2vec_host<Forward, IsDirected>(
                                graph,
                                node_id,
                                prev_node,
                                node_group_start_offset,
                                valid_node_ts_slice_end_pos,
                                node_group_start_offset,
                                node_group_end_offset,
                                node_ts_groups_offsets,
                                node_ts_sorted_indices,
                                weights,
                                group_selector_rand_num);
                        }
                    } else if (PickerType == RandomPickerType::SpatioTemporal) {
                        group_pos = pick_random_spatiotemporal_edge_host<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            node_group_start_offset,
                            valid_node_ts_slice_end_pos,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                    } else {
                        group_pos = random_pickers::pick_using_weight_based_picker_host(
                            PickerType,
                            weights,
                            weights_size,
                            node_group_start_offset,
                            valid_node_ts_slice_end_pos,
                            group_selector_rand_num
                        );
                    }
                    if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            const size_t num_groups = node_group_end_offset - node_group_start_offset;
            if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_pos = Forward
                                ? static_cast<long>(node_group_start_offset + index)
                                : static_cast<long>(node_group_end_offset - 1 - (num_groups - index - 1));
            } else {

                if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                    if (prev_node == -1) {
                        group_pos = random_pickers::pick_using_weight_based_picker_host(
                            RandomPickerType::ExponentialWeight,
                            weights,
                            weights_size,
                            node_group_start_offset,
                            node_group_end_offset,
                            group_selector_rand_num);
                    } else {
                        group_pos = pick_random_temporal_node2vec_host<Forward, IsDirected>(
                            graph,
                            node_id,
                            prev_node,
                            node_group_start_offset,
                            node_group_end_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            group_selector_rand_num);
                    }
                } else if (PickerType == RandomPickerType::SpatioTemporal) {
                    group_pos = pick_random_spatiotemporal_edge_host<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            node_group_start_offset,
                            node_group_start_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                } else {
                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        weights,
                        weights_size,
                        node_group_start_offset,
                        node_group_end_offset,
                        group_selector_rand_num);
                }
                if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
            }
        }

        // Get edge range for selected group
        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        size_t valid_edge_end;

        if (group_pos + 1 < node_group_end_offset) {
            valid_edge_end = node_ts_groups_offsets[group_pos + 1];
        } else {
            valid_edge_end = node_edge_offsets[node_id + 1];
        }

        // Validate range before random selection
        if (valid_edge_start >= valid_edge_end || valid_edge_start >= node_ts_sorted_indices_size || valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        long edge_idx;
        if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
            if (prev_node == -1) {
                edge_idx = static_cast<long>(node_ts_sorted_indices[
                    valid_edge_start +
                    generate_random_number_bounded_by(static_cast<int>(valid_edge_end - valid_edge_start), edge_selector_rand_num)]);
            } else {
                edge_idx = pick_random_temporal_node2vec_edge_host<Forward, IsDirected>(
                    graph,
                    node_id,
                    prev_node,
                    valid_edge_start,
                    valid_edge_end,
                    node_ts_sorted_indices,
                    edge_selector_rand_num);
                if (edge_idx == -1) return InternalEdge{-1, -1, -1, -1};
            }
        } else {
            edge_idx = static_cast<long>(node_ts_sorted_indices[
                valid_edge_start +
                generate_random_number_bounded_by(static_cast<int>(valid_edge_end - valid_edge_start), edge_selector_rand_num)]);
        }

        return InternalEdge{
            graph->edge_data->sources[edge_idx],
            graph->edge_data->targets[edge_idx],
            graph->edge_data->timestamps[edge_idx],
            edge_idx
        };
    }

    /**
     * Device functions
     */

    #ifdef HAS_CUDA

    template<bool Forward, RandomPickerType PickerType>
    DEVICE InternalEdge get_edge_at_device(
        const TemporalGraphStore *graph,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (edge_data::empty(graph->edge_data)) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = edge_data::find_group_after_timestamp_device(graph->edge_data, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = edge_data::find_group_before_timestamp_device(graph->edge_data, timestamp);
                if (last_group == static_cast<size_t>(-1)) return InternalEdge{-1, -1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        // Get selected group's boundaries
        const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            graph->edge_data->sources[random_idx],
            graph->edge_data->targets[random_idx],
            graph->edge_data->timestamps[random_idx],
            static_cast<int64_t>(random_idx),
        };
    }

    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    DEVICE InternalEdge get_node_edge_at_device(
        const TemporalGraphStore *graph,
        const int node_id,
        const int64_t timestamp,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num,
        const int* walk_nodes=nullptr,
        const int walk_len=0) {
        if (!edge_data::is_node_active_device(graph->edge_data, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Get appropriate node indices based on direction and graph type
        const size_t *count_ts_group_per_node = Forward
                                                    ? graph->node_edge_index->count_ts_group_per_node_outbound
                                                    : (IsDirected
                                                           ? graph->node_edge_index->count_ts_group_per_node_inbound
                                                           : graph->node_edge_index->count_ts_group_per_node_outbound);

        const size_t *node_ts_groups_offsets = Forward
                                              ? graph->node_edge_index->node_ts_group_outbound_offsets
                                              : (IsDirected
                                                     ? graph->node_edge_index->node_ts_group_inbound_offsets
                                                     : graph->node_edge_index->node_ts_group_outbound_offsets);

        const size_t *node_ts_sorted_indices = Forward
                                   ? graph->node_edge_index->node_ts_sorted_outbound_indices
                                   : (IsDirected
                                          ? graph->node_edge_index->node_ts_sorted_inbound_indices
                                          : graph->node_edge_index->node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size = Forward
                                   ? graph->node_edge_index->node_ts_sorted_outbound_indices_size
                                   : (IsDirected
                                          ? graph->node_edge_index->node_ts_sorted_inbound_indices_size
                                          : graph->node_edge_index->node_ts_sorted_outbound_indices_size);

        // Node edge boundaries
        const size_t *node_edge_offsets = Forward
                                    ? graph->node_edge_index->node_group_outbound_offsets
                                    : (IsDirected
                                        ? graph->node_edge_index->node_group_inbound_offsets
                                        : graph->node_edge_index->node_group_outbound_offsets);

        // Temporal exponential cumulative weights
        const double *weights = Forward
                                    ? graph->node_edge_index->outbound_forward_cumulative_weights_exponential
                                    : (IsDirected
                                        ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                        : graph->node_edge_index->outbound_backward_cumulative_weights_exponential);

        const size_t weights_size = Forward
                                    ? graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size
                                    : (IsDirected
                                        ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                                        : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size);

        // Get node's group range
        const size_t node_group_start_offset = count_ts_group_per_node[node_id];
        const size_t node_group_end_offset = count_ts_group_per_node[node_id + 1];
        if (node_group_start_offset == node_group_end_offset) return InternalEdge{-1, -1, -1, -1};

        long group_pos;
        if (timestamp != -1) {
            if constexpr (Forward) {
                // Find first group after timestamp
                const auto it = cuda::std::upper_bound(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset),
                    timestamp,
                    [graph, node_ts_sorted_indices](const int64_t ts, const size_t pos) {
                        return ts < graph->edge_data->timestamps[node_ts_sorted_indices[pos]];
                    });

                // Count available groups after timestamp
                const size_t available = std::distance(
                    it,
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset));
                if (available == 0) return InternalEdge{-1, -1, -1, -1};

                const size_t valid_node_ts_slice_start_pos = it - node_ts_groups_offsets;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available), false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available) return InternalEdge{-1, -1, -1, -1};
                    group_pos = static_cast<long>(valid_node_ts_slice_start_pos) + index;
                } else {
                    if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                        if (prev_node == -1) {
                            group_pos = random_pickers::pick_using_weight_based_picker_device(
                                RandomPickerType::ExponentialWeight,
                                weights,
                                weights_size,
                                valid_node_ts_slice_start_pos, node_group_end_offset, group_selector_rand_num);
                        } else {
                            group_pos = pick_random_temporal_node2vec_device<Forward, IsDirected>(
                                graph,
                                node_id,
                                prev_node,
                                valid_node_ts_slice_start_pos,
                                node_group_end_offset,
                                node_group_start_offset,
                                node_group_end_offset,
                                node_ts_groups_offsets,
                                node_ts_sorted_indices,
                                weights,
                                group_selector_rand_num);
                        }
                    } else if (PickerType == RandomPickerType::SpatioTemporal) {
                        group_pos = pick_random_spatiotemporal_edge_device<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            valid_node_ts_slice_start_pos,
                            node_group_end_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                    } else {
                        group_pos = random_pickers::pick_using_weight_based_picker_device(
                            PickerType,
                            weights,
                            weights_size,
                            valid_node_ts_slice_start_pos, node_group_end_offset, group_selector_rand_num);
                    }
                    if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                // Find first group >= timestamp
                auto it = cuda::std::lower_bound(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    node_ts_groups_offsets + static_cast<int>(node_group_end_offset),
                    timestamp,
                    [graph, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                        return graph->edge_data->timestamps[node_ts_sorted_indices[pos]] < ts;
                    });

                const size_t available = std::distance(
                    node_ts_groups_offsets + static_cast<int>(node_group_start_offset),
                    it);
                if (available == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available) return InternalEdge{-1, -1, -1, -1};
                    group_pos = static_cast<long>((it - node_ts_groups_offsets) - 1 - (available - index - 1));
                } else {
                    const auto valid_node_ts_slice_end_pos = static_cast<size_t>(it - node_ts_groups_offsets);
                    if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                        if (prev_node == -1) {
                            group_pos = random_pickers::pick_using_weight_based_picker_device(
                                RandomPickerType::ExponentialWeight,
                                weights,
                                weights_size,
                                node_group_start_offset,
                                valid_node_ts_slice_end_pos,
                                group_selector_rand_num
                            );
                        } else {
                            group_pos = pick_random_temporal_node2vec_device<Forward, IsDirected>(
                                graph,
                                node_id,
                                prev_node,
                                node_group_start_offset,
                                valid_node_ts_slice_end_pos,
                                node_group_start_offset,
                                node_group_end_offset,
                                node_ts_groups_offsets,
                                node_ts_sorted_indices,
                                weights,
                                group_selector_rand_num);
                        }
                    } else if (PickerType == RandomPickerType::SpatioTemporal) {
                        group_pos = pick_random_spatiotemporal_edge_device<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            node_group_start_offset,
                            valid_node_ts_slice_end_pos,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                    } else {
                        group_pos = random_pickers::pick_using_weight_based_picker_device(
                            PickerType,
                            weights,
                            weights_size,
                            node_group_start_offset,
                            valid_node_ts_slice_end_pos,
                            group_selector_rand_num
                        );
                    }
                    if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            const size_t num_groups = node_group_end_offset - node_group_start_offset;
            if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_pos = Forward
                                ? static_cast<long>(node_group_start_offset + index)
                                : static_cast<long>(node_group_end_offset - 1 - (num_groups - index - 1));
            } else {

                if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                    if (prev_node == -1) {
                        group_pos = random_pickers::pick_using_weight_based_picker_device(
                            RandomPickerType::ExponentialWeight,
                            weights,
                            weights_size,
                            node_group_start_offset,
                            node_group_end_offset,
                            group_selector_rand_num);
                    } else {
                        group_pos = pick_random_temporal_node2vec_device<Forward, IsDirected>(
                            graph,
                            node_id,
                            prev_node,
                            node_group_start_offset,
                            node_group_end_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            group_selector_rand_num);
                    }
                } else if (PickerType == RandomPickerType::SpatioTemporal) {
                    group_pos = pick_random_spatiotemporal_edge_device<Forward, IsDirected>(
                            graph,
                            node_id,
                            timestamp,
                            node_group_start_offset,
                            node_group_end_offset,
                            node_group_start_offset,
                            node_group_end_offset,
                            count_ts_group_per_node,
                            node_edge_offsets,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights,
                            walk_nodes,
                            walk_len,
                            edge_selector_rand_num);
                } else {
                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        weights,
                        weights_size,
                        node_group_start_offset,
                        node_group_end_offset,
                        group_selector_rand_num);
                }
                if (group_pos == -1) return InternalEdge{-1, -1, -1, -1};
            }
        }

        // Get edge range for selected group
        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        size_t valid_edge_end;

        if (group_pos + 1 < node_group_end_offset) {
            valid_edge_end = node_ts_groups_offsets[group_pos + 1];
        } else {
            valid_edge_end = node_edge_offsets[node_id + 1];
        }

        // Validate range before random selection
        if (valid_edge_start >= valid_edge_end || valid_edge_start >= node_ts_sorted_indices_size || valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        long edge_idx;
        if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
            if (prev_node == -1) {
                edge_idx = static_cast<long>(node_ts_sorted_indices[
                    valid_edge_start +
                    generate_random_number_bounded_by(static_cast<int>(valid_edge_end - valid_edge_start), edge_selector_rand_num)]);
            } else {
                edge_idx = pick_random_temporal_node2vec_edge_device<Forward, IsDirected>(
                    graph,
                    node_id,
                    prev_node,
                    valid_edge_start,
                    valid_edge_end,
                    node_ts_sorted_indices,
                    edge_selector_rand_num);
                if (edge_idx == -1) return InternalEdge{-1, -1, -1, -1};
            }
        } else {
            edge_idx = static_cast<long>(node_ts_sorted_indices[
                valid_edge_start +
                generate_random_number_bounded_by(static_cast<int>(valid_edge_end - valid_edge_start), edge_selector_rand_num)]);
        }

        return InternalEdge{
            graph->edge_data->sources[edge_idx],
            graph->edge_data->targets[edge_idx],
            graph->edge_data->timestamps[edge_idx],
            edge_idx
        };
    }

    #endif
}

#endif //EDGE_SELECTORS_CUH
