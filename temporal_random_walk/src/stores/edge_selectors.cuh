#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include "temporal_graph.cuh"

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
    HOST Edge get_edge_at_host(
        const TemporalGraphStore *graph,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num) {
        if (edge_data::empty(graph->edge_data)) return Edge{-1, -1, -1};

        const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
        if (num_groups == 0) return Edge{-1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = edge_data::find_group_after_timestamp(graph->edge_data, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return Edge{-1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available_groups) return Edge{-1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            } else {
                const size_t last_group = edge_data::find_group_before_timestamp(graph->edge_data, timestamp);
                if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available_groups) return Edge{-1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= num_groups) return Edge{-1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            }
        }

        // Get selected group's boundaries
        const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
        if (group_range.from == group_range.to) {
            return Edge{-1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return Edge{
            graph->edge_data->sources[random_idx],
            graph->edge_data->targets[random_idx],
            graph->edge_data->timestamps[random_idx]
        };
    }


    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    HOST Edge get_node_edge_at_host(
        const TemporalGraphStore *graph,
        int node_id,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num) {
        if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
            return Edge{-1, -1, -1};
        }

        // Get appropriate node indices based on direction and graph type
        const size_t *timestamp_group_offsets = Forward
                                                    ? graph->node_edge_index->outbound_timestamp_group_offsets
                                                    : (IsDirected
                                                           ? graph->node_edge_index->inbound_timestamp_group_offsets
                                                           : graph->node_edge_index->outbound_timestamp_group_offsets);

        size_t *timestamp_group_indices = Forward
                                              ? graph->node_edge_index->outbound_timestamp_group_indices
                                              : (IsDirected
                                                     ? graph->node_edge_index->inbound_timestamp_group_indices
                                                     : graph->node_edge_index->outbound_timestamp_group_indices);

        size_t *edge_indices = Forward
                                   ? graph->node_edge_index->outbound_indices
                                   : (IsDirected
                                          ? graph->node_edge_index->inbound_indices
                                          : graph->node_edge_index->outbound_indices);

        // Get node's group range
        const size_t group_start_offset = timestamp_group_offsets[node_id];
        const size_t group_end_offset = timestamp_group_offsets[node_id + 1];
        if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

        long group_pos;
        if (timestamp != -1) {
            if constexpr (Forward) {
                // Find first group after timestamp
                const auto it = std::upper_bound(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    timestamp_group_indices + static_cast<int>(group_end_offset),
                    timestamp,
                    [graph, edge_indices](const int64_t ts, const size_t pos) {
                        return ts < graph->edge_data->timestamps[edge_indices[pos]];
                    });

                // Count available groups after timestamp
                const size_t available = std::distance(
                    it,
                    timestamp_group_indices + static_cast<int>(group_end_offset));
                if (available == 0) return Edge{-1, -1, -1};

                const size_t start_pos = it - timestamp_group_indices;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available), false, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available) return Edge{-1, -1, -1};
                    group_pos = static_cast<long>(start_pos) + index;
                } else {
                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                        start_pos, group_end_offset, group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            } else {
                // Find first group >= timestamp
                auto it = std::lower_bound(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    timestamp_group_indices + static_cast<int>(group_end_offset),
                    timestamp,
                    [graph, edge_indices](const size_t pos, const int64_t ts) {
                        return graph->edge_data->timestamps[edge_indices[pos]] < ts;
                    });

                const size_t available = std::distance(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    it);
                if (available == 0) return Edge{-1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available),
                        true, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available) return Edge{-1, -1, -1};
                    group_pos = static_cast<long>((it - timestamp_group_indices) - 1 - (available - index - 1));
                } else {
                    double *weights = IsDirected
                                          ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                          : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                    const size_t weights_size = IsDirected
                                                    ? graph->node_edge_index->
                                                    inbound_backward_cumulative_weights_exponential_size
                                                    : graph->node_edge_index->
                                                    outbound_backward_cumulative_weights_exponential_size;

                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        weights,
                        weights_size,
                        group_start_offset,
                        static_cast<size_t>(it - timestamp_group_indices),
                        group_selector_rand_num
                    );
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            const size_t num_groups = group_end_offset - group_start_offset;
            if (num_groups == 0) return Edge{-1, -1, -1};

            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= num_groups) return Edge{-1, -1, -1};
                group_pos = Forward
                                ? static_cast<long>(group_start_offset + index)
                                : static_cast<long>(group_end_offset - 1 - (num_groups - index - 1));
            } else {
                if constexpr (Forward) {
                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                        group_start_offset,
                        group_end_offset,
                        group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                } else {
                    double *weights = IsDirected
                                          ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                          : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                    const size_t weights_size = IsDirected
                                                    ? graph->node_edge_index->
                                                    inbound_backward_cumulative_weights_exponential_size
                                                    : graph->node_edge_index->
                                                    outbound_backward_cumulative_weights_exponential_size;

                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        weights,
                        weights_size,
                        group_start_offset,
                        group_end_offset,
                        group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            }
        }

        // Get edge range for selected group
        const size_t edge_start = timestamp_group_indices[group_pos];
        size_t edge_end;

        if (group_pos + 1 < group_end_offset) {
            edge_end = timestamp_group_indices[group_pos + 1];
        } else {
            if constexpr (Forward) {
                edge_end = graph->node_edge_index->outbound_offsets[node_id + 1];
            } else {
                edge_end = IsDirected
                               ? graph->node_edge_index->inbound_offsets[node_id + 1]
                               : graph->node_edge_index->outbound_offsets[node_id + 1];
            }
        }

        // Validate range before random selection
        size_t const edge_indices_size = Forward
                                             ? graph->node_edge_index->outbound_indices_size
                                             : (IsDirected
                                                    ? graph->node_edge_index->inbound_indices_size
                                                    : graph->node_edge_index->outbound_indices_size);

        if (edge_start >= edge_end || edge_start >= edge_indices_size || edge_end > edge_indices_size) {
            return Edge{-1, -1, -1};
        }

        // Random selection from group
        const size_t edge_idx = edge_indices[
            edge_start +
            generate_random_number_bounded_by(static_cast<int>(edge_end - edge_start), edge_selector_rand_num)];

        return Edge{
            graph->edge_data->sources[edge_idx],
            graph->edge_data->targets[edge_idx],
            graph->edge_data->timestamps[edge_idx]
        };
    }

    /**
     * Device functions
     */

    #ifdef HAS_CUDA

    template<bool Forward, RandomPickerType PickerType>
    DEVICE Edge get_edge_at_device(
        const TemporalGraphStore *graph,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num) {
        if (edge_data::empty(graph->edge_data)) return Edge{-1, -1, -1};

        const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
        if (num_groups == 0) return Edge{-1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = edge_data::find_group_after_timestamp_device(graph->edge_data, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return Edge{-1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available_groups) return Edge{-1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            } else {
                const size_t last_group = edge_data::find_group_before_timestamp_device(graph->edge_data, timestamp);
                if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available_groups) return Edge{-1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= num_groups) return Edge{-1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->forward_cumulative_weights_exponential,
                        graph->edge_data->forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->edge_data->backward_cumulative_weights_exponential,
                        graph->edge_data->backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return Edge{-1, -1, -1};
                }
            }
        }

        // Get selected group's boundaries
        const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
        if (group_range.from == group_range.to) {
            return Edge{-1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return Edge{
            graph->edge_data->sources[random_idx],
            graph->edge_data->targets[random_idx],
            graph->edge_data->timestamps[random_idx]
        };
    }


    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    DEVICE Edge get_node_edge_at_device(
        const TemporalGraphStore *graph,
        int node_id,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num) {
        if (!edge_data::is_node_active_device(graph->edge_data, node_id)) {
            return Edge{-1, -1, -1};
        }

        // Get appropriate node indices based on direction and graph type
        const size_t *timestamp_group_offsets = Forward
                                                    ? graph->node_edge_index->outbound_timestamp_group_offsets
                                                    : (IsDirected
                                                           ? graph->node_edge_index->inbound_timestamp_group_offsets
                                                           : graph->node_edge_index->outbound_timestamp_group_offsets);

        size_t *timestamp_group_indices = Forward
                                              ? graph->node_edge_index->outbound_timestamp_group_indices
                                              : (IsDirected
                                                     ? graph->node_edge_index->inbound_timestamp_group_indices
                                                     : graph->node_edge_index->outbound_timestamp_group_indices);

        size_t *edge_indices = Forward
                                   ? graph->node_edge_index->outbound_indices
                                   : (IsDirected
                                          ? graph->node_edge_index->inbound_indices
                                          : graph->node_edge_index->outbound_indices);

        // Get node's group range
        const size_t group_start_offset = timestamp_group_offsets[node_id];
        const size_t group_end_offset = timestamp_group_offsets[node_id + 1];
        if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

        long group_pos;
        if (timestamp != -1) {
            if constexpr (Forward) {
                // Find first group after timestamp
                const auto it = cuda::std::upper_bound(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    timestamp_group_indices + static_cast<int>(group_end_offset),
                    timestamp,
                    [graph, edge_indices](const int64_t ts, const size_t pos) {
                        return ts < graph->edge_data->timestamps[edge_indices[pos]];
                    });

                // Count available groups after timestamp
                const size_t available = std::distance(
                    it,
                    timestamp_group_indices + static_cast<int>(group_end_offset));
                if (available == 0) return Edge{-1, -1, -1};

                const size_t start_pos = it - timestamp_group_indices;
                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available), false, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available) return Edge{-1, -1, -1};
                    group_pos = static_cast<long>(start_pos) + index;
                } else {
                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                        start_pos, group_end_offset, group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            } else {
                // Find first group >= timestamp
                auto it = cuda::std::lower_bound(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    timestamp_group_indices + static_cast<int>(group_end_offset),
                    timestamp,
                    [graph, edge_indices](const size_t pos, const int64_t ts) {
                        return graph->edge_data->timestamps[edge_indices[pos]] < ts;
                    });

                const size_t available = std::distance(
                    timestamp_group_indices + static_cast<int>(group_start_offset),
                    it);
                if (available == 0) return Edge{-1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available),
                        true, group_selector_rand_num);
                    if (index == -1) return Edge{-1, -1, -1};

                    if (index >= available) return Edge{-1, -1, -1};
                    group_pos = static_cast<long>((it - timestamp_group_indices) - 1 - (available - index - 1));
                } else {
                    double *weights = IsDirected
                                          ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                          : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                    size_t weights_size = IsDirected
                                              ? graph->node_edge_index->
                                              inbound_backward_cumulative_weights_exponential_size
                                              : graph->node_edge_index->
                                              outbound_backward_cumulative_weights_exponential_size;

                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        weights,
                        weights_size,
                        group_start_offset,
                        static_cast<size_t>(it - timestamp_group_indices),
                        group_selector_rand_num
                    );
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            const size_t num_groups = group_end_offset - group_start_offset;
            if (num_groups == 0) return Edge{-1, -1, -1};

            if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= num_groups) return Edge{-1, -1, -1};
                group_pos = Forward
                                ? static_cast<long>(group_start_offset + index)
                                : static_cast<long>(group_end_offset - 1 - (num_groups - index - 1));
            } else {
                if constexpr (Forward) {
                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                        graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                        group_start_offset,
                        group_end_offset,
                        group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                } else {
                    double *weights = IsDirected
                                          ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                                          : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                    size_t weights_size = IsDirected
                                              ? graph->node_edge_index->
                                              inbound_backward_cumulative_weights_exponential_size
                                              : graph->node_edge_index->
                                              outbound_backward_cumulative_weights_exponential_size;

                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        weights,
                        weights_size,
                        group_start_offset,
                        group_end_offset,
                        group_selector_rand_num);
                    if (group_pos == -1) return Edge{-1, -1, -1};
                }
            }
        }

        // Get edge range for selected group
        const size_t edge_start = timestamp_group_indices[group_pos];
        size_t edge_end;

        if (group_pos + 1 < group_end_offset) {
            edge_end = timestamp_group_indices[group_pos + 1];
        } else {
            if constexpr (Forward) {
                edge_end = graph->node_edge_index->outbound_offsets[node_id + 1];
            } else {
                edge_end = IsDirected
                               ? graph->node_edge_index->inbound_offsets[node_id + 1]
                               : graph->node_edge_index->outbound_offsets[node_id + 1];
            }
        }

        // Validate range before random selection
        const size_t edge_indices_size = Forward
                                             ? graph->node_edge_index->outbound_indices_size
                                             : (IsDirected
                                                    ? graph->node_edge_index->inbound_indices_size
                                                    : graph->node_edge_index->outbound_indices_size);

        if (edge_start >= edge_end || edge_start >= edge_indices_size || edge_end > edge_indices_size) {
            return Edge{-1, -1, -1};
        }

        // Random selection from group
        const size_t edge_idx = edge_indices[
            edge_start +
            generate_random_number_bounded_by(static_cast<int>(edge_end - edge_start), edge_selector_rand_num)];

        return Edge{
            graph->edge_data->sources[edge_idx],
            graph->edge_data->targets[edge_idx],
            graph->edge_data->timestamps[edge_idx]
        };
    }

    #endif
}

#endif //EDGE_SELECTORS_CUH
