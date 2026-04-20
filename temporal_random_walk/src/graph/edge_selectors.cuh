#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include <algorithm>
#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../random/pickers.cuh"
#include "../utils/random.cuh"
#include "temporal_node2vec_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

namespace temporal_graph {

    // -------------------------------------------------------------------------
    // Inline view-based helpers (host variants).
    // Mirror edge_data::find_group_after_timestamp,
    // edge_data::find_group_before_timestamp, edge_data::get_timestamp_group_range,
    // edge_data::is_node_active — but sourced from TemporalGraphView.
    // -------------------------------------------------------------------------
    HOST inline size_t find_group_after_timestamp_host(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    HOST inline size_t find_group_before_timestamp_host(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    HOST inline SizeRange get_timestamp_group_range_host(
        const TemporalGraphView& view, const size_t group_idx) {
        if (group_idx >= view.num_groups) return SizeRange{0, 0};
        return SizeRange{view.timestamp_group_offsets[group_idx],
                         view.timestamp_group_offsets[group_idx + 1]};
    }

    HOST inline bool is_node_active_host(
        const TemporalGraphView& view, const int node_id) {
        return node_id >= 0
            && static_cast<size_t>(node_id) < view.active_node_ids_size
            && view.active_node_ids[node_id] == 1;
    }

    /**
     * Host functions
     */

    template<bool Forward, RandomPickerType PickerType>
    HOST InternalEdge get_edge_at_host(
        const TemporalGraphView& view,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (view.num_edges == 0) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = view.num_groups;
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = find_group_after_timestamp_host(view, timestamp);
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
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = find_group_before_timestamp_host(view, timestamp);
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
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
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
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_host(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        // Get selected group's boundaries
        const SizeRange group_range = get_timestamp_group_range_host(view, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            view.sources[random_idx],
            view.targets[random_idx],
            view.timestamps[random_idx],
            static_cast<int64_t>(random_idx)
        };
    }

    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    HOST InternalEdge get_node_edge_at_host(
        const TemporalGraphView& view,
        const int node_id,
        const int64_t timestamp,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (!is_node_active_host(view, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Direction-dependent index structures
        const size_t* count_ts_group_per_node =
                Forward
                    ? view.count_ts_group_per_node_outbound
                    : (IsDirected
                           ? view.count_ts_group_per_node_inbound
                           : view.count_ts_group_per_node_outbound);

        const size_t* node_ts_groups_offsets =
                Forward
                    ? view.node_ts_group_outbound_offsets
                    : (IsDirected
                           ? view.node_ts_group_inbound_offsets
                           : view.node_ts_group_outbound_offsets);

        const size_t* node_ts_sorted_indices =
                Forward
                    ? view.node_ts_sorted_outbound_indices
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices
                           : view.node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size =
                Forward
                    ? view.node_ts_sorted_outbound_indices_size
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices_size
                           : view.node_ts_sorted_outbound_indices_size);

        const size_t* node_edge_offsets =
                Forward
                    ? view.node_group_outbound_offsets
                    : (IsDirected
                           ? view.node_group_inbound_offsets
                           : view.node_group_outbound_offsets);

        const double* weights =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential
                           : view.outbound_backward_cumulative_weights_exponential);

        const size_t weights_size =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential_size
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential_size
                           : view.outbound_backward_cumulative_weights_exponential_size);

        // Node's full timestamp-group range
        const size_t node_group_begin = count_ts_group_per_node[node_id];
        const size_t node_group_end = count_ts_group_per_node[node_id + 1];

        if (node_group_begin == node_group_end) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // Compute valid timestamp-filtered group slice
        size_t valid_begin = node_group_begin;
        size_t valid_end = node_group_end;

        if (timestamp != -1) {
            const int64_t* view_timestamps = view.timestamps;
            if constexpr (Forward) {
                const auto it = std::upper_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const int64_t ts, const size_t pos) {
                        return ts < view_timestamps[node_ts_sorted_indices[pos]];
                    });

                valid_begin = static_cast<size_t>(it - node_ts_groups_offsets);
                valid_end = node_group_end;
            } else {
                const auto it = std::lower_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                        return view_timestamps[node_ts_sorted_indices[pos]] < ts;
                    });

                valid_begin = node_group_begin;
                valid_end = static_cast<size_t>(it - node_ts_groups_offsets);
            }

            if (valid_begin >= valid_end) {
                return InternalEdge{-1, -1, -1, -1};
            }
        }

        // Group-based pickers
        long group_pos = -1;

        if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
            const size_t available = valid_end - valid_begin;

            const auto index = random_pickers::pick_using_index_based_picker(
                PickerType,
                0,
                static_cast<int>(available),
                !Forward,
                group_selector_rand_num);

            if (index == -1 || index >= available) {
                return InternalEdge{-1, -1, -1, -1};
            }

            group_pos = Forward
                            ? static_cast<long>(valid_begin + index)
                            : static_cast<long>(valid_end - 1 - (available - index - 1));
        } else {
            if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                if (prev_node == -1) {
                    group_pos = random_pickers::pick_using_weight_based_picker_host(
                        RandomPickerType::ExponentialWeight,
                        weights,
                        weights_size,
                        valid_begin,
                        valid_end,
                        group_selector_rand_num);
                } else {
                    group_pos = pick_random_temporal_node2vec_host<Forward, IsDirected>(
                        view,
                        node_id,
                        prev_node,
                        valid_begin,
                        valid_end,
                        node_group_begin,
                        node_group_end,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices,
                        weights,
                        group_selector_rand_num);
                }
            } else {
                group_pos = random_pickers::pick_using_weight_based_picker_host(
                    PickerType,
                    weights,
                    weights_size,
                    valid_begin,
                    valid_end,
                    group_selector_rand_num);
            }

            if (group_pos == -1) {
                return InternalEdge{-1, -1, -1, -1};
            }
        }

        // Resolve selected group to edge range
        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        const size_t valid_edge_end =
                (static_cast<size_t>(group_pos) + 1 < node_group_end)
                    ? node_ts_groups_offsets[group_pos + 1]
                    : node_edge_offsets[node_id + 1];

        if (valid_edge_start >= valid_edge_end ||
            valid_edge_start >= node_ts_sorted_indices_size ||
            valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        long edge_idx = -1;

        if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
            if (prev_node == -1) {
                edge_idx = static_cast<long>(node_ts_sorted_indices[
                    valid_edge_start +
                    generate_random_number_bounded_by(
                        static_cast<int>(valid_edge_end - valid_edge_start),
                        edge_selector_rand_num)]);
            } else {
                edge_idx = pick_random_temporal_node2vec_edge_host<Forward, IsDirected>(
                    view,
                    node_id,
                    prev_node,
                    valid_edge_start,
                    valid_edge_end,
                    node_ts_sorted_indices,
                    edge_selector_rand_num);

                if (edge_idx == -1) {
                    return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            edge_idx = static_cast<long>(node_ts_sorted_indices[
                valid_edge_start +
                generate_random_number_bounded_by(
                    static_cast<int>(valid_edge_end - valid_edge_start),
                    edge_selector_rand_num)]);
        }

        return InternalEdge{
            view.sources[edge_idx],
            view.targets[edge_idx],
            view.timestamps[edge_idx],
            edge_idx
        };
    }

    /**
     * Device functions
     */

    #ifdef HAS_CUDA

    // -------------------------------------------------------------------------
    // Inline view-based helpers (device-only).
    // -------------------------------------------------------------------------
    DEVICE inline size_t find_group_after_timestamp(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = cuda::std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    DEVICE inline size_t find_group_before_timestamp(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = cuda::std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    DEVICE inline SizeRange get_timestamp_group_range(
        const TemporalGraphView& view, size_t group_idx) {
        if (group_idx >= view.num_groups) return SizeRange{0, 0};
        return SizeRange{view.timestamp_group_offsets[group_idx],
                         view.timestamp_group_offsets[group_idx + 1]};
    }

    DEVICE inline bool is_node_active(const TemporalGraphView& view,
                                       const int node_id) {
        return node_id >= 0
            && static_cast<size_t>(node_id) < view.active_node_ids_size
            && view.active_node_ids[node_id] == 1;
    }

    template<bool Forward, RandomPickerType PickerType>
    DEVICE InternalEdge get_edge_at_device(
        const TemporalGraphView& view,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (view.num_edges == 0) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = view.num_groups;
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = find_group_after_timestamp(view, timestamp);
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
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = find_group_before_timestamp(view, timestamp);
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
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
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
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker_device(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        const SizeRange group_range = get_timestamp_group_range(view, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            view.sources[random_idx],
            view.targets[random_idx],
            view.timestamps[random_idx],
            static_cast<int64_t>(random_idx),
        };
    }

    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    DEVICE InternalEdge get_node_edge_at_device(
        const TemporalGraphView& view,
        const int node_id,
        const int64_t timestamp,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (!is_node_active(view, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t* count_ts_group_per_node =
                Forward
                    ? view.count_ts_group_per_node_outbound
                    : (IsDirected
                           ? view.count_ts_group_per_node_inbound
                           : view.count_ts_group_per_node_outbound);

        const size_t* node_ts_groups_offsets =
                Forward
                    ? view.node_ts_group_outbound_offsets
                    : (IsDirected
                           ? view.node_ts_group_inbound_offsets
                           : view.node_ts_group_outbound_offsets);

        const size_t* node_ts_sorted_indices =
                Forward
                    ? view.node_ts_sorted_outbound_indices
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices
                           : view.node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size =
                Forward
                    ? view.node_ts_sorted_outbound_indices_size
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices_size
                           : view.node_ts_sorted_outbound_indices_size);

        const size_t* node_edge_offsets =
                Forward
                    ? view.node_group_outbound_offsets
                    : (IsDirected
                           ? view.node_group_inbound_offsets
                           : view.node_group_outbound_offsets);

        const double* weights =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential
                           : view.outbound_backward_cumulative_weights_exponential);

        const size_t weights_size =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential_size
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential_size
                           : view.outbound_backward_cumulative_weights_exponential_size);

        const size_t node_group_begin = count_ts_group_per_node[node_id];
        const size_t node_group_end = count_ts_group_per_node[node_id + 1];

        if (node_group_begin == node_group_end) {
            return InternalEdge{-1, -1, -1, -1};
        }

        size_t valid_begin = node_group_begin;
        size_t valid_end = node_group_end;

        if (timestamp != -1) {
            const int64_t* view_timestamps = view.timestamps;
            if constexpr (Forward) {
                const auto it = cuda::std::upper_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const int64_t ts, const size_t pos) {
                        return ts < view_timestamps[node_ts_sorted_indices[pos]];
                    });

                valid_begin = static_cast<size_t>(it - node_ts_groups_offsets);
                valid_end = node_group_end;
            } else {
                const auto it = cuda::std::lower_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                        return view_timestamps[node_ts_sorted_indices[pos]] < ts;
                    });

                valid_begin = node_group_begin;
                valid_end = static_cast<size_t>(it - node_ts_groups_offsets);
            }

            if (valid_begin >= valid_end) {
                return InternalEdge{-1, -1, -1, -1};
            }
        }

        long group_pos = -1;

        if constexpr (random_pickers::is_index_based_picker_v<PickerType>) {
            const size_t available = valid_end - valid_begin;

            const auto index = random_pickers::pick_using_index_based_picker(
                PickerType,
                0,
                static_cast<int>(available),
                !Forward,
                group_selector_rand_num);

            if (index == -1 || index >= available) {
                return InternalEdge{-1, -1, -1, -1};
            }

            group_pos = Forward
                            ? static_cast<long>(valid_begin + index)
                            : static_cast<long>(valid_end - 1 - (available - index - 1));
        } else {
            if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                if (prev_node == -1) {
                    group_pos = random_pickers::pick_using_weight_based_picker_device(
                        RandomPickerType::ExponentialWeight,
                        weights,
                        weights_size,
                        valid_begin,
                        valid_end,
                        group_selector_rand_num);
                } else {
                    group_pos = pick_random_temporal_node2vec_device<Forward, IsDirected>(
                        view,
                        node_id,
                        prev_node,
                        valid_begin,
                        valid_end,
                        node_group_begin,
                        node_group_end,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices,
                        weights,
                        group_selector_rand_num);
                }
            } else {
                group_pos = random_pickers::pick_using_weight_based_picker_device(
                    PickerType,
                    weights,
                    weights_size,
                    valid_begin,
                    valid_end,
                    group_selector_rand_num);
            }

            if (group_pos == -1) {
                return InternalEdge{-1, -1, -1, -1};
            }
        }

        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        const size_t valid_edge_end =
                (static_cast<size_t>(group_pos) + 1 < node_group_end)
                    ? node_ts_groups_offsets[group_pos + 1]
                    : node_edge_offsets[node_id + 1];

        if (valid_edge_start >= valid_edge_end ||
            valid_edge_start >= node_ts_sorted_indices_size ||
            valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        long edge_idx = -1;

        if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
            if (prev_node == -1) {
                edge_idx = static_cast<long>(node_ts_sorted_indices[
                    valid_edge_start +
                    generate_random_number_bounded_by(
                        static_cast<int>(valid_edge_end - valid_edge_start),
                        edge_selector_rand_num)]);
            } else {
                edge_idx = pick_random_temporal_node2vec_edge_device<Forward, IsDirected>(
                    view,
                    node_id,
                    prev_node,
                    valid_edge_start,
                    valid_edge_end,
                    node_ts_sorted_indices,
                    edge_selector_rand_num);

                if (edge_idx == -1) {
                    return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            edge_idx = static_cast<long>(node_ts_sorted_indices[
                valid_edge_start +
                generate_random_number_bounded_by(
                    static_cast<int>(valid_edge_end - valid_edge_start),
                    edge_selector_rand_num)]);
        }

        return InternalEdge{
            view.sources[edge_idx],
            view.targets[edge_idx],
            view.timestamps[edge_idx],
            edge_idx
        };
    }

    #endif
}

#endif // EDGE_SELECTORS_CUH
