#ifndef TEMPORAL_RANDOM_WALK_CPU_CUH
#define TEMPORAL_RANDOM_WALK_CPU_CUH

#include "../data/walk_set/walk_set.cuh"
#include "helpers.cuh"
#include "../stores/temporal_graph.cuh"
#include "../stores/edge_selectors.cuh"


namespace temporal_random_walk {

    // Core functions for processing individual walk steps
    template <bool IsDirected, bool Forward, RandomPickerType StartPickerType>
    void pick_start_edges_cpu(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int* start_node_ids,
        const int max_walk_len,
        const size_t walk_idx,
        const double *rand_nums) {

        const int rand_nums_start_offset =
            walk_idx +                         // To account extra value in all previous walk's start pickers.
            (walk_idx * (max_walk_len - 1) * 2);     // To account all 2 rand numbers for all other steps in the previous walks.

        Edge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_host<Forward, StartPickerType>(
                temporal_graph,
                -1, // timestamp
                rand_nums[rand_nums_start_offset],
                rand_nums[rand_nums_start_offset + 1]);
        } else {
            start_edge = temporal_graph::get_node_edge_at_host<Forward, StartPickerType, IsDirected>(
                temporal_graph,
                start_node_ids[walk_idx],
                -1, // timestamp
                rand_nums[rand_nums_start_offset],
                rand_nums[rand_nums_start_offset + 1]);
        }

        if (start_edge.i == -1) {
            return;
        }

        const int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
        const int start_src = start_edge.u;
        const int start_dst = start_edge.i;
        const int64_t start_ts = start_edge.ts;

        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set->add_hop(walk_idx, start_src, sentinel_timestamp);
                walk_set->add_hop(walk_idx, start_dst, start_ts);
            } else {
                walk_set->add_hop(walk_idx, start_dst, sentinel_timestamp);
                walk_set->add_hop(walk_idx, start_src, start_ts);
            }
        } else {
            // For undirected graphs, use specified start node or pick a random node
            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst, rand_nums[rand_nums_start_offset + 2]);
            const int other_node = pick_other_number(start_src, start_dst, picked_node);

            walk_set->add_hop(walk_idx, picked_node, sentinel_timestamp);
            walk_set->add_hop(walk_idx, other_node, start_ts);
        }
    }

    template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
    void pick_intermediate_edges_cpu(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int step_number,
        const int max_walk_len,
        const size_t walk_idx,
        const double *rand_nums) {

        if (step_number >= max_walk_len - 1) {
            return;
        }

        const size_t offset = walk_idx * max_walk_len + step_number; // Get endpoint of previous step (step_number - 1). And endpoint is (step_number - 1 + 1).
        const int last_node = walk_set->nodes[offset];
        const int last_ts = walk_set->timestamps[offset];

        const int rand_nums_start_offset =
            walk_idx +                              // To account extra value in all previous walk's start pickers.
            (walk_idx * (max_walk_len - 1) * 2) +         // To account all 2 rand numbers for all other steps in the previous walks.
            (step_number * 2 + 1);                  // To account for random numbers used in the current walk.

        const Edge next_edge = temporal_graph::get_node_edge_at_host<Forward, EdgePickerType, IsDirected>(
                temporal_graph,
                last_node,
                last_ts,
                rand_nums[rand_nums_start_offset],
                rand_nums[rand_nums_start_offset + 1]);

        if (next_edge.ts == -1) {
            return;
        }

        if constexpr (IsDirected) {
            walk_set->add_hop(walk_idx, Forward ? next_edge.i : next_edge.u, next_edge.ts);
        } else {
            const auto node_to_add = pick_other_number(next_edge.u, next_edge.i, last_node);
            walk_set->add_hop(walk_idx, node_to_add, next_edge.ts);
        }
    }

    inline void reverse_walks_cpu(const WalkSet *walk_set, const size_t num_walks) {
        #pragma omp parallel for
        for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
            walk_set->reverse_walk(walk_idx);
        }
    }

    // Helper function to dispatch start edge processing on CPU
    template <bool IsDirected, bool Forward>
    void dispatch_start_edges_cpu(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int *start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType start_picker_type,
        const double *rand_nums) {

        switch (start_picker_type) {
            case RandomPickerType::Uniform:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::Uniform>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            case RandomPickerType::Linear:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::Linear>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            case RandomPickerType::ExponentialIndex:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::ExponentialIndex>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            case RandomPickerType::ExponentialWeight:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::ExponentialWeight>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            case RandomPickerType::TEST_FIRST:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::TEST_FIRST>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            case RandomPickerType::TEST_LAST:
                #pragma omp parallel for
                for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
                    pick_start_edges_cpu<IsDirected, Forward, RandomPickerType::TEST_LAST>(
                        temporal_graph, walk_set, start_node_ids, max_walk_len, walk_idx, rand_nums);
                }
                break;
            default:
                break;
        }
    }

    // Helper function to dispatch intermediate edge kernels on CPU - optimized version
    template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
    void dispatch_intermediate_edges_cpu(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int step_number,
        const int max_walk_len,
        const size_t num_walks,
        const double *rand_nums) {

        #pragma omp parallel for
        for (size_t walk_idx = 0; walk_idx < num_walks; ++walk_idx) {
            // Call with a single template parameter for EdgePickerType
            pick_intermediate_edges_cpu<IsDirected, Forward, EdgePickerType>(
                temporal_graph, walk_set, step_number, max_walk_len, walk_idx, rand_nums);
        }
    }

    // Helper function to handle intermediate steps with different edge picker types on CPU
    template <bool IsDirected, bool Forward>
    void handle_intermediate_steps_cpu(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const double *rand_nums) {

        for (int step_number = 1; step_number < max_walk_len; step_number++) {
            switch (edge_picker_type) {
                case RandomPickerType::Uniform:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::Uniform>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                case RandomPickerType::Linear:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::Linear>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                case RandomPickerType::ExponentialIndex:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::ExponentialIndex>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                case RandomPickerType::ExponentialWeight:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::ExponentialWeight>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                case RandomPickerType::TEST_FIRST:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::TEST_FIRST>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                case RandomPickerType::TEST_LAST:
                    dispatch_intermediate_edges_cpu<IsDirected, Forward, RandomPickerType::TEST_LAST>(
                        temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                    break;
                default:
                    break;
            }
        }
    }

    // Main CPU launcher function mirroring the GPU launcher structure
    inline void launch_random_walk_cpu(
        const TemporalGraphStore *temporal_graph,
        const bool is_directed,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const double *rand_nums) {

        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        // Launch pick_start_edges_cpu
        if (is_directed) {
            if (should_walk_forward) {
                dispatch_start_edges_cpu<true, true>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks,
                    start_picker_type, rand_nums);
            } else {
                dispatch_start_edges_cpu<true, false>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks,
                    start_picker_type, rand_nums);
            }
        } else {
            if (should_walk_forward) {
                dispatch_start_edges_cpu<false, true>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks,
                    start_picker_type, rand_nums);
            } else {
                dispatch_start_edges_cpu<false, false>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks,
                    start_picker_type, rand_nums);
            }
        }

        // Launch intermediate edge processing for each step
        if (is_directed) {
            if (should_walk_forward) {
                handle_intermediate_steps_cpu<true, true>(
                    temporal_graph, walk_set, max_walk_len, num_walks, edge_picker_type, rand_nums);
            } else {
                handle_intermediate_steps_cpu<true, false>(
                    temporal_graph, walk_set, max_walk_len, num_walks, edge_picker_type, rand_nums);
            }
        } else {
            if (should_walk_forward) {
                handle_intermediate_steps_cpu<false, true>(
                    temporal_graph, walk_set, max_walk_len, num_walks, edge_picker_type, rand_nums);
            } else {
                handle_intermediate_steps_cpu<false, false>(
                    temporal_graph, walk_set, max_walk_len, num_walks, edge_picker_type, rand_nums);
            }
        }

        // Reverse walks if walking backward
        if (!should_walk_forward) {
            reverse_walks_cpu(walk_set, num_walks);
        }
    }
}

#endif //TEMPORAL_RANDOM_WALK_CPU_CUH
