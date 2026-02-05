#ifndef TEMPORAL_RANDOM_WALK_CPU_CUH
#define TEMPORAL_RANDOM_WALK_CPU_CUH

#include "../common/macros.cuh"
#include "../data/walk_set/walk_set.cuh"
#include "helpers.cuh"
#include "../stores/temporal_graph.cuh"
#include "../stores/edge_selectors.cuh"


namespace temporal_random_walk {
    template<bool IsDirected, bool Forward, RandomPickerType EdgePickerType, RandomPickerType StartPickerType>
    HOST void generate_random_walk_and_time_std(
        const TemporalGraphStore *temporal_graph,
        const int walk_idx,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const double *rand_nums) {

        const size_t rand_nums_start_idx_for_walk = static_cast<size_t>(walk_idx)
            + static_cast<size_t>(walk_idx) * static_cast<size_t>(max_walk_len) * 2;

        Edge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_host<Forward, StartPickerType>(
                temporal_graph,
                -1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]);
        } else {
            start_edge = temporal_graph::get_node_edge_at_host<Forward, StartPickerType, IsDirected>(
                temporal_graph,
                start_node_ids[walk_idx],
                -1,
                -1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]
            );
        }

        if (start_edge.i == -1) {
            return;
        }

        int current_node;
        int prev_node = -1;
        int64_t current_timestamp = Forward ? INT64_MIN : INT64_MAX;

        // Extract start edge components
        const int start_src = start_edge.u;
        const int start_dst = start_edge.i;
        const int64_t start_ts = start_edge.ts;

        // Use template parameter IsDirected instead of runtime check
        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set->add_hop(walk_idx, start_src, current_timestamp);
                current_node = start_dst;
            } else {
                walk_set->add_hop(walk_idx, start_dst, current_timestamp);
                current_node = start_src;
            }
        } else {
            // For undirected graphs, use the specified start node or pick a random one
            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst,
                                                             rand_nums[rand_nums_start_idx_for_walk + 2]);
            walk_set->add_hop(walk_idx, picked_node, current_timestamp);
            current_node = pick_other_number(start_src, start_dst, picked_node);
        }

        current_timestamp = start_ts;

        // Perform the walk
        int walk_len = 1; // Starting at 1 since we already added first hop
        while (walk_len < max_walk_len && current_node != -1) {
            const auto step_start_idx = rand_nums_start_idx_for_walk + walk_len * 2 + 1;
            const auto group_selector_rand_num = rand_nums[step_start_idx];
            const auto edge_selector_rand_num = rand_nums[step_start_idx + 1];

            walk_set->add_hop(walk_idx, current_node, current_timestamp);

            Edge next_edge = temporal_graph::get_node_edge_at_host<Forward, EdgePickerType, IsDirected
            >
            (
                temporal_graph,
                current_node,
                current_timestamp,
                prev_node,
                group_selector_rand_num,
                edge_selector_rand_num
            );

            if (next_edge.ts == -1) {
                current_node = -1;
                continue;
            }

            // Use template parameter again
            if constexpr (IsDirected) {
                prev_node = current_node;
                current_node = Forward ? next_edge.i : next_edge.u;
            } else {
                prev_node = current_node;
                current_node = pick_other_number(next_edge.u, next_edge.i, current_node);
            }

            current_timestamp = next_edge.ts;
            walk_len++;
        }

        // Reverse the walk if we walked backward
        if constexpr (!Forward) {
            walk_set->reverse_walk(walk_idx);
        }
    }

    template<bool IsDirected, bool Forward>
    HOST void dispatch_walk_generation(
        TemporalGraphStore *temporal_graph,
        const int walk_idx,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const double *rand_nums,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type) {
        switch (edge_picker_type) {
            case RandomPickerType::Uniform:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Uniform,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::Linear:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::Linear,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::ExponentialIndex:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialIndex,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::ExponentialWeight:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TemporalNode2Vec:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::TemporalNode2Vec>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::ExponentialWeight,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::TemporalNode2Vec:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TemporalNode2Vec:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::TemporalNode2Vec>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TemporalNode2Vec,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::TEST_FIRST:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_FIRST,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;

            case RandomPickerType::TEST_LAST:
                switch (start_picker_type) {
                    case RandomPickerType::Uniform:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::Uniform>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::Linear:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::Linear>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialIndex:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::ExponentialIndex>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::ExponentialWeight:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::ExponentialWeight>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_FIRST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::TEST_FIRST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                    case RandomPickerType::TEST_LAST:
                        generate_random_walk_and_time_std<IsDirected, Forward, RandomPickerType::TEST_LAST,
                            RandomPickerType::TEST_LAST>(
                            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums);
                        break;
                }
                break;
        }
    }

    HOST inline void launch_random_walk_cpu(
        TemporalGraphStore *temporal_graph,
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

        const RandomPickerType normalized_start_picker_type =
            start_picker_type == RandomPickerType::TemporalNode2Vec
                ? RandomPickerType::ExponentialWeight
                : start_picker_type;

        #pragma omp parallel for
        for (int walk_idx = 0; walk_idx < num_walks; walk_idx++) {
            if (is_directed) {
                if (should_walk_forward) {
                    dispatch_walk_generation<true, true>(
                        temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                } else {
                    dispatch_walk_generation<true, false>(
                        temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                }
            } else {
                if (should_walk_forward) {
                    dispatch_walk_generation<false, true>(
                        temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                } else {
                    dispatch_walk_generation<false, false>(
                        temporal_graph, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                }
            }
        }
    }
};

#endif //TEMPORAL_RANDOM_WALK_CPU_CUH
