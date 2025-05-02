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
        const int start_node_id,
        const double *rand_nums) {

        const size_t rand_nums_start_idx_for_walk = walk_idx + walk_idx * max_walk_len * 2;

        Edge start_edge;
        if (start_node_id == -1) {
            start_edge = temporal_graph::get_edge_at_host < Forward, StartPickerType > (
                temporal_graph,
                -1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]);
        } else {
            start_edge = temporal_graph::get_node_edge_at_host < Forward, StartPickerType, IsDirected > (
                temporal_graph,
                start_node_id,
                -1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]
            );
        }

        if (start_edge.i == -1) {
            return;
        }

        int current_node;
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
            const int picked_node = (start_node_id != -1)
                                        ? start_node_id
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

            Edge next_edge = temporal_graph::get_node_edge_at_host < Forward, EdgePickerType, IsDirected
            >
            (
                temporal_graph,
                current_node,
                current_timestamp,
                group_selector_rand_num,
                edge_selector_rand_num
            );

            if (next_edge.ts == -1) {
                current_node = -1;
                continue;
            }

            // Use template parameter again
            if constexpr (IsDirected) {
                current_node = Forward ? next_edge.i : next_edge.u;
            } else {
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

    HOST inline void launch_random_walk_generator(
        const TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int walk_idx,
        const int max_walk_len,
        const int start_node_id,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const double *rand_nums) {

        #define DISPATCH(DIR, FWD, EDGE, START) \
        generate_random_walk_and_time_std<DIR, FWD, RandomPickerType::EDGE, RandomPickerType::START>( \
            temporal_graph, walk_idx, walk_set, max_walk_len, start_node_id, rand_nums); return;

        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        #define HANDLE_EDGE_START(DIR, FWD) \
        switch (edge_picker_type) { \
            case RandomPickerType::Uniform: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, Uniform, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, Uniform, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, Uniform, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, Uniform, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, Uniform, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, Uniform, TEST_LAST); \
                } break; \
            case RandomPickerType::Linear: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, Linear, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, Linear, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, Linear, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, Linear, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, Linear, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, Linear, TEST_LAST); \
                } break; \
            case RandomPickerType::ExponentialIndex: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, ExponentialIndex, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, ExponentialIndex, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, ExponentialIndex, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, ExponentialIndex, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, ExponentialIndex, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, ExponentialIndex, TEST_LAST); \
                } break; \
            case RandomPickerType::ExponentialWeight: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, ExponentialWeight, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, ExponentialWeight, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, ExponentialWeight, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, ExponentialWeight, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, ExponentialWeight, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, ExponentialWeight, TEST_LAST); \
                } break; \
            case RandomPickerType::TEST_FIRST: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, TEST_FIRST, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, TEST_FIRST, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, TEST_FIRST, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, TEST_FIRST, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, TEST_FIRST, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, TEST_FIRST, TEST_LAST); \
                } break; \
            case RandomPickerType::TEST_LAST: \
                switch (start_picker_type) { \
                    case RandomPickerType::Uniform: DISPATCH(DIR, FWD, TEST_LAST, Uniform); \
                    case RandomPickerType::Linear: DISPATCH(DIR, FWD, TEST_LAST, Linear); \
                    case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, TEST_LAST, ExponentialIndex); \
                    case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, TEST_LAST, ExponentialWeight); \
                    case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, TEST_LAST, TEST_FIRST); \
                    case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, TEST_LAST, TEST_LAST); \
                } break; \
        }

        if (temporal_graph->is_directed) {
            if (should_walk_forward) {
                HANDLE_EDGE_START(true, true)
            } else {
                HANDLE_EDGE_START(true, false)
            }
        } else {
            if (should_walk_forward) {
                HANDLE_EDGE_START(false, true)
            } else {
                HANDLE_EDGE_START(false, false)
            }
        }

        #undef DISPATCH
        #undef HANDLE_EDGE_START
    }
};

#endif //TEMPORAL_RANDOM_WALK_CPU_CUH
