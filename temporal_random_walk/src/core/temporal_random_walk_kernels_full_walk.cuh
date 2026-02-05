#ifndef TEMPORAL_RANDOM_WALK_KERNELS_FULL_WALK_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_FULL_WALK_CUH

#include "../data/walk_set/walk_set.cuh"
#include "../stores/temporal_graph.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "helpers.cuh"
#include "../stores/edge_selectors.cuh"

namespace temporal_random_walk {
    #ifdef HAS_CUDA

    template<bool IsDirected, bool Forward, RandomPickerType EdgePickerType, RandomPickerType StartPickerType>
    __global__ void generate_random_walks_kernel(
        const WalkSet *__restrict__ walk_set,
        TemporalGraphStore *__restrict__ temporal_graph,
        const int *__restrict__ start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const uint64_t base_seed) {

        const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        // Optimize rand_nums access - use direct formula instead of storing offset
        // Calculate indices directly to reduce register usage
        const size_t base_idx = static_cast<size_t>(walk_idx) * (1 + static_cast<size_t>(max_walk_len) * 2);

        const double r0 = rng_u01_philox(base_seed, walk_idx, base_idx + 0);
        const double r1 = rng_u01_philox(base_seed, walk_idx, base_idx + 1);

        // Get start edge based on whether we have a starting node
        Edge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
                temporal_graph,
                -1, // timestamp
                r0,
                r1);
        } else {
            start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
                temporal_graph,
                start_node_ids[walk_idx],
                -1, // timestamp
                -1,
                r0,
                r1);
        }

        if (start_edge.i == -1) {
            return;
        }

        int current_node;
        int64_t current_timestamp = Forward ? INT64_MIN : INT64_MAX;
        const int start_src = start_edge.u;
        const int start_dst = start_edge.i;
        const int64_t start_ts = start_edge.ts;

        // Set initial node and add first hop - use template conditions
        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set->add_hop(walk_idx, start_src, current_timestamp);
                current_node = start_dst;
            } else {
                walk_set->add_hop(walk_idx, start_dst, current_timestamp);
                current_node = start_src;
            }
        } else {
            const double r2 = rng_u01_philox(base_seed, walk_idx, base_idx + 2);

            // For undirected graphs, use specified start node or pick a random node
            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst, r2);

            walk_set->add_hop(walk_idx, picked_node, current_timestamp);
            current_node = pick_other_number(start_src, start_dst, picked_node);
        }

        current_timestamp = start_ts;

        // Main walk loop
        int walk_len = 1; // Start at 1 since we already added first hop
        while (walk_len < max_walk_len && current_node != -1) {
            // Calculate random number indices directly based on walk_len
            const size_t step_base_idx = base_idx + static_cast<size_t>(walk_len) * 2 + 1;

            const double r_step0 = rng_u01_philox(base_seed, walk_idx, step_base_idx);
            const double r_step1 = rng_u01_philox(base_seed, walk_idx, step_base_idx + 1);

            walk_set->add_hop(walk_idx, current_node, current_timestamp);

            // Use templated edge selector function
            Edge next_edge = temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
                temporal_graph,
                current_node,
                current_timestamp,
                -1,
                r_step0,
                r_step1);

            if (next_edge.ts == -1) {
                current_node = -1;
                continue;
            }

            // Update current node based on template parameters
            if constexpr (IsDirected) {
                current_node = Forward ? next_edge.i : next_edge.u;
            } else {
                current_node = pick_other_number(next_edge.u, next_edge.i, current_node);
            }

            current_timestamp = next_edge.ts;
            walk_len++;
        }

        // Reverse walk only when walking backward
        if constexpr (!Forward) {
            walk_set->reverse_walk(walk_idx);
        }
    }

    inline void launch_random_walk_kernel_full_walk(
        TemporalGraphStore *temporal_graph,
        const bool is_directed,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const uint64_t base_seed,
        const dim3 &grid_dim,
        const dim3 &block_dim) {
        // Calculate grid dimensions if not provided
        dim3 grid = grid_dim;
        if (grid.x == 0) {
            grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
        }

        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        // Convert to int since the kernel accepts int, not size_t
        const int num_walks_int = static_cast<int>(num_walks);

        #define DISPATCH(DIR, FWD, EDGE, START) \
            generate_random_walks_kernel<DIR, FWD, EDGE, START><<<grid, block_dim>>>( \
                walk_set, temporal_graph, start_node_ids, max_walk_len, num_walks_int, base_seed); return;

        #define HANDLE_EDGE_START(DIR, FWD) \
            switch (edge_picker_type) { \
                case RandomPickerType::Uniform: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::Uniform, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                case RandomPickerType::Linear: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::Linear, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                case RandomPickerType::ExponentialIndex: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::ExponentialIndex, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                case RandomPickerType::ExponentialWeight: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::ExponentialWeight, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                case RandomPickerType::TEST_FIRST: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::TEST_FIRST, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                case RandomPickerType::TEST_LAST: \
                    switch (start_picker_type) { \
                        case RandomPickerType::Uniform: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::Uniform); \
                        case RandomPickerType::Linear: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::Linear); \
                        case RandomPickerType::ExponentialIndex: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::ExponentialIndex); \
                        case RandomPickerType::ExponentialWeight: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::ExponentialWeight); \
                        case RandomPickerType::TEST_FIRST: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::TEST_FIRST); \
                        case RandomPickerType::TEST_LAST: DISPATCH(DIR, FWD, RandomPickerType::TEST_LAST, RandomPickerType::TEST_LAST); \
                        default: break; \
                    } break; \
                default: break; \
            }

        if (is_directed) {
            if (should_walk_forward) {
                HANDLE_EDGE_START(true, true)
            } else {
                HANDLE_EDGE_START(true, false)
            }
        }
        else {
            if (should_walk_forward) {
                HANDLE_EDGE_START(false, true)
            } else {
                HANDLE_EDGE_START(false, false)
            }
        }

        #undef DISPATCH
        #undef HANDLE_EDGE_START
    }

    #endif

}

#endif //TEMPORAL_RANDOM_WALK_KERNELS_FULL_WALK_CUH
