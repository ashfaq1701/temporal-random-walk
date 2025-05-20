#ifndef TEMPORAL_RANDOM_WALK_KERNELS_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_CUH

#include "../data/walk_set/walk_set.cuh"
#include "../stores/temporal_graph.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "helpers.cuh"
#include "../stores/edge_selectors.cuh"

namespace temporal_random_walk {
    #ifdef HAS_CUDA

    template <bool IsDirected, bool Forward, RandomPickerType StartPickerType>
    __global__ void pick_start_edges_kernel(
        TemporalGraphStore *__restrict__ temporal_graph,
        const WalkSet *__restrict__ walk_set,
        const int *__restrict__ start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const double *__restrict__ rand_nums) {
        const int walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        const int rand_nums_start_offset =
            walk_idx +                          // To account extra value in all previous walk's start pickers.
                (walk_idx * max_walk_len * 2);  // To account all 2 rand numbers for all other steps in the previous walks.

        Edge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
                temporal_graph,
                -1, // timestamp
                rand_nums[rand_nums_start_offset],
                rand_nums[rand_nums_start_offset + 1]);
        } else {
            start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
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
    __global__ void pick_intermediate_edges_kernel(
        TemporalGraphStore *__restrict__ temporal_graph,
        const WalkSet *__restrict__ walk_set,
        const int step_number,
        const int max_walk_len,
        const size_t num_walks,
        const double *__restrict__ rand_nums) {

        const int walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        if (step_number >= max_walk_len - 1) {
            return;
        }

        const size_t offset = walk_idx * max_walk_len + step_number; // Get endpoint of previous step (step_number - 1). And endpoint is (step_number - 1 + 1).
        const int last_node = walk_set->nodes[offset];
        const int last_ts = walk_set->timestamps[offset];

        const int rand_nums_start_offset =
            walk_idx +                              // To account extra value in all previous walk's start pickers.
                (walk_idx * max_walk_len * 2) +     // To account all 2 rand numbers for all other steps in the previous walks.
                    (step_number * 2 + 1);          // To account for random numbers used in the current walk.

        const Edge next_edge = temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
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

    __global__ static void reverse_walks_kernel(const WalkSet *__restrict__ walk_set, const size_t num_walks) {
        const int walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;
        walk_set->reverse_walk(walk_idx);
    }

    // Helper function to dispatch start edge kernels
    template <bool IsDirected, bool Forward>
    void dispatch_start_edges_kernel(
        TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int *start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType start_picker_type,
        const double *rand_nums,
        const dim3 &grid,
        const dim3 &block_dim) {

        switch (start_picker_type) {
            case RandomPickerType::Uniform:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::Uniform><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::Linear:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::Linear><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::ExponentialIndex:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialIndex><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::ExponentialWeight:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialWeight><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::TEST_FIRST:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_FIRST><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::TEST_LAST:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_LAST><<<grid, block_dim>>>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks, rand_nums);
                break;
            default:
                break;
        }
    }

    // Helper function to dispatch intermediate edge kernels
    template <bool IsDirected, bool Forward>
    void dispatch_intermediate_edges_kernel(
        TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int step_number,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const double *rand_nums,
        const dim3 &grid,
        const dim3 &block_dim) {

        switch (edge_picker_type) {
            case RandomPickerType::Uniform:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::Uniform><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::Linear:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::Linear><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::ExponentialIndex:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialIndex><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::ExponentialWeight:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialWeight><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::TEST_FIRST:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_FIRST><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            case RandomPickerType::TEST_LAST:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_LAST><<<grid, block_dim>>>(
                    temporal_graph, walk_set, step_number, max_walk_len, num_walks, rand_nums);
                break;
            default:
                break;
        }
    }

    // Helper function to handle intermediate steps with different edge picker types
    template <bool IsDirected, bool Forward>
    void handle_intermediate_steps(
        TemporalGraphStore *temporal_graph,
        const WalkSet *walk_set,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const double *rand_nums,
        const dim3 &grid,
        const dim3 &block_dim) {

        for (int step_number = 1; step_number < max_walk_len; step_number++) {
            dispatch_intermediate_edges_kernel<IsDirected, Forward>(
                temporal_graph,
                walk_set,
                step_number,
                max_walk_len,
                num_walks,
                edge_picker_type,
                rand_nums,
                grid,
                block_dim);
        }
    }

    inline void launch_random_walk_kernels(
        TemporalGraphStore *temporal_graph,
        const bool is_directed,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const double *rand_nums,
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

        // Launch pick_start_edges_kernel
        if (is_directed) {
            if (should_walk_forward) {
                dispatch_start_edges_kernel<true, true>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, rand_nums, grid, block_dim);
            } else {
                dispatch_start_edges_kernel<true, false>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, rand_nums, grid, block_dim);
            }
        } else {
            if (should_walk_forward) {
                dispatch_start_edges_kernel<false, true>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, rand_nums, grid, block_dim);
            } else {
                dispatch_start_edges_kernel<false, false>(
                    temporal_graph, walk_set, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, rand_nums, grid, block_dim);
            }
        }

        // Launch intermediate edge kernels for each step
        if (is_directed) {
            if (should_walk_forward) {
                handle_intermediate_steps<true, true>(
                    temporal_graph, walk_set, max_walk_len, num_walks_int, edge_picker_type, rand_nums, grid, block_dim);
            } else {
                handle_intermediate_steps<true, false>(
                    temporal_graph, walk_set, max_walk_len, num_walks_int, edge_picker_type, rand_nums, grid, block_dim);
            }
        } else {
            if (should_walk_forward) {
                handle_intermediate_steps<false, true>(
                    temporal_graph, walk_set, max_walk_len, num_walks_int, edge_picker_type, rand_nums, grid, block_dim);
            } else {
                handle_intermediate_steps<false, false>(
                    temporal_graph, walk_set, max_walk_len, num_walks_int, edge_picker_type, rand_nums, grid, block_dim);
            }
        }

        // Launch reverse_walk_kernel if walking backward
        if (!should_walk_forward) {
            reverse_walks_kernel<<<grid, block_dim>>>(walk_set, num_walks_int);
        }
    }

    #endif
}

#endif //TEMPORAL_RANDOM_WALK_KERNELS_CUH
