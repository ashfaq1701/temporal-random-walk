#ifndef TEMPORAL_RANDOM_WALK_KERNELS_STEP_BASED_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_STEP_BASED_CUH

#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../graph/temporal_graph.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "helpers.cuh"
#include "../graph/edge_selectors.cuh"

namespace temporal_random_walk {

    #ifdef HAS_CUDA

    template <bool IsDirected, bool Forward, RandomPickerType StartPickerType>
    __global__ void pick_start_edges_kernel(
        TemporalGraphView view,
        WalkSetView walk_set,
        const int *__restrict__ start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const uint64_t base_seed) {
        const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        if (max_walk_len == 0) return;

        // Start kernel draws from the head of this walk's Philox stream.
        // Intermediate-step kernels skip past these 3 positions (see
        // pick_intermediate_edges_kernel's offset argument).
        PhiloxState rng;
        init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx));

        const double r_start_a = draw_u01_philox(rng);
        const double r_start_b = draw_u01_philox(rng);

        const auto padding_value = walk_set.nodes[walk_idx * max_walk_len];
        InternalEdge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
                view,
                -1, // timestamp
                r_start_a,
                r_start_b);
        } else {
            walk_set.nodes[walk_idx * max_walk_len] = start_node_ids[walk_idx];

            start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
                view,
                start_node_ids[walk_idx],
                -1, // timestamp
                -1,
                r_start_a,
                r_start_b);
        }

        if (start_edge.i == -1) {
            walk_set.nodes[walk_idx * max_walk_len] = padding_value;
            return;
        }

        const int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
        const int start_src = start_edge.u;
        const int start_dst = start_edge.i;
        const int64_t start_ts = start_edge.ts;

        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set.add_hop(walk_idx, start_src, sentinel_timestamp);
                walk_set.add_hop(walk_idx, start_dst, start_ts, start_edge.edge_id);
            } else {
                walk_set.add_hop(walk_idx, start_dst, sentinel_timestamp);
                walk_set.add_hop(walk_idx, start_src, start_ts, start_edge.edge_id);
            }
        } else {
            // For undirected graphs, use specified start node or pick a random node.
            // This consumes the 3rd draw from the start kernel's budget so the
            // step-kernel's offset (below) always lands past it.
            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst, draw_u01_philox(rng));
            const int other_node = pick_other_number(start_src, start_dst, picked_node);

            walk_set.add_hop(walk_idx, picked_node, sentinel_timestamp);
            walk_set.add_hop(walk_idx, other_node, start_ts, start_edge.edge_id);
        }
    }

    // Offset in this walk's Philox stream that each step kernel starts from.
    // The start kernel reserves up to 3 positions (directed: 2, undirected: 3);
    // we skip past the max so the two kernels never draw overlapping positions
    // for any walk. Each step kernel consumes exactly 2 positions.
    DEVICE __forceinline__ uint64_t step_kernel_philox_offset(const int step_number) {
        constexpr uint64_t START_KERNEL_BUDGET = 3ULL;
        return START_KERNEL_BUDGET + static_cast<uint64_t>(step_number) * 2ULL;
    }

    template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
    __global__ void pick_intermediate_edges_kernel(
        TemporalGraphView view,
        WalkSetView walk_set,
        const int step_number,
        const int max_walk_len,
        const size_t num_walks,
        const uint64_t base_seed) {

        const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        if (step_number >= max_walk_len - 1) {
            return;
        }

        const size_t offset = static_cast<size_t>(walk_idx) * static_cast<size_t>(max_walk_len) + static_cast<size_t>(step_number); // Get endpoint of previous step (step_number - 1). And endpoint is (step_number - 1 + 1).
        const int last_node = walk_set.nodes[offset];
        const int last_ts = walk_set.timestamps[offset];
        const int prev_node = step_number > 0 ? walk_set.nodes[offset - 1] : -1;

        // One Philox init per thread at the correct step offset, two draws.
        PhiloxState rng;
        init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx),
                          step_kernel_philox_offset(step_number));

        const double r_edge_a = draw_u01_philox(rng);
        const double r_edge_b = draw_u01_philox(rng);

        const InternalEdge next_edge = temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
                view,
                last_node,
                last_ts,
                prev_node,
                r_edge_a,
                r_edge_b);

        if (next_edge.ts == -1) {
            return;
        }

        if constexpr (IsDirected) {
            walk_set.add_hop(walk_idx, Forward ? next_edge.i : next_edge.u, next_edge.ts, next_edge.edge_id);
        } else {
            const auto node_to_add = pick_other_number(next_edge.u, next_edge.i, last_node);
            walk_set.add_hop(walk_idx, node_to_add, next_edge.ts, next_edge.edge_id);
        }
    }

    __global__ static void reverse_walks_kernel(WalkSetView walk_set, const size_t num_walks) {
        const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;
        walk_set.reverse_walk(walk_idx);
    }

    // Helper function to dispatch start edge kernels
    template <bool IsDirected, bool Forward>
    void dispatch_start_edges_kernel(
        const TemporalGraphView& view,
        WalkSetView walk_set_view,
        const int *start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType start_picker_type,
        const uint64_t base_seed,
        const dim3 &grid,
        const dim3 &block_dim) {

        switch (start_picker_type) {
            case RandomPickerType::Uniform:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::Uniform><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::Linear:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::Linear><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::ExponentialIndex:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialIndex><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::ExponentialWeight:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialWeight><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TemporalNode2Vec:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::TemporalNode2Vec><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TEST_FIRST:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_FIRST><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TEST_LAST:
                pick_start_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_LAST><<<grid, block_dim>>>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks, base_seed);
                break;
            default:
                break;
        }
    }

    // Helper function to dispatch intermediate edge kernels
    template <bool IsDirected, bool Forward>
    void dispatch_intermediate_edges_kernel(
        const TemporalGraphView& view,
        WalkSetView walk_set_view,
        const int step_number,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const uint64_t base_seed,
        const dim3 &grid,
        const dim3 &block_dim) {

        switch (edge_picker_type) {
            case RandomPickerType::Uniform:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::Uniform><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::Linear:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::Linear><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::ExponentialIndex:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialIndex><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::ExponentialWeight:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::ExponentialWeight><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TemporalNode2Vec:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::TemporalNode2Vec><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TEST_FIRST:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_FIRST><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            case RandomPickerType::TEST_LAST:
                pick_intermediate_edges_kernel<IsDirected, Forward, RandomPickerType::TEST_LAST><<<grid, block_dim>>>(
                    view, walk_set_view, step_number, max_walk_len, num_walks, base_seed);
                break;
            default:
                break;
        }
    }

    // Helper function to handle intermediate steps with different edge picker types
    template <bool IsDirected, bool Forward>
    void handle_intermediate_steps(
        const TemporalGraphView& view,
        WalkSetView walk_set_view,
        const int max_walk_len,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const uint64_t base_seed,
        const dim3 &grid,
        const dim3 &block_dim) {

        for (int step_number = 1; step_number < max_walk_len; step_number++) {
            dispatch_intermediate_edges_kernel<IsDirected, Forward>(
                view,
                walk_set_view,
                step_number,
                max_walk_len,
                num_walks,
                edge_picker_type,
                base_seed,
                grid,
                block_dim);
        }
    }

    inline void launch_random_walk_kernel_step_based(
        const TemporalGraphView& view,
        const bool is_directed,
        WalkSetView walk_set_view,
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

        // Launch pick_start_edges_kernel
        if (is_directed) {
            if (should_walk_forward) {
                dispatch_start_edges_kernel<true, true>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, base_seed, grid, block_dim);
            } else {
                dispatch_start_edges_kernel<true, false>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, base_seed, grid, block_dim);
            }
        } else {
            if (should_walk_forward) {
                dispatch_start_edges_kernel<false, true>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, base_seed, grid, block_dim);
            } else {
                dispatch_start_edges_kernel<false, false>(
                    view, walk_set_view, start_node_ids, max_walk_len, num_walks_int,
                    start_picker_type, base_seed, grid, block_dim);
            }
        }

        // Launch intermediate edge kernels for each step
        if (is_directed) {
            if (should_walk_forward) {
                handle_intermediate_steps<true, true>(
                    view, walk_set_view, max_walk_len, num_walks_int, edge_picker_type, base_seed, grid, block_dim);
            } else {
                handle_intermediate_steps<true, false>(
                    view, walk_set_view, max_walk_len, num_walks_int, edge_picker_type, base_seed, grid, block_dim);
            }
        } else {
            if (should_walk_forward) {
                handle_intermediate_steps<false, true>(
                    view, walk_set_view, max_walk_len, num_walks_int, edge_picker_type, base_seed, grid, block_dim);
            } else {
                handle_intermediate_steps<false, false>(
                    view, walk_set_view, max_walk_len, num_walks_int, edge_picker_type, base_seed, grid, block_dim);
            }
        }

        // Launch reverse_walk_kernel if walking backward
        if (!should_walk_forward) {
            reverse_walks_kernel<<<grid, block_dim>>>(walk_set_view, num_walks_int);
        }
    }

    #endif

}

#endif //TEMPORAL_RANDOM_WALK_KERNELS_STEP_BASED_CUH
