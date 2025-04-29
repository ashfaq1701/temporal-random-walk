#include "temporal_random_walk_kernels.cuh"

#include "temporal_random_walk.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "../stores/edge_selectors.cuh"

#ifdef HAS_CUDA

__global__ void temporal_random_walk::generate_random_walks_kernel(
    const WalkSet* walk_set,
    TemporalGraphStore* temporal_graph,
    const int* start_node_ids,
    const RandomPickerType edge_picker_type,
    const RandomPickerType start_picker_type,
    const int max_walk_len,
    const bool is_directed,
    const WalkDirection walk_direction,
    const int num_walks,
    const double* rand_nums) {

    const int walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    const size_t rand_nums_start_idx_for_walk = walk_idx + walk_idx * max_walk_len * 2;

    const bool should_walk_forward = get_should_walk_forward(walk_direction);

    Edge start_edge;
    if (start_node_ids[walk_idx] == -1) {
        start_edge = temporal_graph::get_edge_at_device(
            temporal_graph,
            start_picker_type,
            -1,
            should_walk_forward,
            rand_nums[rand_nums_start_idx_for_walk],
            rand_nums[rand_nums_start_idx_for_walk + 1]);
    } else {
        start_edge = temporal_graph::get_node_edge_at_device(
            temporal_graph,
            start_node_ids[walk_idx],
            start_picker_type,
            -1,
            should_walk_forward,
            rand_nums[rand_nums_start_idx_for_walk],
            rand_nums[rand_nums_start_idx_for_walk + 1]);
    }

    if (start_edge.i == -1) {
        return;
    }

    int current_node;
    int64_t current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;
    int start_src = start_edge.u;
    int start_dst = start_edge.i;
    int64_t start_ts = start_edge.ts;

    if (is_directed) {
        if (should_walk_forward) {
            walk_set->add_hop(walk_idx, start_src, current_timestamp);
            current_node = start_dst;
        } else {
            walk_set->add_hop(walk_idx, start_dst, current_timestamp);
            current_node = start_src;
        }
    } else {
        // For undirected graphs, use specified start node or pick a random node
        const int picked_node = (start_node_ids[walk_idx] != -1)
            ? start_node_ids[walk_idx]
            : pick_random_number(start_src, start_dst, rand_nums[rand_nums_start_idx_for_walk + 2]);

        walk_set->add_hop(walk_idx, picked_node, current_timestamp);
        current_node = pick_other_number(start_src, start_dst, picked_node);
    }

    current_timestamp = start_ts;

    while (walk_set->get_walk_len_device(walk_idx) < max_walk_len && current_node != -1) {
        const auto walk_len = walk_set->get_walk_len_device(walk_idx);
        const auto step_start_idx = rand_nums_start_idx_for_walk + walk_len * 2 + 1;
        const auto group_selector_rand_num = rand_nums[step_start_idx];
        const auto edge_selector_rand_num = rand_nums[step_start_idx + 1];

        walk_set->add_hop(walk_idx, current_node, current_timestamp);

        Edge next_edge = temporal_graph::get_node_edge_at_device(
            temporal_graph,
            current_node,
            edge_picker_type,
            current_timestamp,
            should_walk_forward,
            group_selector_rand_num,
            edge_selector_rand_num);

        if (next_edge.ts == -1) {
            current_node = -1;
            continue;
        }

        if (is_directed) {
            current_node = should_walk_forward ? next_edge.i : next_edge.u;
        } else {
            current_node = pick_other_number(next_edge.u, next_edge.i, current_node);
        }

        current_timestamp = next_edge.ts;
    }

    // If walking backward, reverse the walk
    if (!should_walk_forward) {
        walk_set->reverse_walk(walk_idx);
    }
}

#endif
