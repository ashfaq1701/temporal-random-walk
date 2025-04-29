#include "temporal_random_walk.cuh"

#include <omp.h>

#include "../utils/utils.cuh"
#include "../utils/random.cuh"
#include "../common/setup.cuh"
#include "../common/random_gen.cuh"

HOST void temporal_random_walk::add_multiple_edges(
    const TemporalRandomWalkStore* temporal_random_walk,
    const Edge* edge_infos,
    const size_t num_edges) {

    #ifdef HAS_CUDA
    if (temporal_random_walk->use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            temporal_random_walk->temporal_graph,
            edge_infos,
            num_edges);
    }
    else
    #endif
    {
        temporal_graph::add_multiple_edges_std(
            temporal_random_walk->temporal_graph,
            edge_infos,
            num_edges);
    }
}

HOST size_t temporal_random_walk::get_node_count(const TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_graph::get_node_count(temporal_random_walk->temporal_graph);
}

HOST DEVICE size_t temporal_random_walk::get_edge_count(const TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_graph::get_total_edges(temporal_random_walk->temporal_graph);
}

HOST DataBlock<int> temporal_random_walk::get_node_ids(const TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
}

HOST DataBlock<Edge> temporal_random_walk::get_edges(const TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_graph::get_edges(temporal_random_walk->temporal_graph);
}

HOST bool temporal_random_walk::get_is_directed(const TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_random_walk->is_directed;
}

HOST void temporal_random_walk::clear(TemporalRandomWalkStore* temporal_random_walk) {
   temporal_random_walk->temporal_graph = new TemporalGraphStore(
       temporal_random_walk->is_directed,
       temporal_random_walk->use_gpu,
       temporal_random_walk->max_time_capacity,
       temporal_random_walk->enable_weight_computation,
       temporal_random_walk->timescale_bound);
}

HOST DEVICE bool temporal_random_walk::get_should_walk_forward(const WalkDirection walk_direction) {
    switch (walk_direction)
    {
        case WalkDirection::Forward_In_Time:
            return true;
        case WalkDirection::Backward_In_Time:
            return false;
        default:
            return true;
    }
}

HOST void temporal_random_walk::generate_random_walk_and_time_std(
    const TemporalRandomWalkStore* temporal_random_walk,
    const int walk_idx,
    const WalkSet* walk_set,
    const RandomPickerType* edge_picker_type,
    const RandomPickerType* start_picker_type,
    const int max_walk_len,
    const bool should_walk_forward,
    const int start_node_id,
    const double* rand_nums) {

    const size_t rand_nums_start_idx_for_walk = walk_idx * max_walk_len * 3;

    Edge start_edge;
    if (start_node_id == -1) {
        start_edge = temporal_graph::get_edge_at_host(
            temporal_random_walk->temporal_graph,
            *start_picker_type,
            -1,
            should_walk_forward,
            rand_nums[rand_nums_start_idx_for_walk],
            rand_nums[rand_nums_start_idx_for_walk + 1]);
    } else {
        start_edge = temporal_graph::get_node_edge_at_host(
            temporal_random_walk->temporal_graph,
            start_node_id,
            *start_picker_type,
            -1,
            should_walk_forward,
            rand_nums[rand_nums_start_idx_for_walk],
            rand_nums[rand_nums_start_idx_for_walk + 1]
        );
    }

    if (start_edge.i == -1) {
        return;
    }

    int current_node;
    int64_t current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;

    // Extract start edge components
    const int start_src = start_edge.u;
    const int start_dst = start_edge.i;
    const int64_t start_ts = start_edge.ts;

    if (temporal_random_walk->temporal_graph->is_directed) {
        if (should_walk_forward) {
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
            : pick_random_number(start_src, start_dst, rand_nums[rand_nums_start_idx_for_walk + 2]);
        walk_set->add_hop(walk_idx, picked_node, current_timestamp);
        current_node = pick_other_number(start_src, start_dst, picked_node);
    }

    current_timestamp = start_ts;

    // Perform the walk
    while (walk_set->get_walk_len(walk_idx) < max_walk_len && current_node != -1) {
        const auto walk_len = walk_set->get_walk_len(walk_idx);
        const auto step_start_idx = rand_nums_start_idx_for_walk + walk_len * 3;
        const auto group_selector_rand_num = rand_nums[step_start_idx];
        const auto edge_selector_rand_num = rand_nums[step_start_idx + 1];


        walk_set->add_hop(walk_idx, current_node, current_timestamp);

        Edge next_edge = temporal_graph::get_node_edge_at_host(
            temporal_random_walk->temporal_graph,
            current_node,
            *edge_picker_type,
            current_timestamp,
            should_walk_forward,
            group_selector_rand_num,
            edge_selector_rand_num
        );

        if (next_edge.ts == -1) {
            current_node = -1;
            continue;
        }

        if (temporal_random_walk->temporal_graph->is_directed) {
            current_node = should_walk_forward ? next_edge.i : next_edge.u;
        } else {
            current_node = pick_other_number(next_edge.u, next_edge.i, current_node);
        }

        current_timestamp = next_edge.ts;
    }

    // Reverse the walk if we walked backward
    if (!should_walk_forward) {
        walk_set->reverse_walk(walk_idx);
    }
}

HOST WalkSet temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
    const TemporalRandomWalkStore* temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const auto repeated_node_ids = repeat_elements(
        temporal_graph::get_node_ids(temporal_random_walk->temporal_graph),
        num_walks_per_node,
        temporal_random_walk->use_gpu);
    shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);

    WalkSet walk_set(repeated_node_ids.size, max_walk_len, temporal_random_walk->use_gpu);
    double* rand_nums = generate_n_random_numbers(repeated_node_ids.size * max_walk_len * 3, false);

    #pragma omp parallel for schedule(dynamic)
    for (int walk_idx = 0; walk_idx < repeated_node_ids.size; ++walk_idx) {
        const int start_node_id = repeated_node_ids.data[walk_idx];
        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        generate_random_walk_and_time_std(
            temporal_random_walk,
            walk_idx,
            &walk_set,
            walk_bias,
            initial_edge_bias,
            max_walk_len,
            should_walk_forward,
            start_node_id,
            rand_nums);
    }

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

HOST WalkSet temporal_random_walk::get_random_walks_and_times_std(
    const TemporalRandomWalkStore* temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    WalkSet walk_set(num_walks_total, max_walk_len, temporal_random_walk->use_gpu);
    double* rand_nums = generate_n_random_numbers(num_walks_total * max_walk_len * 3, false);

    #pragma omp parallel for schedule(dynamic)
    for (int walk_idx = 0; walk_idx < num_walks_total; ++walk_idx) {
        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        generate_random_walk_and_time_std(
            temporal_random_walk,
            walk_idx,
            &walk_set,
            walk_bias,
            initial_edge_bias,
            max_walk_len,
            should_walk_forward,
            -1,
            rand_nums);
    }

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

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

    const size_t rand_nums_start_idx_for_walk = walk_idx * max_walk_len * 3;

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
        const auto step_start_idx = rand_nums_start_idx_for_walk + walk_len * 3;
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

HOST WalkSet temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
    const TemporalRandomWalkStore* temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Get all node IDs and repeat them for multiple walks per node
    const auto node_ids = temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
    const auto repeated_node_ids = repeat_elements(
        node_ids,
        num_walks_per_node,
        temporal_random_walk->use_gpu);

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        temporal_random_walk->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Initialize random numbers between [0.0, 1.0)
    double* rand_nums = generate_n_random_numbers(repeated_node_ids.size * max_walk_len * 3, true);

    // Shuffle node IDs for randomization
    shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
    CUDA_KERNEL_CHECK("After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");

    // Create and initialize the walk set on device
    WalkSet walk_set(repeated_node_ids.size, max_walk_len, true);
    WalkSet* d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore* d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

    temporal_random_walk::generate_random_walks_kernel<<<grid_dim, block_dim>>>(
        d_walk_set,
        d_temporal_graph,
        repeated_node_ids.data,
        *walk_bias,
        *initial_edge_bias,
        max_walk_len,
        temporal_random_walk->is_directed,
        walk_direction,
        static_cast<int>(repeated_node_ids.size),
        rand_nums
    );
    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_for_all_nodes_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(repeated_node_ids.size, max_walk_len, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    clear_memory(&rand_nums, true);
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST WalkSet temporal_random_walk::get_random_walks_and_times_cuda(
    const TemporalRandomWalkStore* temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        num_walks_total,
        temporal_random_walk->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Initialize all start node IDs to -1 (indicating random start)
    int* start_node_ids;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&start_node_ids, num_walks_total * sizeof(int)));
    fill_memory(start_node_ids, num_walks_total, -1, temporal_random_walk->use_gpu);

    // Initialize random numbers between [0.0, 1.0)
    double* rand_nums = generate_n_random_numbers(num_walks_total * max_walk_len * 3, true);

    // Create and initialize the walk set on device
    WalkSet walk_set(num_walks_total, max_walk_len, true);
    WalkSet* d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore* d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

    // Launch kernel
    temporal_random_walk::generate_random_walks_kernel<<<grid_dim, block_dim>>>(
        d_walk_set,
        d_temporal_graph,
        start_node_ids,
        *walk_bias,
        *initial_edge_bias,
        max_walk_len,
        temporal_random_walk->is_directed,
        walk_direction,
        num_walks_total,
        rand_nums
    );
    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(num_walks_total, max_walk_len, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    clear_memory(&rand_nums, true);
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(start_node_ids));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST TemporalRandomWalkStore* temporal_random_walk::to_device_ptr(const TemporalRandomWalkStore* temporal_random_walk) {
    // Create a new TemporalRandomWalk object on the device
    TemporalRandomWalkStore* device_temporal_random_walk;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_temporal_random_walk, sizeof(TemporalRandomWalkStore)));

    // Create a temporary copy to modify for device pointers
    TemporalRandomWalkStore temp_temporal_random_walk = *temporal_random_walk;

    // Copy TemporalGraph to device
    if (temporal_random_walk->temporal_graph) {
        temp_temporal_random_walk.temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);
    }

    // cudaDeviceProp aren't needed on device, set to nullptr
    temp_temporal_random_walk.cuda_device_prop = nullptr;

    // Make sure use_gpu is set to true
    temp_temporal_random_walk.use_gpu = true;

    // Copy the updated struct to device
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_temporal_random_walk, &temp_temporal_random_walk, sizeof(TemporalRandomWalkStore), cudaMemcpyHostToDevice));

    temp_temporal_random_walk.owns_data = false;

    return device_temporal_random_walk;
}

#endif
