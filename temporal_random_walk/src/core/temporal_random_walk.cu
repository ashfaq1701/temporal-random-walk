#include "temporal_random_walk.cuh"

#include "../common/setup.cuh"

#include "temporal_random_walk_cpu.cuh"
#include "temporal_random_walk_kernels_full_walk.cuh"
#include "temporal_random_walk_kernels_step_based.cuh"

/**
 * Common functions
*/

HOST void temporal_random_walk::add_multiple_edges(
    const TemporalRandomWalkStore *temporal_random_walk,
    const int *sources,
    const int *targets,
    const int64_t *timestamps,
    const size_t num_edges) {
    #ifdef HAS_CUDA
    if (temporal_random_walk->use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            temporal_random_walk->temporal_graph,
            sources,
            targets,
            timestamps,
            num_edges);
    } else
    #endif
    {
        temporal_graph::add_multiple_edges_std(
            temporal_random_walk->temporal_graph,
            sources,
            targets,
            timestamps,
            num_edges);
    }
}

HOST size_t temporal_random_walk::get_node_count(const TemporalRandomWalkStore *temporal_random_walk) {
    return temporal_graph::get_node_count(temporal_random_walk->temporal_graph);
}

HOST DEVICE size_t temporal_random_walk::get_edge_count(const TemporalRandomWalkStore *temporal_random_walk) {
    return temporal_graph::get_total_edges(temporal_random_walk->temporal_graph);
}

HOST DataBlock<int> temporal_random_walk::get_node_ids(const TemporalRandomWalkStore *temporal_random_walk) {
    return temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
}

HOST DataBlock<Edge> temporal_random_walk::get_edges(const TemporalRandomWalkStore *temporal_random_walk) {
    return temporal_graph::get_edges(temporal_random_walk->temporal_graph);
}

HOST bool temporal_random_walk::get_is_directed(const TemporalRandomWalkStore *temporal_random_walk) {
    return temporal_random_walk->is_directed;
}

HOST void temporal_random_walk::clear(TemporalRandomWalkStore *temporal_random_walk) {
    temporal_random_walk->temporal_graph = new TemporalGraphStore(
        temporal_random_walk->is_directed,
        temporal_random_walk->use_gpu,
        temporal_random_walk->max_time_capacity,
        temporal_random_walk->enable_weight_computation,
        temporal_random_walk->timescale_bound,
        temporal_random_walk->node2vec_p,
        temporal_random_walk->node2vec_q);
}

/**
 * Std implementations
 */

HOST WalkSet temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
    const TemporalRandomWalkStore *temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_per_node,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const auto repeated_node_ids = repeat_elements(
        temporal_graph::get_node_ids(temporal_random_walk->temporal_graph),
        num_walks_per_node,
        temporal_random_walk->use_gpu);
    shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);

    WalkSet walk_set(repeated_node_ids.size, max_walk_len, temporal_random_walk->walk_padding_value, temporal_random_walk->use_gpu);

    // max_walk_len requires walk_len - 1 steps
    double *rand_nums = generate_n_random_numbers(repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    launch_random_walk_cpu(
        temporal_random_walk->temporal_graph,
        temporal_random_walk->is_directed,
        &walk_set,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums);

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

HOST WalkSet temporal_random_walk::get_random_walks_and_times_std(
    const TemporalRandomWalkStore *temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_total,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    WalkSet walk_set(num_walks_total, max_walk_len, temporal_random_walk->walk_padding_value, temporal_random_walk->use_gpu);
    // max_walk_len requires walk_len - 1 steps
    double *rand_nums = generate_n_random_numbers(num_walks_total + num_walks_total * max_walk_len * 2, false);

    const std::vector<int> start_node_ids(num_walks_total, -1);

    launch_random_walk_cpu(
        temporal_random_walk->temporal_graph,
        temporal_random_walk->is_directed,
        &walk_set,
        max_walk_len,
        start_node_ids.data(),
        num_walks_total,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums);

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

/**
 * CUDA implementations
 */

#ifdef HAS_CUDA

HOST WalkSet temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
    const TemporalRandomWalkStore *temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_per_node,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Get all node IDs and repeat them for multiple walks per node
    const auto node_ids = temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
    const auto repeated_node_ids = repeat_elements(
        node_ids,
        num_walks_per_node,
        temporal_random_walk->use_gpu);

    uint64_t base_seed;
    if (temporal_random_walk->global_seed != EMPTY_GLOBAL_SEED) {
        base_seed = temporal_random_walk->global_seed;
    } else {
        base_seed = secure_random_seed();
    }

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        temporal_random_walk->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Shuffle node IDs for randomization
    shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
    CUDA_KERNEL_CHECK("After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");

    // Create and initialize the walk set on device
    const WalkSet walk_set(repeated_node_ids.size, max_walk_len, temporal_random_walk->walk_padding_value, true);
    WalkSet *d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

    switch (kernel_launch_type) {
        case KernelLaunchType::FULL_WALK:
            launch_random_walk_kernel_full_walk(
                d_temporal_graph,
                temporal_random_walk->is_directed,
                d_walk_set,
                max_walk_len,
                repeated_node_ids.data,
                repeated_node_ids.size,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                base_seed,
                grid_dim,
                block_dim);
        break;

        case KernelLaunchType::STEP_BASED:
            launch_random_walk_kernel_step_based(
                d_temporal_graph,
                temporal_random_walk->is_directed,
                d_walk_set,
                max_walk_len,
                repeated_node_ids.data,
                repeated_node_ids.size,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                base_seed,
                grid_dim,
                block_dim);
        break;

        default:
            throw std::runtime_error("Unknown KernelLaunchType");
    }

    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_for_all_nodes_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(repeated_node_ids.size, max_walk_len, temporal_random_walk->walk_padding_value, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST WalkSet temporal_random_walk::get_random_walks_and_times_cuda(
    const TemporalRandomWalkStore *temporal_random_walk,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_total,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    uint64_t base_seed;
    if (temporal_random_walk->global_seed != EMPTY_GLOBAL_SEED) {
        base_seed = temporal_random_walk->global_seed;
    } else {
        base_seed = secure_random_seed();
    }

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        num_walks_total,
        temporal_random_walk->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Initialize all start node IDs to -1 (indicating random start)
    int *start_node_ids;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&start_node_ids, num_walks_total * sizeof(int)));
    fill_memory(start_node_ids, num_walks_total, -1, temporal_random_walk->use_gpu);

    // Create and initialize the walk set on device
    const WalkSet walk_set(num_walks_total, max_walk_len, temporal_random_walk->walk_padding_value, true);
    WalkSet *d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

    switch (kernel_launch_type) {
        case KernelLaunchType::FULL_WALK:
            launch_random_walk_kernel_full_walk(
                d_temporal_graph,
                temporal_random_walk->is_directed,
                d_walk_set,
                max_walk_len,
                start_node_ids,
                num_walks_total,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                base_seed,
                grid_dim,
                block_dim);
        break;

        case KernelLaunchType::STEP_BASED:
            launch_random_walk_kernel_step_based(
                d_temporal_graph,
                temporal_random_walk->is_directed,
                d_walk_set,
                max_walk_len,
                start_node_ids,
                num_walks_total,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                base_seed,
                grid_dim,
                block_dim);
        break;

        default:
            throw std::runtime_error("Unknown KernelLaunchType");
    }

    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(num_walks_total, max_walk_len, temporal_random_walk->walk_padding_value, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(start_node_ids));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST TemporalRandomWalkStore* temporal_random_walk::to_device_ptr(const TemporalRandomWalkStore *temporal_random_walk) {
    // Create a new TemporalRandomWalk object on the device
    TemporalRandomWalkStore *device_temporal_random_walk;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_temporal_random_walk, sizeof(TemporalRandomWalkStore)));

    // Create a temporary copy to modify for device pointers
    TemporalRandomWalkStore temp_temporal_random_walk = *temporal_random_walk;

    // Copy TemporalGraph to device
    if (temporal_random_walk->temporal_graph) {
        temp_temporal_random_walk.temporal_graph = temporal_graph::to_device_ptr(
            temporal_random_walk->temporal_graph);
    }

    // cudaDeviceProp aren't needed on device, set to nullptr
    temp_temporal_random_walk.cuda_device_prop = nullptr;

    // Make sure use_gpu is set to true
    temp_temporal_random_walk.use_gpu = true;

    // Copy the updated struct to device
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(device_temporal_random_walk, &temp_temporal_random_walk, sizeof(TemporalRandomWalkStore),
            cudaMemcpyHostToDevice));

    temp_temporal_random_walk.owns_data = false;

    return device_temporal_random_walk;
}

HOST void temporal_random_walk::free_device_pointers(TemporalRandomWalkStore *d_temporal_random_walk) {
    if (!d_temporal_random_walk) return;

    // Copy the struct from device to host to access pointers
    TemporalRandomWalkStore h_temporal_random_walk;
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&h_temporal_random_walk, d_temporal_random_walk, sizeof(TemporalRandomWalkStore),
            cudaMemcpyDeviceToHost));
    h_temporal_random_walk.owns_data = false;

    // Free only the nested device pointers (not their underlying data)
    if (h_temporal_random_walk.temporal_graph) temporal_graph::free_device_pointers(
        h_temporal_random_walk.temporal_graph);
    if (h_temporal_random_walk.cuda_device_prop) clear_memory(&h_temporal_random_walk.cuda_device_prop, true);

    clear_memory(&d_temporal_random_walk, true);
}

#endif

HOST size_t temporal_random_walk::get_memory_used(TemporalRandomWalkStore* temporal_random_walk) {
    return temporal_graph::get_memory_used(temporal_random_walk->temporal_graph);
}
