#ifndef TEMPORAL_RANDOM_WALK_STORE_H
#define TEMPORAL_RANDOM_WALK_STORE_H

#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../data/walk_set/walk_set.cuh"
#include "../stores/temporal_graph.cuh"
#include "../utils/utils.cuh"
#include "../utils/random.cuh"
#include "../common/setup.cuh"
#include "../common/random_gen.cuh"
#include "temporal_random_walk_cpu.cuh"
#include "temporal_random_walk_kernels.cuh"

struct TemporalRandomWalkStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data = true;
    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;

    #ifdef HAS_CUDA
    cudaDeviceProp *cuda_device_prop;
    #endif
    TemporalGraphStore *temporal_graph;

    TemporalRandomWalkStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound) {
        this->is_directed = is_directed;
        this->use_gpu = use_gpu;
        this->max_time_capacity = max_time_capacity;
        this->enable_weight_computation = enable_weight_computation;
        this->timescale_bound = timescale_bound;

        this->temporal_graph = new TemporalGraphStore(
            is_directed,
            use_gpu,
            max_time_capacity,
            enable_weight_computation,
            timescale_bound);

        #ifdef HAS_CUDA
        cuda_device_prop = new cudaDeviceProp();
        cudaGetDeviceProperties(cuda_device_prop, 0);
        #endif
    }

    ~TemporalRandomWalkStore() {
        if (owns_data) {
            delete temporal_graph;

            #ifdef HAS_CUDA
            if (use_gpu) {
                clear_memory(&cuda_device_prop, use_gpu);
            } else
            #endif
            {
                #ifdef HAS_CUDA
                delete cuda_device_prop;
                #endif
            }
        }
    }
};

namespace temporal_random_walk {
    /**
     * Common functions
     */
    HOST inline void add_multiple_edges(
        const TemporalRandomWalkStore *temporal_random_walk,
        const Edge *edge_infos,
        const size_t num_edges) {
        #ifdef HAS_CUDA
        if (temporal_random_walk->use_gpu) {
            temporal_graph::add_multiple_edges_cuda(
                temporal_random_walk->temporal_graph,
                edge_infos,
                num_edges);
        } else
        #endif
        {
            temporal_graph::add_multiple_edges_std(
                temporal_random_walk->temporal_graph,
                edge_infos,
                num_edges);
        }
    }

    HOST inline size_t get_node_count(const TemporalRandomWalkStore *temporal_random_walk) {
        return temporal_graph::get_node_count(temporal_random_walk->temporal_graph);
    }

    HOST DEVICE inline size_t get_edge_count(const TemporalRandomWalkStore *temporal_random_walk) {
        return temporal_graph::get_total_edges(temporal_random_walk->temporal_graph);
    }

    HOST inline DataBlock<int> get_node_ids(const TemporalRandomWalkStore *temporal_random_walk) {
        return temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
    }

    HOST inline DataBlock<Edge> get_edges(const TemporalRandomWalkStore *temporal_random_walk) {
        return temporal_graph::get_edges(temporal_random_walk->temporal_graph);
    }

    HOST inline bool get_is_directed(const TemporalRandomWalkStore *temporal_random_walk) {
        return temporal_random_walk->is_directed;
    }

    HOST inline void clear(TemporalRandomWalkStore *temporal_random_walk) {
        temporal_random_walk->temporal_graph = new TemporalGraphStore(
            temporal_random_walk->is_directed,
            temporal_random_walk->use_gpu,
            temporal_random_walk->max_time_capacity,
            temporal_random_walk->enable_weight_computation,
            temporal_random_walk->timescale_bound);
    }

    /**
     * Std implementations
     */

    HOST inline WalkSet get_random_walks_and_times_for_all_nodes_std(
        const TemporalRandomWalkStore *temporal_random_walk,
        const int max_walk_len,
        const RandomPickerType *walk_bias,
        const int num_walks_per_node,
        const RandomPickerType *initial_edge_bias = nullptr,
        const WalkDirection walk_direction = WalkDirection::Forward_In_Time) {
        if (!initial_edge_bias) {
            initial_edge_bias = walk_bias;
        }

        const auto repeated_node_ids = repeat_elements(
            temporal_graph::get_node_ids(temporal_random_walk->temporal_graph),
            num_walks_per_node,
            temporal_random_walk->use_gpu);
        shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);

        WalkSet walk_set(repeated_node_ids.size, max_walk_len, temporal_random_walk->use_gpu);
        double *rand_nums = generate_n_random_numbers(
            repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

        #pragma omp parallel for schedule(dynamic)
        for (int walk_idx = 0; walk_idx < repeated_node_ids.size; ++walk_idx) {
            const int start_node_id = repeated_node_ids.data[walk_idx];

            launch_random_walk_generator(
                temporal_random_walk->temporal_graph,
                &walk_set,
                walk_idx,
                max_walk_len,
                start_node_id,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                rand_nums);
        }

        // Clean up
        clear_memory(&rand_nums, false);

        return walk_set;
    }

    HOST inline WalkSet get_random_walks_and_times_std(
        const TemporalRandomWalkStore *temporal_random_walk,
        const int max_walk_len,
        const RandomPickerType *walk_bias,
        const int num_walks_total,
        const RandomPickerType *initial_edge_bias = nullptr,
        const WalkDirection walk_direction = WalkDirection::Forward_In_Time) {
        if (!initial_edge_bias) {
            initial_edge_bias = walk_bias;
        }

        WalkSet walk_set(num_walks_total, max_walk_len, temporal_random_walk->use_gpu);
        double *rand_nums = generate_n_random_numbers(num_walks_total + num_walks_total * max_walk_len * 2, false);

        #pragma omp parallel for schedule(dynamic)
        for (int walk_idx = 0; walk_idx < num_walks_total; ++walk_idx) {
            launch_random_walk_generator(
                temporal_random_walk->temporal_graph,
                &walk_set,
                walk_idx,
                max_walk_len,
                -1,
                *walk_bias,
                *initial_edge_bias,
                walk_direction,
                rand_nums);
        }

        // Clean up
        clear_memory(&rand_nums, false);

        return walk_set;
    }

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA

    HOST inline WalkSet get_random_walks_and_times_for_all_nodes_cuda(
        const TemporalRandomWalkStore *temporal_random_walk,
        const int max_walk_len,
        const RandomPickerType *walk_bias,
        const int num_walks_per_node,
        const RandomPickerType *initial_edge_bias = nullptr,
        const WalkDirection walk_direction = WalkDirection::Forward_In_Time) {

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
        double *rand_nums = generate_n_random_numbers(
            repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, true);

        // Shuffle node IDs for randomization
        shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
        CUDA_KERNEL_CHECK("After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");

        // Create and initialize the walk set on device
        WalkSet walk_set(repeated_node_ids.size, max_walk_len, true);
        WalkSet *d_walk_set;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

        // Create device pointer for the temporal graph
        TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

        temporal_random_walk::launch_random_walks_kernel(
            d_temporal_graph,
            temporal_random_walk->is_directed,
            d_walk_set,
            max_walk_len,
            repeated_node_ids.data,
            repeated_node_ids.size,
            *walk_bias,
            *initial_edge_bias,
            walk_direction,
            rand_nums,
            grid_dim,
            block_dim);

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

    HOST inline WalkSet get_random_walks_and_times_cuda(
        const TemporalRandomWalkStore *temporal_random_walk,
        const int max_walk_len,
        const RandomPickerType *walk_bias,
        const int num_walks_total,
        const RandomPickerType *initial_edge_bias = nullptr,
        const WalkDirection walk_direction = WalkDirection::Forward_In_Time) {

        if (!initial_edge_bias) {
            initial_edge_bias = walk_bias;
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

        // Initialize random numbers between [0.0, 1.0)
        double *rand_nums = generate_n_random_numbers(num_walks_total + num_walks_total * max_walk_len * 2, true);

        // Create and initialize the walk set on device
        WalkSet walk_set(num_walks_total, max_walk_len, true);
        WalkSet *d_walk_set;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

        // Create device pointer for the temporal graph
        TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(temporal_random_walk->temporal_graph);

        // Launch kernel
        temporal_random_walk::launch_random_walks_kernel(
            d_temporal_graph,
            temporal_random_walk->is_directed,
            d_walk_set,
            max_walk_len,
            start_node_ids,
            num_walks_total,
            *walk_bias,
            *initial_edge_bias,
            walk_direction,
            rand_nums,
            grid_dim,
            block_dim);

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

    HOST inline TemporalRandomWalkStore *to_device_ptr(const TemporalRandomWalkStore* temporal_random_walk) {
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

    #endif
}
#endif // TEMPORAL_RANDOM_WALK_STORE_H
