#include "TemporalRandomWalkCUDA.cuh"

#include "../utils/rand_utils.cuh"
#include "../utils/utils.h"
#include "../cuda_common/setup.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
HOST TemporalRandomWalkCUDA<GPUUsage>::TemporalRandomWalkCUDA(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound):
    ITemporalRandomWalk<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound)
{
    cuda_device_prop = new cudaDeviceProp();
    cudaGetDeviceProperties(cuda_device_prop, 0);

    this->temporal_graph = new typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
}

#ifdef HAS_CUDA

template<GPUUsageMode GPUUsage>
__global__ void generate_random_walks_kernel(
    WalkSet<GPUUsage>* walk_set,
    typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType* temporal_graph,
    int* start_node_ids,
    RandomPicker<GPUUsage>* edge_picker,
    RandomPicker<GPUUsage>* start_picker,
    curandState* rand_states,
    int max_walk_len,
    const bool is_directed,
    const WalkDirection walk_direction,
    const int num_walks) {

    const int walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    bool should_walk_forward = get_should_walk_forward(walk_direction);

    // Get thread-specific random state
    curandState local_state = rand_states[walk_idx];

    Edge start_edge;
    if (start_node_ids[walk_idx] == -1) {
        start_edge = temporal_graph->get_edge_at_device(
            start_picker,
            &local_state,
            -1,
            should_walk_forward);
    } else {
        start_edge = temporal_graph->get_node_edge_at_device(
            start_node_ids[walk_idx],
            start_picker,
            &local_state,
            -1,
            should_walk_forward
        );
    }

    if (start_edge.i == -1) {
        return;
    }

    int current_node = -1;
    size_t current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;
    auto [start_src, start_dst, start_ts] = start_edge;

    if (is_directed) {
        if (should_walk_forward) {
            walk_set->add_hop(walk_idx, start_src, current_timestamp);
            current_node = start_dst;
        } else {
            walk_set->add_hop(walk_idx, start_dst, current_timestamp);
            current_node = start_src;
        }
    } else {
        const int picked_node = start_node_ids[walk_idx];
        walk_set->add_hop(walk_idx, picked_node, current_timestamp);
        current_node = pick_other_number(start_src, start_dst, picked_node);
    }

    current_timestamp = start_ts;

    while (walk_set->get_walk_len_device(walk_idx) < max_walk_len && current_node != -1) {
        walk_set->add_hop(walk_idx, current_node, current_timestamp);

        auto [picked_src, picked_dst, picked_ts] = temporal_graph->get_node_edge_at_device(
            current_node,
            edge_picker,
            &local_state,
            current_timestamp,
            should_walk_forward
        );

        if (picked_ts == -1) {
            current_node = -1;
            continue;
        }

        if (is_directed) {
            current_node = should_walk_forward ? picked_dst : picked_src;
        } else {
            current_node = pick_other_number(picked_src, picked_dst, current_node);
        }

        current_timestamp = picked_ts;
    }

    // If walking backward, reverse the walk
    if (!should_walk_forward) {
        walk_set->reverse_walk(walk_idx);
    }

    // Update random state
    rand_states[walk_idx] = local_state;
}


template<GPUUsageMode GPUUsage>
HOST WalkSet<GPUUsage> TemporalRandomWalkCUDA<GPUUsage>::get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction) {

    RandomPicker<GPUUsage>* edge_picker = this->get_random_picker(walk_bias)->to_device_ptr();
    RandomPicker<GPUUsage>* start_picker = initial_edge_bias ?
        this->get_random_picker(initial_edge_bias)->to_device_ptr()
        : edge_picker;

    auto repeated_node_ids = repeat_elements<GPUUsage>(this->get_node_ids(), num_walks_per_node);
    auto [grid_dim, block_dim] = get_optimal_launch_params(repeated_node_ids.size(), this->cuda_device_prop);
    shuffle_vector_device<int>(thrust::raw_pointer_cast(repeated_node_ids.data()), repeated_node_ids.size(), grid_dim, block_dim);

    typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType* graph = this->temporal_graph->to_device_ptr();
    WalkSet<GPUUsage> walk_set(repeated_node_ids.size(), max_walk_len);
    WalkSet<GPUUsage>* d_walk_set = walk_set.to_device_ptr();

    curandState* rand_states = get_cuda_rand_states(grid_dim, block_dim);

    generate_random_walks_kernel<<<grid_dim, block_dim>>>(
        d_walk_set,
        graph,
        thrust::raw_pointer_cast(repeated_node_ids.data()),
        edge_picker,
        start_picker,
        rand_states,
        max_walk_len,
        this->is_directed,
        walk_direction,
        repeated_node_ids.size()
    );

    cudaDeviceSynchronize();

    walk_set.copy_from_device(d_walk_set);

    cudaFree(rand_states);
    cudaFree(edge_picker);
    cudaFree(start_picker);
    cudaFree(d_walk_set);

    return walk_set;
}

template<GPUUsageMode GPUUsage>
HOST WalkSet<GPUUsage> TemporalRandomWalkCUDA<GPUUsage>::get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction) {

    RandomPicker<GPUUsage>* edge_picker = this->get_random_picker(walk_bias)->to_device_ptr();
    RandomPicker<GPUUsage>* start_picker = initial_edge_bias ?
        this->get_random_picker(initial_edge_bias)->to_device_ptr()
        : edge_picker;

    auto [grid_dim, block_dim] = get_optimal_launch_params(num_walks_total, this->cuda_device_prop);
    typename SelectVectorType<int, GPUUsage>::type start_node_ids(num_walks_total, -1);

    typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType* graph = this->temporal_graph->to_device_ptr();
    WalkSet<GPUUsage> walk_set(num_walks_total, max_walk_len);
    WalkSet<GPUUsage>* d_walk_set = walk_set.to_device_ptr();

    curandState* rand_states = get_cuda_rand_states(grid_dim, block_dim);

    generate_random_walks_kernel<<<grid_dim, block_dim>>>(
        d_walk_set,
        graph,
        thrust::raw_pointer_cast(start_node_ids.data()),
        edge_picker,
        start_picker,
        rand_states,
        max_walk_len,
        this->is_directed,
        walk_direction,
        num_walks_total
    );

    cudaDeviceSynchronize();

    walk_set.copy_from_device(d_walk_set);

    cudaFree(rand_states);
    cudaFree(edge_picker);
    cudaFree(start_picker);
    cudaFree(d_walk_set);

    return walk_set;
}

template class TemporalRandomWalkCUDA<GPUUsageMode::ON_GPU>;
#endif
