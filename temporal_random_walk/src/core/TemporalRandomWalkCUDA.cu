#include "TemporalRandomWalkCUDA.cuh"

#include "../utils/utils.h"

#include "../cuda_common/setup.cuh"

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

    WalkSet<GPUUsage> walk_set(repeated_node_ids.size(), max_walk_len);
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

    WalkSet<GPUUsage> walk_set(num_walks_total, max_walk_len);
    return walk_set;
}

template class TemporalRandomWalkCUDA<GPUUsageMode::ON_GPU>;
#endif
