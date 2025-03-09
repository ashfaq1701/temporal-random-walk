#ifndef TEMPORAL_RANDOM_WALK_CUDA_CUH
#define TEMPORAL_RANDOM_WALK_CUDA_CUH

#include "ITemporalRandomWalk.cuh"
#include "../data/enums.h"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

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
    bool is_directed,
    WalkDirection walk_direction,
    int num_walks);
#endif

template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCUDA : public ITemporalRandomWalk<GPUUsage> {
    cudaDeviceProp* cuda_device_prop;

public:

    explicit HOST TemporalRandomWalkCUDA(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND);

    #ifdef HAS_CUDA
    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) override;

    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) override;
    #endif
};

#endif //TEMPORAL_RANDOM_WALK_CUDA_CUH
