#include "TemporalRandomWalkCUDA.cuh"

#include "../cuda_common/setup.cuh"

template<GPUUsageMode GPUUsage>
HOST TemporalRandomWalkCUDA<GPUUsage>::TemporalRandomWalkCUDA(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound):
    ITemporalRandomWalk<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound)
{
    #ifdef HAS_CUDA
    /**
     * Very important to initiate CudaRandomStates in the entrypoint.
     * Other code will assume it's initialized.
     */
    CudaRandomStates::initialize();
    #endif

    this->temporal_graph = new typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
}

template <GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCUDA<GPUUsage>::clear() {
    this->temporal_graph = new typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType(
        this->is_directed, this->max_time_capacity,
        this->enable_weight_computation, this->timescale_bound);
}

#ifdef HAS_CUDA
template class TemporalRandomWalkCUDA<GPUUsageMode::ON_GPU>;
#endif
