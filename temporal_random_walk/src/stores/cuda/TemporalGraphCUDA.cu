#include "TemporalGraphCUDA.cuh"

#include "../../cuda_common/cuda_config.cuh"

#include "EdgeDataCUDA.cuh"
#include "NodeEdgeIndexCUDA.cuh"
#include "NodeMappingCUDA.cuh"

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCUDA<GPUUsage>::TemporalGraphCUDA(
    const bool directed,
    const int64_t window,
    const bool enable_weight_computation,
    const double timescale_bound)
    : ITemporalGraph<GPUUsage>(directed, window, enable_weight_computation, timescale_bound) {
    this->node_index = new NodeEdgeIndexCUDA<GPUUsage>();
    this->edges = new EdgeDataCUDA<GPUUsage>();
    this->node_mapping = new NodeMappingCUDA<GPUUsage>();
}

template<GPUUsageMode GPUUsage>
HOST void TemporalGraphCUDA<GPUUsage>::add_multiple_edges(const typename ITemporalGraph<GPUUsage>::EdgeVector& new_edges) {
    const size_t start_idx = this->edges->size();
    const size_t new_size = start_idx + new_edges.size();
    this->edges->reserve(new_size);

    // Allocate device vectors for new edge data
    typename ITemporalGraph<GPUUsage>::IntVector sources(new_edges.size());
    typename ITemporalGraph<GPUUsage>::IntVector targets(new_edges.size());
    typename ITemporalGraph<GPUUsage>::Int64TVector timestamps(new_edges.size());

    const auto d_edges = thrust::raw_pointer_cast(new_edges.data());
    const bool is_directed = this->is_directed;

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(new_edges.size()),
        thrust::make_zip_iterator(thrust::make_tuple(
            sources.begin(),
            targets.begin()
        )),
        [d_edges, is_directed] __device__ (size_t i) {
            const Edge& edge = d_edges[i];
            int src, tgt;

            if (!is_directed && edge.u > edge.i) {
                src = edge.i;
                tgt = edge.u;
            } else {
                src = edge.u;
                tgt = edge.i;
            }

            return thrust::make_tuple(src, tgt);
        }
    );

    // Extract timestamps separately
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        new_edges.begin(),
        new_edges.end(),
        timestamps.begin(),
        [] __device__ (const Edge& edge) {
            return edge.ts;
        }
    );

    // Update latest timestamp (need to find max on device)
    int64_t device_max_timestamp = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        timestamps.begin(),
        timestamps.end(),
        INT64_MIN,
        thrust::maximum<int64_t>()
    );

    // Update latest timestamp with the max from the device
    this->latest_timestamp = std::max(this->latest_timestamp, device_max_timestamp);

    // Add edges using device pointers
    this->edges->add_edges(
        thrust::raw_pointer_cast(sources.data()),
        thrust::raw_pointer_cast(targets.data()),
        thrust::raw_pointer_cast(timestamps.data()),
        new_edges.size());

    this->node_mapping->update(this->edges, start_idx, this->edges->size());
    this->sort_and_merge_edges(start_idx);
    this->edges->update_timestamp_groups();

    if (this->time_window > 0) {
        this->delete_old_edges();
    }

    this->node_index->rebuild(this->edges, this->node_mapping, this->is_directed);

    if (this->enable_weight_computation) {
        this->update_temporal_weights();
    }
}

#ifdef HAS_CUDA
template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
