#include "TemporalGraphCUDA.cuh"

#include "../../cuda_common/cuda_config.cuh"

#include "EdgeDataCUDA.cuh"
#include "NodeEdgeIndexCUDA.cuh"
#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/detail/sequence.inl>
#include <thrust/detail/sort.inl>
#endif

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

#ifdef HAS_CUDA
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

template<GPUUsageMode GPUUsage>
void TemporalGraphCUDA<GPUUsage>::sort_and_merge_edges(const size_t start_idx) {
    if (start_idx >= this->edges->size()) return;

    // Create index array
    typename ITemporalGraph<GPUUsage>::SizeVector indices(this->edges->size() - start_idx);
    thrust::sequence(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end(),
        start_idx
    );

    // Sort indices based on timestamps
    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges->timestamps.data());
    thrust::sort(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end(),
        [timestamps_ptr] __host__ __device__ (const size_t i, const size_t j) {
            return timestamps_ptr[i] < timestamps_ptr[j];
        }
    );

    // Create temporary vectors for sorted data
    typename ITemporalGraph<GPUUsage>::IntVector sorted_sources(this->edges->size() - start_idx);
    typename ITemporalGraph<GPUUsage>::IntVector sorted_targets(this->edges->size() - start_idx);
    typename ITemporalGraph<GPUUsage>::Int64TVector sorted_timestamps(this->edges->size() - start_idx);

    // Apply permutation using gather
    thrust::gather(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end(),
        this->edges->sources.begin(),
        sorted_sources.begin()
    );
    thrust::gather(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end(),
        this->edges->targets.begin(),
        sorted_targets.begin()
    );
    thrust::gather(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end(),
        this->edges->timestamps.begin(),
        sorted_timestamps.begin()
    );

    // Copy sorted data back
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_sources.begin(),
        sorted_sources.end(),
        this->edges->sources.begin() + start_idx
    );
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_targets.begin(),
        sorted_targets.end(),
        this->edges->targets.begin() + start_idx
    );
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_timestamps.begin(),
        sorted_timestamps.end(),
        this->edges->timestamps.begin() + start_idx
    );

    // Handle merging if we have existing edges
    if (start_idx > 0) {
        typename ITemporalGraph<GPUUsage>::IntVector merged_sources(this->edges->size());
        typename ITemporalGraph<GPUUsage>::IntVector merged_targets(this->edges->size());
        typename ITemporalGraph<GPUUsage>::Int64TVector merged_timestamps(this->edges->size());

        // Create iterators for merge operation
        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges->sources.begin(),
            this->edges->targets.begin(),
            this->edges->timestamps.begin()
        ));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges->sources.begin() + start_idx,
            this->edges->targets.begin() + start_idx,
            this->edges->timestamps.begin() + start_idx
        ));
        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges->sources.begin() + start_idx,
            this->edges->targets.begin() + start_idx,
            this->edges->timestamps.begin() + start_idx
        ));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges->sources.end(),
            this->edges->targets.end(),
            this->edges->timestamps.end()
        ));
        auto result = thrust::make_zip_iterator(thrust::make_tuple(
            merged_sources.begin(),
            merged_targets.begin(),
            merged_timestamps.begin()
        ));

        // Merge based on timestamps
        thrust::merge(
            DEVICE_EXECUTION_POLICY,
            first1, last1,
            first2, last2,
            result,
            [] __host__ __device__ (const thrust::tuple<int, int, int64_t>& a,
                                   const thrust::tuple<int, int, int64_t>& b) {
                return thrust::get<2>(a) <= thrust::get<2>(b);
            }
        );

        // Move merged results back
        this->edges->sources.swap(merged_sources);
        this->edges->targets.swap(merged_targets);
        this->edges->timestamps.swap(merged_timestamps);
    }
}

template<GPUUsageMode GPUUsage>
void TemporalGraphCUDA<GPUUsage>::delete_old_edges() {
    if (this->time_window <= 0 || this->edges->empty()) return;

    const int64_t cutoff_time = this->latest_timestamp - this->time_window;

    auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        this->edges->timestamps.begin(),
        this->edges->timestamps.end(),
        cutoff_time
    );
    if (it == this->edges->timestamps.begin()) return;

    const int delete_count = static_cast<int>(it - this->edges->timestamps.begin());
    const size_t remaining = this->edges->size() - delete_count;

    // Create bool vector for tracking nodes with edges
    typename SelectVectorType<bool, GPUUsage>::type has_edges(this->node_mapping->sparse_to_dense.size(), false);
    bool* has_edges_ptr = thrust::raw_pointer_cast(has_edges.data());

    if (remaining > 0) {
        // Move edges using thrust::copy
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            this->edges->sources.begin() + delete_count,
            this->edges->sources.end(),
            this->edges->sources.begin()
        );
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            this->edges->targets.begin() + delete_count,
            this->edges->targets.end(),
            this->edges->targets.begin()
        );
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            this->edges->timestamps.begin() + delete_count,
            this->edges->timestamps.end(),
            this->edges->timestamps.begin()
        );

        // Mark nodes with edges in parallel
        const int* sources_ptr = thrust::raw_pointer_cast(this->edges->sources.data());
        const int* targets_ptr = thrust::raw_pointer_cast(this->edges->targets.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(remaining),
            [sources_ptr, targets_ptr, has_edges_ptr] __host__ __device__ (const size_t i) {
                has_edges_ptr[sources_ptr[i]] = true;
                has_edges_ptr[targets_ptr[i]] = true;
            }
        );
    }

    this->edges->resize(remaining);

    bool* d_is_deleted = thrust::raw_pointer_cast(this->node_mapping->is_deleted.data());
    const auto is_deleted_size = this->node_mapping->is_deleted.size();

    // Mark deleted nodes in parallel
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(has_edges.size()),
        [has_edges_ptr, d_is_deleted, is_deleted_size] __host__ __device__ (const size_t i) {
            if (!has_edges_ptr[i]) {
                mark_node_deleted(d_is_deleted, static_cast<int>(i), is_deleted_size);
            }
        }
    );

    // Update data structures
    this->edges->update_timestamp_groups();
    this->node_mapping->update(this->edges, 0, this->edges->size());
    this->node_index->rebuild(this->edges, this->node_mapping, this->is_directed);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const {
    if (this->edges->empty()) return 0;

    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        this->edges->unique_timestamps.begin(),
        this->edges->unique_timestamps.end(),
        timestamp);
    return it - this->edges->unique_timestamps.begin();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const {
    if (this->edges->empty()) return 0;

    auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        this->edges->unique_timestamps.begin(),
        this->edges->unique_timestamps.end(),
        timestamp);
    return this->edges->unique_timestamps.end() - it;
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = this->node_mapping->to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->is_directed ?
        this->node_index->inbound_timestamp_group_offsets : this->node_index->outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->is_directed ?
        this->node_index->inbound_timestamp_group_indices : this->node_index->outbound_timestamp_group_indices;
    const auto& edge_indices = this->is_directed ?
        this->node_index->inbound_indices : this->node_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges->timestamps.data());
    const unsigned long* edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    // Binary search on group indices
    auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const size_t group_pos, const int64_t ts)
        {
            return timestamps_ptr[edge_indices_ptr[group_pos]] < ts;
        });

    return thrust::distance(timestamp_group_indices.begin() + static_cast<int>(group_start), it);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = this->node_mapping->to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->node_index->outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->node_index->outbound_timestamp_group_indices;
    const auto& edge_indices = this->node_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges->timestamps.data());
    const unsigned long* edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    // Binary search on group indices
    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const int64_t ts, const size_t group_pos)
        {
            return ts < timestamps_ptr[edge_indices_ptr[group_pos]];
        });

    return thrust::distance(it, timestamp_group_indices.begin() + static_cast<int>(group_end));
}

template<GPUUsageMode GPUUsage>
DEVICE Edge TemporalGraphCUDA<GPUUsage>::get_edge_at_device(
    RandomPicker<GPUUsage>* picker,
    int64_t timestamp,
    bool forward) {
    return Edge{-1, -1, -1};
}

template<GPUUsageMode GPUUsage>
DEVICE Edge TemporalGraphCUDA<GPUUsage>::get_node_edge_at_device(
    int node_id,
    RandomPicker<GPUUsage>* picker,
    int64_t timestamp,
    bool forward) const {
    return Edge{-1, -1, -1};
}

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCUDA<GPUUsage>* TemporalGraphCUDA<GPUUsage>::to_device_ptr() {
    return nullptr;
}

template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
