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
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

#include "../../random/IndexBasedRandomPicker.cuh"
#include "../../random/WeightBasedRandomPicker.cuh"
#include "../../random/RandomPicker.h"
#include "../../utils/rand_utils.cuh"

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
        curandState* rand_state,
        int64_t timestamp,
        bool forward) const {
    if (this->edges->empty_device()) return Edge{-1, -1, -1};

    const size_t num_groups = this->edges->get_timestamp_group_count_device();
    if (num_groups == 0) return Edge{-1, -1, -1};

    long group_idx;
    if (timestamp != -1) {
        if (forward) {
            const size_t first_group = this->edges->find_group_after_timestamp_device(timestamp);
            const size_t available_groups = num_groups - first_group;
            if (available_groups == 0) return Edge{-1, -1, -1};

            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const auto index = index_picker->pick_random_device(0, static_cast<int>(available_groups), false, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = first_group + index;
            }
            else {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_idx = weight_picker->pick_random_device(
                    this->edges->forward_cumulative_weights_exponential_ptr,
                    this->edges->forward_cumulative_weights_exponential_size,
                    static_cast<int>(first_group),
                    static_cast<int>(num_groups),
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        } else {
            const size_t last_group = this->edges->find_group_before_timestamp_device(timestamp);
            if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

            const size_t available_groups = last_group + 1;
            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const auto index = index_picker->pick_random_device(0, static_cast<int>(available_groups), true, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = last_group - (available_groups - index - 1);
            }
            else {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_idx = weight_picker->pick_random_device(
                    this->edges->backward_cumulative_weights_exponential_ptr,
                    this->edges->backward_cumulative_weights_exponential_size,
                    0,
                    static_cast<int>(last_group + 1),
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
            auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
            const auto index = index_picker->pick_random_device(0, static_cast<int>(num_groups), !forward, rand_state);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_idx = index;
        } else {
            auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
            if (forward) {
                group_idx = weight_picker->pick_random_device(
                    this->edges->forward_cumulative_weights_exponential_ptr,
                    this->edges->forward_cumulative_weights_exponential_size,
                    0,
                    static_cast<int>(num_groups),
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
            else {
                group_idx = weight_picker->pick_random_device(
                    this->edges->backward_cumulative_weights_exponential_ptr,
                    this->edges->backward_cumulative_weights_exponential_size,
                    0,
                    static_cast<int>(num_groups),
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get selected group's boundaries
    auto [group_start, group_end] = this->edges->get_timestamp_group_range_device(group_idx);
    if (group_start == group_end) {
        return Edge{-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_start + generate_random_number_bounded_by_device(static_cast<int>(group_end - group_start), rand_state);
    return Edge {
        this->edges->sources[random_idx],
        this->edges->targets[random_idx],
        this->edges->timestamps[random_idx]
    };
}

template<GPUUsageMode GPUUsage>
DEVICE Edge TemporalGraphCUDA<GPUUsage>::get_node_edge_at_device(
    int node_id,
    RandomPicker<GPUUsage>* picker,
    curandState* rand_state,
    int64_t timestamp,
    bool forward) const {

    const int dense_idx = this->node_mapping->to_dense_device(node_id);
    if (dense_idx < 0) return Edge{-1, -1, -1};

    // Get appropriate node indices based on direction and graph type
    const auto timestamp_group_offsets = forward
        ? this->node_index->outbound_timestamp_group_offsets_ptr
        : (this->is_directed ? this->node_index->inbound_timestamp_group_offsets_ptr : this->node_index->outbound_timestamp_group_offsets_ptr);

    const auto timestamp_group_indices = forward
        ? this->node_index->outbound_timestamp_group_indices_ptr
        : (this->is_directed ? this->node_index->inbound_timestamp_group_indices_ptr : this->node_index->outbound_timestamp_group_indices_ptr);

    const auto edge_indices = forward
        ? this->node_index->outbound_indices_ptr
        : (this->is_directed ? this->node_index->inbound_indices_ptr : this->node_index->outbound_indices_ptr);

    const auto edge_indices_size = forward
        ? this->node_index->outbound_indices_size
        : (this->is_directed ? this->node_index->inbound_indices_size : this->node_index->outbound_indices_size);

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

    long group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            auto it = cuda::std::upper_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [this, edge_indices](int64_t ts, size_t pos) {
                    return ts < this->edges->timestamps_ptr[edge_indices[pos]];
                });

            // Count available groups after timestamp
            const size_t available = cuda::std::distance(
                it,
                timestamp_group_indices + static_cast<int>(group_end_offset));
            if (available == 0) return Edge{-1, -1, -1};

            const size_t start_pos = it - timestamp_group_indices;
            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const auto index = index_picker->pick_random_device(0, static_cast<int>(available), false, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = start_pos + index;
            }
            else
            {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_pos = weight_picker->pick_random_device(
                    this->node_index->outbound_forward_cumulative_weights_exponential_ptr,
                    this->node_index->outbound_forward_cumulative_weights_exponential_size,
                    static_cast<int>(start_pos),
                    static_cast<int>(group_end_offset),
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        } else {
            // Find first group >= timestamp
            auto it = cuda::std::lower_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [this, edge_indices](size_t pos, int64_t ts) {
                    return this->edges->timestamps_ptr[edge_indices[pos]] < ts;
                });

            const size_t available = cuda::std::distance(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                it);
            if (available == 0) return Edge{-1, -1, -1};

            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const auto index = index_picker->pick_random_device(0, static_cast<int>(available), true, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = (it - timestamp_group_indices) - 1 - (available - index - 1);
            }
            else
            {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_pos = weight_picker->pick_random_device(
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential_ptr
                        : this->node_index->outbound_backward_cumulative_weights_exponential_ptr,
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential_size
                        : this->node_index->outbound_backward_cumulative_weights_exponential_size,
                    static_cast<int>(group_start_offset), // start from node's first group
                    static_cast<int>(it - timestamp_group_indices), // up to and excluding first group >= timestamp
                    rand_state
                );
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return Edge{-1, -1, -1};

        if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
            auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
            const auto index = index_picker->pick_random_device(0, static_cast<int>(num_groups), !forward, rand_state);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_pos = forward
                ? group_start_offset + index
                : group_end_offset - 1 - (num_groups - index - 1);
        }
        else
        {
            auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
            if (forward)
            {
                group_pos = weight_picker->pick_random_device(
                    this->node_index->outbound_forward_cumulative_weights_exponential_ptr,
                    this->node_index->outbound_forward_cumulative_weights_exponential_size,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset),
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
            else
            {
                group_pos = weight_picker->pick_random_device(
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential_ptr
                        : this->node_index->outbound_backward_cumulative_weights_exponential_ptr,
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential_size
                        : this->node_index->outbound_backward_cumulative_weights_exponential_size,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset),
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get edge range for selected group
    const size_t edge_start = timestamp_group_indices[group_pos];
    const size_t edge_end = (group_pos + 1 < group_end_offset)
        ? timestamp_group_indices[group_pos + 1]
        : (forward ? this->node_index->outbound_offsets[dense_idx + 1]
                  : (this->is_directed ? this->node_index->inbound_offsets[dense_idx + 1]
                                : this->node_index->outbound_offsets[dense_idx + 1]));

    // Validate range before random selection
    if (edge_start >= edge_end || edge_start >= edge_indices_size || edge_end > edge_indices_size) {
        return Edge{-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by_device(static_cast<int>(edge_end - edge_start), rand_state)];

    return Edge {
        this->edges->sources[edge_idx],
        this->edges->targets[edge_idx],
        this->edges->timestamps[edge_idx]
    };
}

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCUDA<GPUUsage>* TemporalGraphCUDA<GPUUsage>::to_device_ptr() {
    // Allocate device memory for the TemporalGraphCUDA object
    TemporalGraphCUDA<GPUUsage>* device_temporal_graph;
    cudaMalloc(&device_temporal_graph, sizeof(TemporalGraphCUDA<GPUUsage>));

    // Create device copies of the component objects
    if (this->node_index) {
        this->node_index_device = this->node_index->to_device_ptr();
    }

    if (this->edges) {
        this->edges_device = this->edges->to_device_ptr();
    }

    if (this->node_mapping) {
        this->node_mapping_device = this->node_mapping->to_device_ptr();
    }

    // Copy the object with device pointers to the device
    cudaMemcpy(device_temporal_graph, this, sizeof(TemporalGraphCUDA<GPUUsage>), cudaMemcpyHostToDevice);

    return device_temporal_graph;
}

template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
