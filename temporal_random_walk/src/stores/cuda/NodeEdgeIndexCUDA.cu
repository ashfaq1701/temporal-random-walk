#include "NodeEdgeIndexCUDA.cuh"
#include "NodeMappingCUDA.cuh"

#include "../../cuda_common/cuda_config.cuh"

#ifdef HAS_CUDA

#include <thrust/sequence.h>
#include <thrust/sort.h>

/**
 * START METHODS FOR REBUILD
*/
template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::populate_dense_ids(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    const typename INodeEdgeIndex<GPUUsage>::NodeMappingType* mapping,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets) {

    const int* d_sparse_to_dense = thrust::raw_pointer_cast(mapping->sparse_to_dense.data());
    const auto sparse_to_dense_size = mapping->sparse_to_dense.size();

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        edges->sources.begin(),
        edges->sources.end(),
        dense_sources.begin(),
        [d_sparse_to_dense, sparse_to_dense_size] __host__ __device__ (const int id) {
            return to_dense(d_sparse_to_dense, id, sparse_to_dense_size);
        });

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        edges->targets.begin(),
        edges->targets.end(),
        dense_targets.begin(),
        [d_sparse_to_dense, sparse_to_dense_size] __host__ __device__ (const int id) {
            return to_dense(d_sparse_to_dense, id, sparse_to_dense_size);
        });
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_node_edge_offsets(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    bool is_directed) {

    const size_t num_edges = edges->size();

    size_t* d_outbound_offsets_ptr = thrust::raw_pointer_cast(this->outbound_offsets.data());
    size_t* d_inbound_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_offsets.data()) : nullptr;
    const int* d_src_ptr = thrust::raw_pointer_cast(dense_sources.data());
    const int* d_tgt_ptr = thrust::raw_pointer_cast(dense_targets.data());

    auto counter_device_lambda = [
        d_outbound_offsets_ptr, d_inbound_offsets_ptr,
        d_src_ptr, d_tgt_ptr, is_directed] __device__ (const size_t i) {
        const int src_idx = d_src_ptr[i];
        const int tgt_idx = d_tgt_ptr[i];

        atomicAdd(reinterpret_cast<unsigned int *>(&d_outbound_offsets_ptr[src_idx + 1]), 1);
        if (is_directed) {
            atomicAdd(reinterpret_cast<unsigned int *>(&d_inbound_offsets_ptr[tgt_idx + 1]), 1);
        } else {
            atomicAdd(reinterpret_cast<unsigned int *>(&d_outbound_offsets_ptr[tgt_idx + 1]), 1);
        }
    };

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        counter_device_lambda);

    // Calculate prefix sums for edge offsets
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        this->outbound_offsets.begin() + 1,
        this->outbound_offsets.end(),
        this->outbound_offsets.begin() + 1
    );

    if (is_directed) {
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            this->inbound_offsets.begin() + 1,
            this->inbound_offsets.end(),
            this->inbound_offsets.begin() + 1
        );
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_node_edge_indices(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    typename INodeEdgeIndex<GPUUsage>::EdgeWithEndpointTypeVector& outbound_edge_indices_buffer,
    bool is_directed)
{
    auto edges_size = edges->size();
    auto buffer_size = is_directed ? edges_size : edges_size * 2;

    // Initialize outbound_edge_indices_buffer
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(edges_size),
        [outbound_buffer = thrust::raw_pointer_cast(outbound_edge_indices_buffer.data()),
         is_directed] __device__ (const size_t i) {
            size_t outbound_index = is_directed ? i : i * 2;
            outbound_buffer[outbound_index] = EdgeWithEndpointType{i, true};

            if (!is_directed) {
                outbound_buffer[outbound_index + 1] = EdgeWithEndpointType{i, false};
            }
        }
    );

    // Initialize inbound_indices for directed graphs
    if (is_directed) {
        thrust::sequence(
            DEVICE_EXECUTION_POLICY,
            this->inbound_indices.begin(),
            this->inbound_indices.begin() + edges_size
        );
    }

    // Sort outbound_edge_indices_buffer by node ID
    thrust::stable_sort(
        DEVICE_EXECUTION_POLICY,
        outbound_edge_indices_buffer.begin(),
        outbound_edge_indices_buffer.begin() + buffer_size,
        [d_sources = thrust::raw_pointer_cast(dense_sources.data()),
         d_targets = thrust::raw_pointer_cast(dense_targets.data())] __device__ (
            const EdgeWithEndpointType& a, const EdgeWithEndpointType& b) {
            const int node_a = a.is_source ? d_sources[a.edge_id] : d_targets[a.edge_id];
            const int node_b = b.is_source ? d_sources[b.edge_id] : d_targets[b.edge_id];
            return node_a < node_b;
        }
    );

    // Sort inbound_indices for directed graphs
    if (is_directed) {
        thrust::stable_sort(
            DEVICE_EXECUTION_POLICY,
            this->inbound_indices.begin(),
            this->inbound_indices.begin() + edges_size,
            [d_targets = thrust::raw_pointer_cast(dense_targets.data())] __device__ (
                size_t a, size_t b) {
                return d_targets[a] < d_targets[b];
            }
        );
    }

    // Extract edge_id from outbound_edge_indices_buffer to outbound_indices
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        outbound_edge_indices_buffer.begin(),
        outbound_edge_indices_buffer.begin() + buffer_size,
        this->outbound_indices.begin(),
        [] __device__ (const EdgeWithEndpointType& edge_with_type) {
            return edge_with_type.edge_id;
        }
    );
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_node_timestamp_offsets(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    size_t num_nodes,
    bool is_directed) {
    typename SelectVectorType<size_t, GPUUsage>::type d_outbound_group_count(num_nodes, 0);
    typename SelectVectorType<size_t, GPUUsage>::type d_inbound_group_count;
    if (is_directed) {
        d_inbound_group_count.resize(num_nodes, 0);
    }

    const int64_t* d_timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
    size_t* d_outbound_offsets_ptr = thrust::raw_pointer_cast(this->outbound_offsets.data());
    size_t* d_inbound_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_offsets.data()) : nullptr;
    size_t* d_outbound_indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
    size_t* d_inbound_indices_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_indices.data()) : nullptr;


    size_t* d_outbound_group_count_ptr = thrust::raw_pointer_cast(d_outbound_group_count.data());
    size_t* d_inbound_group_count_ptr = is_directed ? thrust::raw_pointer_cast(d_inbound_group_count.data()) : nullptr;


    auto fill_timestamp_groups_device_lambda = [d_outbound_offsets_ptr, d_inbound_offsets_ptr,
                d_outbound_indices_ptr, d_inbound_indices_ptr,
                d_outbound_group_count_ptr, d_inbound_group_count_ptr,
                d_timestamps_ptr, is_directed] __device__ (const size_t node) {
        size_t start = d_outbound_offsets_ptr[node];
        size_t end = d_outbound_offsets_ptr[node + 1];

        if (start < end) {
            d_outbound_group_count_ptr[node] = 1; // First group
            for (size_t i = start + 1; i < end; ++i) {
                if (d_timestamps_ptr[d_outbound_indices_ptr[i]] !=
                    d_timestamps_ptr[d_outbound_indices_ptr[i - 1]]) {
                    atomicAdd(reinterpret_cast<unsigned int *>(&d_outbound_group_count_ptr[node]), 1);
                    }
            }
        }

        if (is_directed) {
            start = d_inbound_offsets_ptr[node];
            end = d_inbound_offsets_ptr[node + 1];

            if (start < end) {
                d_inbound_group_count_ptr[node] = 1; // First group
                for (size_t i = start + 1; i < end; ++i) {
                    if (d_timestamps_ptr[d_inbound_indices_ptr[i]] !=
                        d_timestamps_ptr[d_inbound_indices_ptr[i - 1]]) {
                        atomicAdd(reinterpret_cast<unsigned int *>(&d_inbound_group_count_ptr[node]), 1);
                        }
                }
            }
        }
    };

    thrust::for_each(
    DEVICE_EXECUTION_POLICY,
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_nodes),
    fill_timestamp_groups_device_lambda);

    // Calculate prefix sums for group offsets
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_outbound_group_count.begin(),
        d_outbound_group_count.end(),
        thrust::make_permutation_iterator(
            this->outbound_timestamp_group_offsets.begin() + 1,
            thrust::make_counting_iterator<size_t>(0)
        )
    );

    if (is_directed) {
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            d_inbound_group_count.begin(),
            d_inbound_group_count.end(),
            thrust::make_permutation_iterator(
                this->inbound_timestamp_group_offsets.begin() + 1,
                thrust::make_counting_iterator<size_t>(0)
            )
        );
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_node_timestamp_indices(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    size_t num_nodes,
    bool is_directed) {

    const int64_t* d_timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
    size_t* d_outbound_offsets_ptr = thrust::raw_pointer_cast(this->outbound_offsets.data());
    size_t* d_inbound_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_offsets.data()) : nullptr;

    // Get raw pointers for filling group indices
    size_t* d_outbound_indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
    size_t* d_inbound_indices_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_indices.data()) : nullptr;

    size_t* d_outbound_group_indices_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_indices.data());
    size_t* d_inbound_group_indices_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_timestamp_group_indices.data()) : nullptr;
    const size_t* d_outbound_group_offsets_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_offsets.data());
    const size_t* d_inbound_group_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_timestamp_group_offsets.data()) : nullptr;

    // Fill group indices
    auto fill_node_time_group_lambda = [d_outbound_offsets_ptr, d_inbound_offsets_ptr,
            d_outbound_indices_ptr, d_inbound_indices_ptr,
            d_outbound_group_offsets_ptr, d_inbound_group_offsets_ptr,
            d_outbound_group_indices_ptr, d_inbound_group_indices_ptr,
            d_timestamps_ptr, is_directed] __host__ __device__ (const size_t node) {
        size_t start = d_outbound_offsets_ptr[node];
        size_t end = d_outbound_offsets_ptr[node + 1];
        size_t group_pos = d_outbound_group_offsets_ptr[node];

        if (start < end) {
            d_outbound_group_indices_ptr[group_pos++] = start;
            for (size_t i = start + 1; i < end; ++i) {
                if (d_timestamps_ptr[d_outbound_indices_ptr[i]] !=
                    d_timestamps_ptr[d_outbound_indices_ptr[i-1]]) {
                    d_outbound_group_indices_ptr[group_pos++] = i;
                }
            }
        }

        if (is_directed) {
            start = d_inbound_offsets_ptr[node];
            end = d_inbound_offsets_ptr[node + 1];
            group_pos = d_inbound_group_offsets_ptr[node];

            if (start < end) {
                d_inbound_group_indices_ptr[group_pos++] = start;
                for (size_t i = start + 1; i < end; ++i) {
                    if (d_timestamps_ptr[d_inbound_indices_ptr[i]] !=
                        d_timestamps_ptr[d_inbound_indices_ptr[i-1]]) {
                        d_inbound_group_indices_ptr[group_pos++] = i;
                    }
                }
            }
        }
    };

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_nodes),
        fill_node_time_group_lambda);
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_temporal_weights(
    const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
    double timescale_bound,
    size_t num_nodes) {

    // Process outbound weights
    {
        const auto& outbound_offsets = this->get_timestamp_offset_vector(true, false);
        typename SelectVectorType<double, GPUUsage>::type forward_weights(this->outbound_timestamp_group_indices.size());
        typename SelectVectorType<double, GPUUsage>::type backward_weights(this->outbound_timestamp_group_indices.size());

        auto timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(outbound_offsets.data());
        auto forward_weights_ptr = thrust::raw_pointer_cast(forward_weights.data());
        auto backward_weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate initial weights
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_nodes),
            [
                timestamps_ptr,
                indices_ptr,
                group_indices_ptr,
                offsets_ptr,
                forward_weights_ptr,
                backward_weights_ptr,
                timescale_bound
            ] __host__ __device__ (size_t node) {
                const size_t out_start = offsets_ptr[node];
                const size_t out_end = offsets_ptr[node + 1];

                if (out_start < out_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = group_indices_ptr[out_start];
                    const size_t last_group_start = group_indices_ptr[out_end - 1];
                    const int64_t min_ts = timestamps_ptr[indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                        timescale_bound / time_diff : 1.0;

                    double forward_sum = 0.0;
                    double backward_sum = 0.0;

                    // Calculate weights for each group
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        const size_t edge_start = group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[indices_ptr[edge_start]];

                        const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                        const double forward_scaled = timescale_bound > 0 ?
                            time_diff_forward * time_scale : time_diff_forward;
                        const double backward_scaled = timescale_bound > 0 ?
                            time_diff_backward * time_scale : time_diff_backward;

                        const double forward_weight = exp(forward_scaled);
                        forward_weights_ptr[pos] = forward_weight;
                        forward_sum += forward_weight;

                        const double backward_weight = exp(backward_scaled);
                        backward_weights_ptr[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sums
                    double forward_cumsum = 0.0, backward_cumsum = 0.0;
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        forward_weights_ptr[pos] /= forward_sum;
                        backward_weights_ptr[pos] /= backward_sum;

                        forward_cumsum += forward_weights_ptr[pos];
                        backward_cumsum += backward_weights_ptr[pos];

                        forward_weights_ptr[pos] = forward_cumsum;
                        backward_weights_ptr[pos] = backward_cumsum;
                    }
                }
            }
        );

        // Copy results back
        this->outbound_forward_cumulative_weights_exponential = forward_weights;
        this->outbound_backward_cumulative_weights_exponential = backward_weights;
    }

    // Process inbound weights if directed
    if (!this->inbound_offsets.empty()) {
        const auto& inbound_offsets = this->get_timestamp_offset_vector(false, true);
        typename SelectVectorType<double, GPUUsage>::type backward_weights(this->inbound_timestamp_group_indices.size());

        auto timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->inbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->inbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(inbound_offsets.data());
        auto weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate weights
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_nodes),
            [
                timestamps_ptr,
                indices_ptr,
                group_indices_ptr,
                offsets_ptr,
                weights_ptr,
                timescale_bound
            ] __host__ __device__ (size_t node) {
                const size_t in_start = offsets_ptr[node];
                const size_t in_end = offsets_ptr[node + 1];

                if (in_start < in_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = group_indices_ptr[in_start];
                    const size_t last_group_start = group_indices_ptr[in_end - 1];
                    const int64_t min_ts = timestamps_ptr[indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                        timescale_bound / time_diff : 1.0;

                    // Calculate weights
                    double backward_sum = 0.0;

                    // Calculate weights and sum in single pass
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        const size_t edge_start = group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[indices_ptr[edge_start]];

                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                        const double backward_scaled = timescale_bound > 0 ?
                            time_diff_backward * time_scale : time_diff_backward;

                        const double backward_weight = exp(backward_scaled);
                        weights_ptr[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sum
                    double backward_cumsum = 0.0;
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        weights_ptr[pos] /= backward_sum;
                        backward_cumsum += weights_ptr[pos];
                        weights_ptr[pos] = backward_cumsum;
                    }
                }
            }
        );

        // Copy results
        this->inbound_backward_cumulative_weights_exponential = backward_weights;
    }
}

template<GPUUsageMode GPUUsage>
HOST NodeEdgeIndexCUDA<GPUUsage>* NodeEdgeIndexCUDA<GPUUsage>::to_device_ptr() {
    return nullptr;
}

template class NodeEdgeIndexCUDA<GPUUsageMode::ON_GPU>;
#endif
