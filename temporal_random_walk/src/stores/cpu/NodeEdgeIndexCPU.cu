#include "NodeEdgeIndexCPU.cuh"

/**
 * START METHODS FOR REBUILD
*/
template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::populate_dense_ids(
        const IEdgeData<GPUUsage>* edges,
        const INodeMapping<GPUUsage>* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets)
{
    for (size_t i = 0; i < edges->size(); i++)
    {
        dense_sources[i] = mapping->to_dense(edges->sources[i]);
        dense_targets[i] = mapping->to_dense(edges->targets[i]);
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_offsets(
    const IEdgeData<GPUUsage>* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    bool is_directed)
{
    // First pass: count edges per node
    for (size_t i = 0; i < edges->size(); i++) {
        int src_idx = dense_sources[i];
        int tgt_idx = dense_targets[i];

        ++this->outbound_offsets[src_idx + 1];
        if (is_directed) {
            ++this->inbound_offsets[tgt_idx + 1];
        } else {
            ++this->outbound_offsets[tgt_idx + 1];
        }
    }

    // Calculate prefix sums for edge offsets
    for (size_t i = 1; i <= this->outbound_offsets.size(); i++) {
        this->outbound_offsets[i] += this->outbound_offsets[i-1];
        if (is_directed) {
            this->inbound_offsets[i] += this->inbound_offsets[i-1];
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_indices(
    const IEdgeData<GPUUsage>* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    typename INodeEdgeIndex<GPUUsage>::EdgeWithEndpointTypeVector& outbound_edge_indices_buffer,
    bool is_directed)
{
    auto edges_size = edges->size();

    for (size_t i = 0; i < edges_size; i++)
    {
        size_t outbound_index = is_directed ? i : i * 2;
        outbound_edge_indices_buffer[outbound_index] = EdgeWithEndpointType{i, true};

        if (is_directed)
        {
            this->inbound_indices[i] = i;
        }
        else
        {
            outbound_edge_indices_buffer[outbound_index + 1] = EdgeWithEndpointType{i, false};
        }
    }

    auto buffer_size = is_directed ? edges_size : edges_size * 2;

    std::stable_sort(outbound_edge_indices_buffer.begin(),
                     outbound_edge_indices_buffer.begin() + buffer_size,
        [&dense_sources, &dense_targets](const EdgeWithEndpointType& a, const EdgeWithEndpointType& b) {
            const int node_a = a.is_source ? dense_sources[a.edge_id] : dense_targets[a.edge_id];
            const int node_b = b.is_source ? dense_sources[b.edge_id] : dense_targets[b.edge_id];
            return node_a < node_b;
        });

    if (is_directed)
    {
        std::stable_sort(this->inbound_indices.begin(),
                         this->inbound_indices.begin() + edges_size,
            [&dense_targets](size_t a, size_t b) {
                return dense_targets[a] < dense_targets[b];
            });
    }

    for (size_t i = 0; i < buffer_size; i++)
    {
        this->outbound_indices[i] = outbound_edge_indices_buffer[i].edge_id;
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_timestamp_offsets(
    const IEdgeData<GPUUsage>* edges,
    size_t num_nodes,
    bool is_directed)
{
    // Third pass: count timestamp groups
    typename INodeEdgeIndex<GPUUsage>::SizeVector outbound_group_count(num_nodes);
    typename INodeEdgeIndex<GPUUsage>::SizeVector inbound_group_count;
    if (is_directed) {
        inbound_group_count.resize(num_nodes);
    }

    for (size_t node = 0; node < num_nodes; node++) {
        size_t start = this->outbound_offsets[node];
        size_t end = this->outbound_offsets[node + 1];

        if (start < end) {
            outbound_group_count[node] = 1;  // First group
            for (size_t i = start + 1; i < end; ++i) {
                if (edges->timestamps[this->outbound_indices[i]] !=
                    edges->timestamps[this->outbound_indices[i-1]]) {
                    ++outbound_group_count[node];
                    }
            }
        }

        if (is_directed) {
            start = this->inbound_offsets[node];
            end = this->inbound_offsets[node + 1];

            if (start < end) {
                inbound_group_count[node] = 1;  // First group
                for (size_t i = start + 1; i < end; ++i) {
                    if (edges->timestamps[this->inbound_indices[i]] !=
                        edges->timestamps[this->inbound_indices[i-1]]) {
                        ++inbound_group_count[node];
                        }
                }
            }
        }
    }

    // Calculate prefix sums for group offsets
    for (size_t i = 0; i < num_nodes; i++) {
        this->outbound_timestamp_group_offsets[i + 1] = this->outbound_timestamp_group_offsets[i] + outbound_group_count[i];
        if (is_directed) {
            this->inbound_timestamp_group_offsets[i + 1] = this->inbound_timestamp_group_offsets[i] + inbound_group_count[i];
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_timestamp_indices(
    const IEdgeData<GPUUsage>* edges,
    size_t num_nodes,
    bool is_directed)
{
    // Final pass: fill group indices
    for (size_t node = 0; node < num_nodes; node++) {
        size_t start = this->outbound_offsets[node];
        size_t end = this->outbound_offsets[node + 1];
        size_t group_pos = this->outbound_timestamp_group_offsets[node];

        if (start < end) {
            this->outbound_timestamp_group_indices[group_pos++] = start;
            for (size_t i = start + 1; i < end; ++i) {
                if (edges->timestamps[this->outbound_indices[i]] !=
                    edges->timestamps[this->outbound_indices[i-1]]) {
                    this->outbound_timestamp_group_indices[group_pos++] = i;
                    }
            }
        }

        if (is_directed) {
            start = this->inbound_offsets[node];
            end = this->inbound_offsets[node + 1];
            group_pos = this->inbound_timestamp_group_offsets[node];

            if (start < end) {
                this->inbound_timestamp_group_indices[group_pos++] = start;
                for (size_t i = start + 1; i < end; ++i) {
                    if (edges->timestamps[this->inbound_indices[i]] !=
                        edges->timestamps[this->inbound_indices[i-1]]) {
                        this->inbound_timestamp_group_indices[group_pos++] = i;
                        }
                }
            }
        }
    }
}
/**
 * END METHODS FOR REBUILD
*/

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_temporal_weights(
    const IEdgeData<GPUUsage>* edges,
    double timescale_bound,
    size_t num_nodes)
{
    // Process each node
    for (size_t node = 0; node < num_nodes; node++) {
        // Outbound weights
        const auto& outbound_offsets = this->get_timestamp_offset_vector(true, false);
        const size_t out_start = outbound_offsets[node];
        const size_t out_end = outbound_offsets[node + 1];

        if (out_start < out_end) {
            const size_t first_group_start = this->outbound_timestamp_group_indices[out_start];
            const size_t last_group_start = this->outbound_timestamp_group_indices[out_end - 1];
            const int64_t min_ts = edges->timestamps[this->outbound_indices[first_group_start]];
            const int64_t max_ts = edges->timestamps[this->outbound_indices[last_group_start]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                timescale_bound / time_diff : 1.0;

            double forward_sum = 0.0;
            double backward_sum = 0.0;

            // Calculate weights and sums
            for (size_t pos = out_start; pos < out_end; ++pos) {
                const size_t edge_start = this->outbound_timestamp_group_indices[pos];
                const int64_t group_ts = edges->timestamps[this->outbound_indices[edge_start]];

                const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                const double forward_scaled = timescale_bound > 0 ?
                    time_diff_forward * time_scale : time_diff_forward;
                const double backward_scaled = timescale_bound > 0 ?
                    time_diff_backward * time_scale : time_diff_backward;

                const double forward_weight = exp(forward_scaled);
                this->outbound_forward_cumulative_weights_exponential[pos] = forward_weight;
                forward_sum += forward_weight;

                const double backward_weight = exp(backward_scaled);
                this->outbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                backward_sum += backward_weight;
            }

            // Normalize and compute cumulative sums
            double forward_cumsum = 0.0, backward_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                this->outbound_forward_cumulative_weights_exponential[pos] /= forward_sum;
                this->outbound_backward_cumulative_weights_exponential[pos] /= backward_sum;

                forward_cumsum += this->outbound_forward_cumulative_weights_exponential[pos];
                backward_cumsum += this->outbound_backward_cumulative_weights_exponential[pos];

                this->outbound_forward_cumulative_weights_exponential[pos] = forward_cumsum;
                this->outbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
            }
        }

        // Inbound weights
        if (!this->inbound_offsets.empty()) {
            const auto& inbound_group_offsets = this->get_timestamp_offset_vector(false, true);
            const size_t in_start = inbound_group_offsets[node];
            const size_t in_end = inbound_group_offsets[node + 1];

            if (in_start < in_end) {
                const size_t first_group_start = this->inbound_timestamp_group_indices[in_start];
                const size_t last_group_start = this->inbound_timestamp_group_indices[in_end - 1];
                const int64_t min_ts = edges->timestamps[this->inbound_indices[first_group_start]];
                const int64_t max_ts = edges->timestamps[this->inbound_indices[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                    timescale_bound / time_diff : 1.0;

                double backward_sum = 0.0;

                // Calculate weights and sum
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    const size_t edge_start = this->inbound_timestamp_group_indices[pos];
                    const int64_t group_ts = edges->timestamps[this->inbound_indices[edge_start]];

                    const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                    const double backward_scaled = timescale_bound > 0 ?
                        time_diff_backward * time_scale : time_diff_backward;

                    const double backward_weight = exp(backward_scaled);
                    this->inbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                    backward_sum += backward_weight;
                }

                // Normalize and compute cumulative sum
                double backward_cumsum = 0.0;
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    this->inbound_backward_cumulative_weights_exponential[pos] /= backward_sum;
                    backward_cumsum += this->inbound_backward_cumulative_weights_exponential[pos];
                    this->inbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
                }
            }
        }
    }
}

template class NodeEdgeIndexCPU<GPUUsageMode::ON_CPU>;
