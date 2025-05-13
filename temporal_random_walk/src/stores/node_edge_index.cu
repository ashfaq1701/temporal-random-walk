#include "node_edge_index.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

#include <omp.h>
#include <cmath>
#include <algorithm>
#include "../utils/omp_utils.cuh"
#include "../common/parallel_algorithms.cuh"
#include "../common/cuda_config.cuh"

/**
 * Common Functions
 */

HOST void node_edge_index::clear(NodeEdgeIndexStore *node_edge_index) {
    // Clear edge CSR structures
    clear_memory(&node_edge_index->outbound_offsets, node_edge_index->use_gpu);
    node_edge_index->outbound_offsets_size = 0;

    clear_memory(&node_edge_index->outbound_indices, node_edge_index->use_gpu);
    node_edge_index->outbound_indices_size = 0;

    clear_memory(&node_edge_index->outbound_timestamp_group_offsets, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_offsets_size = 0;

    clear_memory(&node_edge_index->outbound_timestamp_group_indices, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_indices_size = 0;

    // Clear inbound structures
    clear_memory(&node_edge_index->inbound_offsets, node_edge_index->use_gpu);
    node_edge_index->inbound_offsets_size = 0;

    clear_memory(&node_edge_index->inbound_indices, node_edge_index->use_gpu);
    node_edge_index->inbound_indices_size = 0;

    clear_memory(&node_edge_index->inbound_timestamp_group_offsets, node_edge_index->use_gpu);
    node_edge_index->inbound_timestamp_group_offsets_size = 0;

    clear_memory(&node_edge_index->inbound_timestamp_group_indices, node_edge_index->use_gpu);
    node_edge_index->inbound_timestamp_group_indices_size = 0;

    // Clear temporal weights
    clear_memory(&node_edge_index->outbound_forward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->outbound_forward_cumulative_weights_exponential_size = 0;

    clear_memory(&node_edge_index->outbound_backward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->outbound_backward_cumulative_weights_exponential_size = 0;

    clear_memory(&node_edge_index->inbound_backward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->inbound_backward_cumulative_weights_exponential_size = 0;
}

HOST DEVICE SizeRange node_edge_index::get_edge_range(const NodeEdgeIndexStore *node_edge_index, const int dense_node_id,
                                            const bool forward, const bool is_directed) {
    if (is_directed) {
        const size_t *offsets = forward ? node_edge_index->outbound_offsets : node_edge_index->inbound_offsets;
        size_t offsets_size = forward ? node_edge_index->outbound_offsets_size : node_edge_index->inbound_offsets_size;

        if (dense_node_id < 0 || dense_node_id >= offsets_size - 1) {
            return SizeRange{0, 0};
        }

        const size_t start = offsets[dense_node_id];
        const size_t end = offsets[dense_node_id + 1];

        return SizeRange{start, end};
    } else {
        if (dense_node_id < 0 || dense_node_id >= node_edge_index->outbound_offsets_size - 1) {
            return SizeRange{0, 0};
        }

        const size_t start = node_edge_index->outbound_offsets[dense_node_id];
        const size_t end = node_edge_index->outbound_offsets[dense_node_id + 1];

        return SizeRange{start, end};
    }
}

HOST DEVICE SizeRange node_edge_index::get_timestamp_group_range(const NodeEdgeIndexStore *node_edge_index,
                                                       const int dense_node_id, const size_t group_idx,
                                                       const bool forward, const bool is_directed) {
    const size_t *group_offsets = nullptr;
    size_t group_offsets_size = 0;
    const size_t *group_indices = nullptr;
    const size_t *edge_offsets = nullptr;

    if (is_directed && !forward) {
        group_offsets = node_edge_index->inbound_timestamp_group_offsets;
        group_offsets_size = node_edge_index->inbound_timestamp_group_offsets_size;
        group_indices = node_edge_index->inbound_timestamp_group_indices;
        edge_offsets = node_edge_index->inbound_offsets;
    } else {
        group_offsets = node_edge_index->outbound_timestamp_group_offsets;
        group_offsets_size = node_edge_index->outbound_timestamp_group_offsets_size;
        group_indices = node_edge_index->outbound_timestamp_group_indices;
        edge_offsets = node_edge_index->outbound_offsets;
    }

    if (dense_node_id < 0 || dense_node_id >= group_offsets_size - 1) {
        return SizeRange{0, 0};
    }

    const size_t node_group_start = group_offsets[dense_node_id];
    const size_t node_group_end = group_offsets[dense_node_id + 1];

    const size_t num_groups = node_group_end - node_group_start;
    if (group_idx >= num_groups) {
        return SizeRange{0, 0};
    }

    const size_t group_start_idx = node_group_start + group_idx;
    const size_t group_start = group_indices[group_start_idx];

    // Group end is either next group's start or node's edge range end
    size_t group_end = 0;
    if (group_idx == num_groups - 1) {
        group_end = edge_offsets[dense_node_id + 1];
    } else {
        group_end = group_indices[group_start_idx + 1];
    }

    return SizeRange{group_start, group_end};
}

HOST DEVICE MemoryView<size_t> node_edge_index::get_timestamp_offset_vector(const NodeEdgeIndexStore *node_edge_index,
                                                                  const bool forward, const bool is_directed) {
    if (is_directed && !forward) {
        return MemoryView<size_t>{
            node_edge_index->inbound_timestamp_group_offsets,
            node_edge_index->inbound_timestamp_group_offsets_size
        };
    } else {
        return MemoryView<size_t>{
            node_edge_index->outbound_timestamp_group_offsets,
            node_edge_index->outbound_timestamp_group_offsets_size
        };
    }
}

HOST DEVICE size_t node_edge_index::get_timestamp_group_count(const NodeEdgeIndexStore *node_edge_index, const int dense_node_id,
                                                    const bool forward, const bool is_directed) {
    // Get the appropriate timestamp offset vector
    MemoryView<size_t> offsets_block = get_timestamp_offset_vector(node_edge_index, forward, is_directed);
    const size_t *offsets = offsets_block.data;
    size_t offsets_size = offsets_block.size;

    // Check if the node ID is valid
    if (dense_node_id < 0 || dense_node_id >= offsets_size - 1) {
        return 0;
    }

    // Get start and end offsets for the node
    const size_t start = offsets[dense_node_id];
    const size_t end = offsets[dense_node_id + 1];

    return end - start;
}

/**
 * Rebuild related functions
 */

HOST void node_edge_index::allocate_node_edge_offsets(NodeEdgeIndexStore *node_edge_index, const size_t node_index_capacity,
                                            const bool is_directed) {
    allocate_memory(&node_edge_index->outbound_offsets, node_index_capacity + 1, node_edge_index->use_gpu);
    node_edge_index->outbound_offsets_size = node_index_capacity + 1;
    fill_memory(node_edge_index->outbound_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                node_edge_index->use_gpu);

    allocate_memory(&node_edge_index->outbound_timestamp_group_offsets, node_index_capacity + 1,
                    node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_offsets_size = node_index_capacity + 1;
    fill_memory(node_edge_index->outbound_timestamp_group_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                node_edge_index->use_gpu);

    // For directed graphs, also allocate inbound structures
    if (is_directed) {
        allocate_memory(&node_edge_index->inbound_offsets, node_index_capacity + 1, node_edge_index->use_gpu);
        node_edge_index->inbound_offsets_size = node_index_capacity + 1;
        fill_memory(node_edge_index->inbound_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                    node_edge_index->use_gpu);

        allocate_memory(&node_edge_index->inbound_timestamp_group_offsets, node_index_capacity + 1,
                        node_edge_index->use_gpu);
        node_edge_index->inbound_timestamp_group_offsets_size = node_index_capacity + 1;
        fill_memory(node_edge_index->inbound_timestamp_group_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                    node_edge_index->use_gpu);
    }
}

HOST void node_edge_index::allocate_node_edge_indices(NodeEdgeIndexStore *node_edge_index, const bool is_directed) {
    size_t num_outbound_edges = 0;

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // For GPU memory, we need to copy the value back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_outbound_edges,
            node_edge_index->outbound_offsets + (node_edge_index->outbound_offsets_size - 1),
            sizeof(size_t),
            cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        // For CPU memory, we can access it directly
        num_outbound_edges = node_edge_index->outbound_offsets[node_edge_index->outbound_offsets_size - 1];
    }

    // Allocate memory for outbound indices
    allocate_memory(&node_edge_index->outbound_indices, num_outbound_edges, node_edge_index->use_gpu);
    node_edge_index->outbound_indices_size = num_outbound_edges;

    // For directed graphs, also allocate inbound indices
    if (is_directed) {
        size_t num_inbound_edges = 0;

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_inbound_edges,
                node_edge_index->inbound_offsets + (node_edge_index->inbound_offsets_size - 1),
                sizeof(size_t),
                cudaMemcpyDeviceToHost));
        } else
        #endif
        {
            num_inbound_edges = node_edge_index->inbound_offsets[node_edge_index->inbound_offsets_size - 1];
        }

        allocate_memory(&node_edge_index->inbound_indices, num_inbound_edges, node_edge_index->use_gpu);
        node_edge_index->inbound_indices_size = num_inbound_edges;
    }
}

HOST void node_edge_index::allocate_node_timestamp_indices(NodeEdgeIndexStore *node_edge_index, const bool is_directed) {
    size_t num_outbound_groups = 0;

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // For GPU memory, we need to copy the value back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_outbound_groups,
            node_edge_index->outbound_timestamp_group_offsets + (node_edge_index->outbound_timestamp_group_offsets_size
                - 1),
            sizeof(size_t),
            cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        // For CPU memory, we can access it directly
        num_outbound_groups = node_edge_index->outbound_timestamp_group_offsets[
            node_edge_index->outbound_timestamp_group_offsets_size - 1];
    }

    // Allocate memory for outbound timestamp group indices
    allocate_memory(&node_edge_index->outbound_timestamp_group_indices, num_outbound_groups, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_indices_size = num_outbound_groups;

    // For directed graphs, also allocate inbound timestamp group indices
    if (is_directed) {
        size_t num_inbound_groups = 0;

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_inbound_groups,
                node_edge_index->inbound_timestamp_group_offsets + (node_edge_index->
                    inbound_timestamp_group_offsets_size - 1),
                sizeof(size_t),
                cudaMemcpyDeviceToHost));
        } else
        #endif
        {
            num_inbound_groups = node_edge_index->inbound_timestamp_group_offsets[
                node_edge_index->inbound_timestamp_group_offsets_size - 1];
        }

        allocate_memory(&node_edge_index->inbound_timestamp_group_indices, num_inbound_groups,
                        node_edge_index->use_gpu);
        node_edge_index->inbound_timestamp_group_indices_size = num_inbound_groups;
    }
}

/**
 * Std implementations
 */
HOST void node_edge_index::compute_node_edge_offsets_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const bool is_directed
) {
    const size_t num_edges = edge_data->timestamps_size;

    auto* outbound_offsets = node_edge_index->outbound_offsets;
    auto* inbound_offsets  = node_edge_index->inbound_offsets;
    const auto* sources    = edge_data->sources;
    const auto* targets    = edge_data->targets;

    const size_t offset_size = node_edge_index->outbound_offsets_size;

    // Step 1: Zero out offset arrays
    std::fill_n(outbound_offsets, offset_size, 0);
    if (is_directed) {
        std::fill_n(inbound_offsets, node_edge_index->inbound_offsets_size, 0);
    }

    // Step 2: Count edge occurrences (use atomic to avoid collisions)
    #pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
        const int src_idx = sources[i];
        const int tgt_idx = targets[i];

        #pragma omp atomic
        outbound_offsets[src_idx + 1]++;

        if (is_directed) {
            #pragma omp atomic
            inbound_offsets[tgt_idx + 1]++;
        } else {
            #pragma omp atomic
            outbound_offsets[tgt_idx + 1]++;
        }
    }

    // Step 3: Inclusive scan over offsets[1..]
    parallel_inclusive_scan(outbound_offsets + 1, offset_size - 1);

    if (is_directed) {
        parallel_inclusive_scan(inbound_offsets + 1, node_edge_index->inbound_offsets_size - 1);
    }
}

HOST void node_edge_index::compute_node_edge_indices_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    EdgeWithEndpointType* outbound_edge_indices_buffer,
    const bool is_directed
) {
    const size_t edges_size = edge_data->timestamps_size;
    const size_t buffer_size = is_directed ? edges_size : edges_size * 2;

    const auto* sources = edge_data->sources;
    const auto* targets = edge_data->targets;

    // Step 1: Fill outbound_edge_indices_buffer
    #pragma omp parallel for
    for (size_t i = 0; i < edges_size; ++i) {
        size_t outbound_index = is_directed ? i : i * 2;
        outbound_edge_indices_buffer[outbound_index] = EdgeWithEndpointType{static_cast<long>(i), true};

        if (!is_directed) {
            outbound_edge_indices_buffer[outbound_index + 1] = EdgeWithEndpointType{static_cast<long>(i), false};
        }
    }

    // Step 2: Fill inbound_indices for directed graphs
    if (is_directed) {
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            node_edge_index->inbound_indices[i] = i;
        }
    }

    // Step 3: Stable sort outbound_edge_indices_buffer by node ID (source or target depending on flag)
    parallel::stable_sort(
        outbound_edge_indices_buffer,
        outbound_edge_indices_buffer + buffer_size,
        [sources, targets](const EdgeWithEndpointType& a, const EdgeWithEndpointType& b) {
            const int node_a = a.is_source ? sources[a.edge_id] : targets[a.edge_id];
            const int node_b = b.is_source ? sources[b.edge_id] : targets[b.edge_id];
            return node_a < node_b;
        }
    );

    // Step 4: Stable sort inbound_indices by target node (if directed)
    if (is_directed) {
        parallel::stable_sort(
            node_edge_index->inbound_indices,
            node_edge_index->inbound_indices + edges_size,
            [targets](const size_t a, const size_t b) {
                return targets[a] < targets[b];
            }
        );
    }

    // Step 5: Extract edge_id from buffer to outbound_indices
    #pragma omp parallel for
    for (size_t i = 0; i < buffer_size; ++i) {
        node_edge_index->outbound_indices[i] = outbound_edge_indices_buffer[i].edge_id;
    }
}

HOST void node_edge_index::compute_node_timestamp_offsets_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const size_t node_count,
    const bool is_directed
) {
    std::vector<size_t> outbound_group_count(node_count, 0);
    std::vector<size_t> inbound_group_count;

    if (is_directed) {
        inbound_group_count.resize(node_count, 0);
    }

    const int64_t* timestamps = edge_data->timestamps;
    const size_t* outbound_offsets = node_edge_index->outbound_offsets;
    const size_t* outbound_indices = node_edge_index->outbound_indices;
    const size_t* inbound_offsets = node_edge_index->inbound_offsets;
    const size_t* inbound_indices = node_edge_index->inbound_indices;

    size_t* outbound_ts_group_offsets = node_edge_index->outbound_timestamp_group_offsets;
    size_t* inbound_ts_group_offsets = node_edge_index->inbound_timestamp_group_offsets;

    // Step 1: Count timestamp groups in parallel
    #pragma omp parallel for
    for (size_t node = 0; node < node_count; ++node) {
        // Outbound groups
        size_t start = outbound_offsets[node];
        size_t end = outbound_offsets[node + 1];

        if (start < end) {
            outbound_group_count[node] = 1;
            for (size_t i = start + 1; i < end; ++i) {
                if (timestamps[outbound_indices[i]] != timestamps[outbound_indices[i - 1]]) {
                    ++outbound_group_count[node];
                }
            }
        }

        // Inbound groups (if directed)
        if (is_directed) {
            start = inbound_offsets[node];
            end = inbound_offsets[node + 1];

            if (start < end) {
                inbound_group_count[node] = 1;
                for (size_t i = start + 1; i < end; ++i) {
                    if (timestamps[inbound_indices[i]] != timestamps[inbound_indices[i - 1]]) {
                        ++inbound_group_count[node];
                    }
                }
            }
        }
    }

    // Step 2: Compute exclusive prefix sums
    parallel_exclusive_scan(
        outbound_group_count.data(),
        outbound_ts_group_offsets,
        node_count
    );

    if (is_directed) {
        parallel_exclusive_scan(
            inbound_group_count.data(),
            inbound_ts_group_offsets,
            node_count
        );
    }
}

HOST void node_edge_index::compute_node_timestamp_indices_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const size_t node_index_capacity,
    const bool is_directed
) {
    const auto timestamps = edge_data->timestamps;

    const auto outbound_offsets = node_edge_index->outbound_offsets;
    const auto outbound_indices = node_edge_index->outbound_indices;
    const auto outbound_ts_group_offsets = node_edge_index->outbound_timestamp_group_offsets;
    const auto outbound_ts_group_indices = node_edge_index->outbound_timestamp_group_indices;

    const auto inbound_offsets = node_edge_index->inbound_offsets;
    const auto inbound_indices = node_edge_index->inbound_indices;
    const auto inbound_ts_group_offsets = node_edge_index->inbound_timestamp_group_offsets;
    const auto inbound_ts_group_indices = node_edge_index->inbound_timestamp_group_indices;

    // Parallel over nodes
    #pragma omp parallel for
    for (size_t node = 0; node < node_index_capacity; ++node) {
        // Outbound timestamp group indices
        size_t start = outbound_offsets[node];
        size_t end = outbound_offsets[node + 1];
        size_t group_pos = outbound_ts_group_offsets[node];

        if (start < end) {
            outbound_ts_group_indices[group_pos++] = start;
            for (size_t i = start + 1; i < end; ++i) {
                const size_t curr_id = outbound_indices[i];
                size_t prev_id = outbound_indices[i - 1];
                if (timestamps[curr_id] != timestamps[prev_id]) {
                    outbound_ts_group_indices[group_pos++] = i;
                }
            }
        }

        // Inbound timestamp group indices (if directed)
        if (is_directed) {
            start = inbound_offsets[node];
            end = inbound_offsets[node + 1];
            group_pos = inbound_ts_group_offsets[node];

            if (start < end) {
                inbound_ts_group_indices[group_pos++] = start;
                for (size_t i = start + 1; i < end; ++i) {
                    const size_t curr_id = inbound_indices[i];
                    size_t prev_id = inbound_indices[i - 1];
                    if (timestamps[curr_id] != timestamps[prev_id]) {
                        inbound_ts_group_indices[group_pos++] = i;
                    }
                }
            }
        }
    }
}

HOST void node_edge_index::update_temporal_weights_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const double timescale_bound
) {
    const size_t node_index_capacity = node_edge_index->outbound_offsets_size - 1;
    const size_t outbound_groups_size = node_edge_index->outbound_timestamp_group_indices_size;

    // Resize memory for outbound weights
    resize_memory(
        &node_edge_index->outbound_forward_cumulative_weights_exponential,
        node_edge_index->outbound_forward_cumulative_weights_exponential_size,
        outbound_groups_size,
        node_edge_index->use_gpu
    );
    node_edge_index->outbound_forward_cumulative_weights_exponential_size = outbound_groups_size;

    resize_memory(
        &node_edge_index->outbound_backward_cumulative_weights_exponential,
        node_edge_index->outbound_backward_cumulative_weights_exponential_size,
        outbound_groups_size,
        node_edge_index->use_gpu
    );
    node_edge_index->outbound_backward_cumulative_weights_exponential_size = outbound_groups_size;

    const bool is_directed = node_edge_index->inbound_offsets_size > 0;

    if (is_directed) {
        const size_t inbound_groups_size = node_edge_index->inbound_timestamp_group_indices_size;
        resize_memory(
            &node_edge_index->inbound_backward_cumulative_weights_exponential,
            node_edge_index->inbound_backward_cumulative_weights_exponential_size,
            inbound_groups_size,
            node_edge_index->use_gpu
        );
        node_edge_index->inbound_backward_cumulative_weights_exponential_size = inbound_groups_size;
    }

    // Parallel over all nodes
    #pragma omp parallel for
    for (size_t node = 0; node < node_index_capacity; ++node) {
        // === Outbound ===
        auto outbound_offsets = get_timestamp_offset_vector(node_edge_index, true, false);
        const size_t out_start = outbound_offsets.data[node];
        const size_t out_end = outbound_offsets.data[node + 1];

        if (out_start < out_end) {
            const auto* ts_group_indices = node_edge_index->outbound_timestamp_group_indices;
            const auto* edge_indices = node_edge_index->outbound_indices;
            auto* f_weights = node_edge_index->outbound_forward_cumulative_weights_exponential;
            auto* b_weights = node_edge_index->outbound_backward_cumulative_weights_exponential;
            const auto* timestamps = edge_data->timestamps;

            const int64_t min_ts = timestamps[edge_indices[ts_group_indices[out_start]]];
            const int64_t max_ts = timestamps[edge_indices[ts_group_indices[out_end - 1]]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

            double forward_sum = 0.0, backward_sum = 0.0;

            for (size_t pos = out_start; pos < out_end; ++pos) {
                const size_t edge_start = ts_group_indices[pos];
                const int64_t group_ts = timestamps[edge_indices[edge_start]];

                const double f_scaled = (timescale_bound > 0) ? static_cast<double>(max_ts - group_ts) * time_scale : static_cast<double>(max_ts - group_ts);
                const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);

                const double fw = std::exp(f_scaled);
                const double bw = std::exp(b_scaled);

                f_weights[pos] = fw;
                b_weights[pos] = bw;
                forward_sum += fw;
                backward_sum += bw;
            }

            double f_cumsum = 0.0, b_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                f_weights[pos] /= forward_sum;
                b_weights[pos] /= backward_sum;
                f_cumsum += f_weights[pos];
                b_cumsum += b_weights[pos];
                f_weights[pos] = f_cumsum;
                b_weights[pos] = b_cumsum;
            }
        }

        // === Inbound ===
        if (is_directed) {
            auto inbound_offsets = get_timestamp_offset_vector(node_edge_index, false, true);
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            if (in_start < in_end) {
                const auto* ts_group_indices = node_edge_index->inbound_timestamp_group_indices;
                const auto* edge_indices = node_edge_index->inbound_indices;
                auto* b_weights = node_edge_index->inbound_backward_cumulative_weights_exponential;
                const auto* timestamps = edge_data->timestamps;

                const int64_t min_ts = timestamps[edge_indices[ts_group_indices[in_start]]];
                const int64_t max_ts = timestamps[edge_indices[ts_group_indices[in_end - 1]]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

                double backward_sum = 0.0;

                for (size_t pos = in_start; pos < in_end; ++pos) {
                    const size_t edge_start = ts_group_indices[pos];
                    const int64_t group_ts = timestamps[edge_indices[edge_start]];
                    const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);
                    const double bw = std::exp(b_scaled);
                    b_weights[pos] = bw;
                    backward_sum += bw;
                }

                double b_cumsum = 0.0;
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    b_weights[pos] /= backward_sum;
                    b_cumsum += b_weights[pos];
                    b_weights[pos] = b_cumsum;
                }
            }
        }
    }
}

/**
 * Cuda implementations
 */
#ifdef HAS_CUDA

HOST void node_edge_index::compute_node_edge_offsets_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    bool is_directed
) {
    const size_t num_edges = edge_data->timestamps_size;

    // Get raw pointers to work with
    size_t *outbound_offsets_ptr = node_edge_index->outbound_offsets;
    size_t *inbound_offsets_ptr = is_directed ? node_edge_index->inbound_offsets : nullptr;
    int *src_ptr = edge_data->sources;
    int *tgt_ptr = edge_data->targets;

    // Count edges per node using atomics
    auto counter_device_lambda = [
                outbound_offsets_ptr, inbound_offsets_ptr,
                src_ptr, tgt_ptr, is_directed] DEVICE (const size_t i) {
        const int src_idx = src_ptr[i];
        const int tgt_idx = tgt_ptr[i];

        atomicAdd(reinterpret_cast<unsigned int *>(&outbound_offsets_ptr[src_idx + 1]), 1);
        if (is_directed) {
            atomicAdd(reinterpret_cast<unsigned int *>(&inbound_offsets_ptr[tgt_idx + 1]), 1);
        } else {
            atomicAdd(reinterpret_cast<unsigned int *>(&outbound_offsets_ptr[tgt_idx + 1]), 1);
        }
    };

    // Process all edges in parallel
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        counter_device_lambda);
    CUDA_KERNEL_CHECK("After thrust for_each in compute_node_edge_offsets_cuda");

    // Calculate prefix sums for outbound edge offsets
    thrust::device_ptr<size_t> d_outbound_offsets(outbound_offsets_ptr);
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_outbound_offsets + 1,
        d_outbound_offsets + static_cast<long>(node_edge_index->outbound_offsets_size),
        d_outbound_offsets + 1
    );
    CUDA_KERNEL_CHECK("After thrust inclusive_scan outbound in compute_node_edge_offsets_cuda");

    // Calculate prefix sums for inbound edge offsets (if directed)
    if (is_directed) {
        thrust::device_ptr<size_t> d_inbound_offsets(inbound_offsets_ptr);
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            d_inbound_offsets + 1,
            d_inbound_offsets + static_cast<long>(node_edge_index->inbound_offsets_size),
            d_inbound_offsets + 1
        );
        CUDA_KERNEL_CHECK("After thrust inclusive_scan inbound in compute_node_edge_offsets_cuda");
    }
}

HOST void node_edge_index::compute_node_edge_indices_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    EdgeWithEndpointType *outbound_edge_indices_buffer,
    bool is_directed
) {
    const size_t edges_size = edge_data->timestamps_size;
    const size_t buffer_size = is_directed ? edges_size : edges_size * 2;

    // Initialize outbound_edge_indices_buffer with edge IDs
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(edges_size),
        [outbound_edge_indices_buffer, is_directed] DEVICE (const size_t i) {
            size_t outbound_index = is_directed ? i : i * 2;
            outbound_edge_indices_buffer[outbound_index] = EdgeWithEndpointType{static_cast<long>(i), true};

            if (!is_directed) {
                outbound_edge_indices_buffer[outbound_index + 1] = EdgeWithEndpointType
                        {static_cast<long>(i), false};
            }
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each initialize buffer in compute_node_edge_indices_cuda");

    // Initialize inbound_indices for directed graphs
    if (is_directed) {
        thrust::device_ptr<size_t> d_inbound_indices(node_edge_index->inbound_indices);
        thrust::sequence(
            DEVICE_EXECUTION_POLICY,
            d_inbound_indices,
            d_inbound_indices + static_cast<long>(edges_size)
        );
        CUDA_KERNEL_CHECK("After thrust sequence in compute_node_edge_indices_cuda");
    }

    // Wrap buffer with device pointer for sorting
    const thrust::device_ptr<EdgeWithEndpointType> d_buffer(outbound_edge_indices_buffer);

    auto sources = edge_data->sources;
    auto targets = edge_data->targets;

    // Sort outbound_edge_indices_buffer by node ID
    thrust::stable_sort(
        DEVICE_EXECUTION_POLICY,
        d_buffer,
        d_buffer + static_cast<long>(buffer_size),
        [sources, targets] DEVICE (
    const EdgeWithEndpointType &a, const EdgeWithEndpointType &b) {
            const int node_a = a.is_source ? sources[a.edge_id] : targets[a.edge_id];
            const int node_b = b.is_source ? sources[b.edge_id] : targets[b.edge_id];
            return node_a < node_b;
        }
    );
    CUDA_KERNEL_CHECK("After thrust stable_sort outbound in compute_node_edge_indices_cuda");

    // Sort inbound_indices for directed graphs
    if (is_directed) {
        const thrust::device_ptr<size_t> d_inbound_indices(node_edge_index->inbound_indices);
        thrust::stable_sort(
            DEVICE_EXECUTION_POLICY,
            d_inbound_indices,
            d_inbound_indices + static_cast<long>(edges_size),
            [targets] DEVICE (const size_t a, const size_t b) {
                return targets[a] < targets[b];
            }
        );
        CUDA_KERNEL_CHECK("After thrust stable_sort inbound in compute_node_edge_indices_cuda");
    }

    // Extract edge_id from buffer to outbound_indices
    const thrust::device_ptr<size_t> d_outbound_indices(node_edge_index->outbound_indices);
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_buffer,
        d_buffer + static_cast<long>(buffer_size),
        d_outbound_indices,
        [] DEVICE (const EdgeWithEndpointType &edge_with_type) {
            return edge_with_type.edge_id;
        }
    );
    CUDA_KERNEL_CHECK("After thrust transform outbound in compute_node_edge_indices_cuda");
}

HOST void node_edge_index::compute_node_timestamp_offsets_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    const size_t node_count,
    bool is_directed
) {
    // Allocate device memory for temporary arrays to count groups per node
    size_t *d_outbound_group_count = nullptr;
    size_t *d_inbound_group_count = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_outbound_group_count, node_count * sizeof(size_t)));
    CUDA_CHECK_AND_CLEAR(cudaMemset(d_outbound_group_count, 0, node_count * sizeof(size_t)));

    if (is_directed) {
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_inbound_group_count, node_count * sizeof(size_t)));
        CUDA_CHECK_AND_CLEAR(cudaMemset(d_inbound_group_count, 0, node_count * sizeof(size_t)));
    }

    // Get raw pointers for data access in kernel
    int64_t *timestamps_ptr = edge_data->timestamps;
    size_t *outbound_offsets_ptr = node_edge_index->outbound_offsets;
    size_t *inbound_offsets_ptr = is_directed ? node_edge_index->inbound_offsets : nullptr;
    size_t *outbound_indices_ptr = node_edge_index->outbound_indices;
    size_t *inbound_indices_ptr = is_directed ? node_edge_index->inbound_indices : nullptr;

    size_t *outbound_group_count_ptr = d_outbound_group_count;
    size_t *inbound_group_count_ptr = d_inbound_group_count;

    // Fill timestamp groups counts
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(node_count),
        [outbound_offsets_ptr, inbound_offsets_ptr,
            outbound_indices_ptr, inbound_indices_ptr,
            outbound_group_count_ptr, inbound_group_count_ptr,
            timestamps_ptr, is_directed] DEVICE (const size_t node) {
            // Outbound groups
            size_t start = outbound_offsets_ptr[node];
            size_t end = outbound_offsets_ptr[node + 1];

            if (start < end) {
                outbound_group_count_ptr[node] = 1; // First group always exists

                for (size_t i = start + 1; i < end; ++i) {
                    if (timestamps_ptr[outbound_indices_ptr[i]] !=
                        timestamps_ptr[outbound_indices_ptr[i - 1]]) {
                        atomicAdd(reinterpret_cast<unsigned int *>(&outbound_group_count_ptr[node]), 1);
                    }
                }
            }

            // Inbound groups for directed graphs
            if (is_directed) {
                start = inbound_offsets_ptr[node];
                end = inbound_offsets_ptr[node + 1];

                if (start < end) {
                    inbound_group_count_ptr[node] = 1; // First group always exists

                    for (size_t i = start + 1; i < end; ++i) {
                        if (timestamps_ptr[inbound_indices_ptr[i]] !=
                            timestamps_ptr[inbound_indices_ptr[i - 1]]) {
                            atomicAdd(reinterpret_cast<unsigned int *>(&inbound_group_count_ptr[node]), 1);
                        }
                    }
                }
            }
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each fill groups in compute_node_timestamp_offsets_cuda");

    // Create device pointers for prefix scan
    thrust::device_ptr<size_t> d_outbound_group_count_thrust(d_outbound_group_count);
    thrust::device_ptr<size_t>
            d_outbound_timestamp_group_offsets(node_edge_index->outbound_timestamp_group_offsets);

    // First element should be 0
    CUDA_CHECK_AND_CLEAR(cudaMemset(node_edge_index->outbound_timestamp_group_offsets, 0, sizeof(size_t)));

    // Calculate prefix sum for outbound group offsets
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_outbound_group_count_thrust,
        d_outbound_group_count_thrust + static_cast<long>(node_count),
        d_outbound_timestamp_group_offsets + 1
    );
    CUDA_KERNEL_CHECK("After thrust inclusive_scan outbound in compute_node_timestamp_offsets_cuda");

    // Inbound processing for directed graphs
    if (is_directed) {
        thrust::device_ptr<size_t> d_inbound_group_count_thrust(d_inbound_group_count);
        thrust::device_ptr<size_t> d_inbound_timestamp_group_offsets(
            node_edge_index->inbound_timestamp_group_offsets);

        // First element should be 0
        CUDA_CHECK_AND_CLEAR(cudaMemset(node_edge_index->inbound_timestamp_group_offsets, 0, sizeof(size_t)));

        // Calculate prefix sum for inbound group offsets
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            d_inbound_group_count_thrust,
            d_inbound_group_count_thrust + static_cast<long>(node_count),
            d_inbound_timestamp_group_offsets + 1
        );
        CUDA_KERNEL_CHECK("After thrust inclusive_scan inbound in compute_node_timestamp_offsets_cuda");
    }

    // Free temporary memory
    CUDA_CHECK_AND_CLEAR(cudaFree(d_outbound_group_count));
    if (is_directed) {
        CUDA_CHECK_AND_CLEAR(cudaFree(d_inbound_group_count));
    }
}

HOST void node_edge_index::compute_node_timestamp_indices_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    const size_t node_index_capacity,
    bool is_directed
) {
    // Get raw pointers for data access in kernel
    int64_t *timestamps_ptr = edge_data->timestamps;
    size_t *outbound_offsets_ptr = node_edge_index->outbound_offsets;
    size_t *inbound_offsets_ptr = is_directed ? node_edge_index->inbound_offsets : nullptr;
    size_t *outbound_indices_ptr = node_edge_index->outbound_indices;
    size_t *inbound_indices_ptr = is_directed ? node_edge_index->inbound_indices : nullptr;
    size_t *outbound_timestamp_group_indices_ptr = node_edge_index->outbound_timestamp_group_indices;
    size_t *inbound_timestamp_group_indices_ptr = is_directed
                                                      ? node_edge_index->inbound_timestamp_group_indices
                                                      : nullptr;
    size_t *outbound_timestamp_group_offsets_ptr = node_edge_index->outbound_timestamp_group_offsets;
    size_t *inbound_timestamp_group_offsets_ptr = is_directed
                                                      ? node_edge_index->inbound_timestamp_group_offsets
                                                      : nullptr;

    // Fill timestamp group indices
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(node_index_capacity),
        [outbound_offsets_ptr, inbound_offsets_ptr,
            outbound_indices_ptr, inbound_indices_ptr,
            outbound_timestamp_group_offsets_ptr, inbound_timestamp_group_offsets_ptr,
            outbound_timestamp_group_indices_ptr, inbound_timestamp_group_indices_ptr,
            timestamps_ptr, is_directed] DEVICE (const size_t node) {
            // Outbound timestamp groups
            size_t start = outbound_offsets_ptr[node];
            size_t end = outbound_offsets_ptr[node + 1];
            size_t group_pos = outbound_timestamp_group_offsets_ptr[node];

            if (start < end) {
                // First group always starts at the first edge
                outbound_timestamp_group_indices_ptr[group_pos++] = start;

                for (size_t i = start + 1; i < end; ++i) {
                    if (timestamps_ptr[outbound_indices_ptr[i]] !=
                        timestamps_ptr[outbound_indices_ptr[i - 1]]) {
                        outbound_timestamp_group_indices_ptr[group_pos++] = i;
                    }
                }
            }

            // Inbound timestamp groups for directed graphs
            if (is_directed) {
                start = inbound_offsets_ptr[node];
                end = inbound_offsets_ptr[node + 1];
                group_pos = inbound_timestamp_group_offsets_ptr[node];

                if (start < end) {
                    // First group always starts at the first edge
                    inbound_timestamp_group_indices_ptr[group_pos++] = start;

                    for (size_t i = start + 1; i < end; ++i) {
                        if (timestamps_ptr[inbound_indices_ptr[i]] !=
                            timestamps_ptr[inbound_indices_ptr[i - 1]]) {
                            inbound_timestamp_group_indices_ptr[group_pos++] = i;
                        }
                    }
                }
            }
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each fill indices in compute_node_timestamp_indices_cuda");
}

HOST void node_edge_index::update_temporal_weights_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    double timescale_bound
) {
    // Get the number of nodes and timestamp groups
    const size_t node_index_capacity = node_edge_index->outbound_offsets_size - 1;
    const size_t outbound_groups_size = node_edge_index->outbound_timestamp_group_indices_size;

    // Resize outbound weight arrays
    resize_memory(
        &node_edge_index->outbound_forward_cumulative_weights_exponential,
        node_edge_index->outbound_forward_cumulative_weights_exponential_size,
        outbound_groups_size,
        node_edge_index->use_gpu);
    node_edge_index->outbound_forward_cumulative_weights_exponential_size = outbound_groups_size;

    resize_memory(
        &node_edge_index->outbound_backward_cumulative_weights_exponential,
        node_edge_index->outbound_backward_cumulative_weights_exponential_size,
        outbound_groups_size,
        node_edge_index->use_gpu
    );
    node_edge_index->outbound_backward_cumulative_weights_exponential_size = outbound_groups_size;

    // Resize inbound weights array if directed graph
    if (node_edge_index->inbound_offsets_size > 0) {
        const size_t inbound_groups_size = node_edge_index->inbound_timestamp_group_indices_size;
        resize_memory(
            &node_edge_index->inbound_backward_cumulative_weights_exponential,
            node_edge_index->inbound_backward_cumulative_weights_exponential_size,
            inbound_groups_size,
            node_edge_index->use_gpu);
        node_edge_index->inbound_backward_cumulative_weights_exponential_size = inbound_groups_size;
    }

    // Process outbound weights
    {
        // Get outbound timestamp group offsets
        MemoryView<size_t> outbound_offsets = get_timestamp_offset_vector(node_edge_index, true, false);

        // Allocate temporary device memory for weights
        double *d_forward_weights = nullptr;
        double *d_backward_weights = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_forward_weights, outbound_groups_size * sizeof(double)));
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, outbound_groups_size * sizeof(double)));

        // Get raw pointers for device code
        int64_t *timestamps_ptr = edge_data->timestamps;
        size_t *outbound_indices_ptr = node_edge_index->outbound_indices;
        size_t *outbound_group_indices_ptr = node_edge_index->outbound_timestamp_group_indices;
        size_t *outbound_offsets_ptr = outbound_offsets.data;

        // Calculate weights in parallel for each node
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, outbound_indices_ptr, outbound_group_indices_ptr,
                outbound_offsets_ptr, d_forward_weights, d_backward_weights, timescale_bound]
            DEVICE (const size_t node) {
                const size_t out_start = outbound_offsets_ptr[node];
                const size_t out_end = outbound_offsets_ptr[node + 1];

                if (out_start < out_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = outbound_group_indices_ptr[out_start];
                    const size_t last_group_start = outbound_group_indices_ptr[out_end - 1];
                    const int64_t min_ts = timestamps_ptr[outbound_indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[outbound_indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0)
                                                  ? timescale_bound / time_diff
                                                  : 1.0;

                    double forward_sum = 0.0;
                    double backward_sum = 0.0;

                    // Calculate weights for each group
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        const size_t edge_start = outbound_group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[outbound_indices_ptr[edge_start]];

                        const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                        const double forward_scaled = timescale_bound > 0
                                                          ? time_diff_forward * time_scale
                                                          : time_diff_forward;
                        const double backward_scaled = timescale_bound > 0
                                                           ? time_diff_backward * time_scale
                                                           : time_diff_backward;

                        const double forward_weight = exp(forward_scaled);
                        d_forward_weights[pos] = forward_weight;
                        forward_sum += forward_weight;

                        const double backward_weight = exp(backward_scaled);
                        d_backward_weights[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sums
                    double forward_cumsum = 0.0, backward_cumsum = 0.0;
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        d_forward_weights[pos] /= forward_sum;
                        d_backward_weights[pos] /= backward_sum;

                        forward_cumsum += d_forward_weights[pos];
                        backward_cumsum += d_backward_weights[pos];

                        d_forward_weights[pos] = forward_cumsum;
                        d_backward_weights[pos] = backward_cumsum;
                    }
                }
            }
        );
        CUDA_KERNEL_CHECK("After thrust for_each outbound weights in update_temporal_weights_cuda");

        // Copy results to destination arrays
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            node_edge_index->outbound_forward_cumulative_weights_exponential,
            d_forward_weights,
            outbound_groups_size * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            node_edge_index->outbound_backward_cumulative_weights_exponential,
            d_backward_weights,
            outbound_groups_size * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));

        // Clean up temporary memory
        CUDA_CHECK_AND_CLEAR(cudaFree(d_forward_weights));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_backward_weights));
    }

    // Process inbound weights if directed
    if (node_edge_index->inbound_offsets_size > 0) {
        // Get inbound timestamp group offsets
        MemoryView<size_t> inbound_offsets = get_timestamp_offset_vector(node_edge_index, false, true);
        const size_t inbound_groups_size = node_edge_index->inbound_timestamp_group_indices_size;

        // Allocate temporary device memory for weights
        double *d_backward_weights = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, inbound_groups_size * sizeof(double)));

        // Get raw pointers for device code
        int64_t *timestamps_ptr = edge_data->timestamps;
        size_t *inbound_indices_ptr = node_edge_index->inbound_indices;
        size_t *inbound_group_indices_ptr = node_edge_index->inbound_timestamp_group_indices;
        size_t *inbound_offsets_ptr = inbound_offsets.data;

        // Calculate weights in parallel for each node
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, inbound_indices_ptr, inbound_group_indices_ptr,
                inbound_offsets_ptr, d_backward_weights, timescale_bound]
            DEVICE (const size_t node) {
                const size_t in_start = inbound_offsets_ptr[node];
                const size_t in_end = inbound_offsets_ptr[node + 1];

                if (in_start < in_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = inbound_group_indices_ptr[in_start];
                    const size_t last_group_start = inbound_group_indices_ptr[in_end - 1];
                    const int64_t min_ts = timestamps_ptr[inbound_indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[inbound_indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0)
                                                  ? timescale_bound / time_diff
                                                  : 1.0;

                    double backward_sum = 0.0;

                    // Calculate weights and sum
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        const size_t edge_start = inbound_group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[inbound_indices_ptr[edge_start]];

                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                        const double backward_scaled = timescale_bound > 0
                                                           ? time_diff_backward * time_scale
                                                           : time_diff_backward;

                        const double backward_weight = exp(backward_scaled);
                        d_backward_weights[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sum
                    double backward_cumsum = 0.0;
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        d_backward_weights[pos] /= backward_sum;
                        backward_cumsum += d_backward_weights[pos];
                        d_backward_weights[pos] = backward_cumsum;
                    }
                }
            }
        );
        CUDA_KERNEL_CHECK("After thrust for_each inbound weights in update_temporal_weights_cuda");

        // Copy results to destination array
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            node_edge_index->inbound_backward_cumulative_weights_exponential,
            d_backward_weights,
            inbound_groups_size * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));

        // Clean up temporary memory
        CUDA_CHECK_AND_CLEAR(cudaFree(d_backward_weights));
    }
}

HOST NodeEdgeIndexStore* node_edge_index::to_device_ptr(const NodeEdgeIndexStore *node_edge_index) {
    // Create a new NodeEdgeIndex object on the device
    NodeEdgeIndexStore *device_node_edge_index;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_node_edge_index, sizeof(NodeEdgeIndexStore)));

    // Create a temporary copy to modify for device pointers
    NodeEdgeIndexStore temp_node_edge_index = *node_edge_index;
    temp_node_edge_index.owns_data = false;

    // If already using GPU, just copy the struct with its pointers
    if (!node_edge_index->use_gpu) {
        temp_node_edge_index.owns_data = true;

        // Copy each array to device if it exists
        if (node_edge_index->outbound_offsets) {
            size_t *d_outbound_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_offsets, node_edge_index->outbound_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_offsets, node_edge_index->outbound_offsets, node_edge_index->
                    outbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.outbound_offsets = d_outbound_offsets;
        }

        if (node_edge_index->inbound_offsets) {
            size_t *d_inbound_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_offsets, node_edge_index->inbound_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_offsets, node_edge_index->inbound_offsets, node_edge_index->
                    inbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.inbound_offsets = d_inbound_offsets;
        }

        if (node_edge_index->outbound_indices) {
            size_t *d_outbound_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_indices, node_edge_index->outbound_indices_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_indices, node_edge_index->outbound_indices, node_edge_index->
                    outbound_indices_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.outbound_indices = d_outbound_indices;
        }

        if (node_edge_index->inbound_indices) {
            size_t *d_inbound_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_indices, node_edge_index->inbound_indices_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_indices, node_edge_index->inbound_indices, node_edge_index->
                    inbound_indices_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.inbound_indices = d_inbound_indices;
        }

        if (node_edge_index->outbound_timestamp_group_offsets) {
            size_t *d_outbound_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_timestamp_group_offsets, node_edge_index->
                    outbound_timestamp_group_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_timestamp_group_offsets, node_edge_index->outbound_timestamp_group_offsets,
                    node_edge_index->outbound_timestamp_group_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice
                ));
            temp_node_edge_index.outbound_timestamp_group_offsets = d_outbound_timestamp_group_offsets;
        }

        if (node_edge_index->inbound_timestamp_group_offsets) {
            size_t *d_inbound_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_timestamp_group_offsets, node_edge_index->inbound_timestamp_group_offsets_size
                    * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_timestamp_group_offsets, node_edge_index->inbound_timestamp_group_offsets,
                    node_edge_index->inbound_timestamp_group_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice))
            ;
            temp_node_edge_index.inbound_timestamp_group_offsets = d_inbound_timestamp_group_offsets;
        }

        if (node_edge_index->outbound_timestamp_group_indices) {
            size_t *d_outbound_timestamp_group_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_timestamp_group_indices, node_edge_index->
                    outbound_timestamp_group_indices_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_timestamp_group_indices, node_edge_index->outbound_timestamp_group_indices,
                    node_edge_index->outbound_timestamp_group_indices_size * sizeof(size_t), cudaMemcpyHostToDevice
                ));
            temp_node_edge_index.outbound_timestamp_group_indices = d_outbound_timestamp_group_indices;
        }

        if (node_edge_index->inbound_timestamp_group_indices) {
            size_t *d_inbound_timestamp_group_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_timestamp_group_indices, node_edge_index->inbound_timestamp_group_indices_size
                    * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_timestamp_group_indices, node_edge_index->inbound_timestamp_group_indices,
                    node_edge_index->inbound_timestamp_group_indices_size * sizeof(size_t), cudaMemcpyHostToDevice))
            ;
            temp_node_edge_index.inbound_timestamp_group_indices = d_inbound_timestamp_group_indices;
        }

        if (node_edge_index->outbound_forward_cumulative_weights_exponential) {
            double *d_outbound_forward_weights;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_forward_weights, node_edge_index->
                    outbound_forward_cumulative_weights_exponential_size * sizeof(double)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_forward_weights, node_edge_index->
                    outbound_forward_cumulative_weights_exponential,
                    node_edge_index->outbound_forward_cumulative_weights_exponential_size * sizeof(double),
                    cudaMemcpyHostToDevice));
            temp_node_edge_index.outbound_forward_cumulative_weights_exponential = d_outbound_forward_weights;
        }

        if (node_edge_index->outbound_backward_cumulative_weights_exponential) {
            double *d_outbound_backward_weights;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_backward_weights, node_edge_index->
                    outbound_backward_cumulative_weights_exponential_size * sizeof(double)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_backward_weights, node_edge_index->
                    outbound_backward_cumulative_weights_exponential,
                    node_edge_index->outbound_backward_cumulative_weights_exponential_size * sizeof(double),
                    cudaMemcpyHostToDevice));
            temp_node_edge_index.outbound_backward_cumulative_weights_exponential = d_outbound_backward_weights;
        }

        if (node_edge_index->inbound_backward_cumulative_weights_exponential) {
            double *d_inbound_backward_weights;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_backward_weights, node_edge_index->
                    inbound_backward_cumulative_weights_exponential_size * sizeof(double)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_backward_weights, node_edge_index->
                    inbound_backward_cumulative_weights_exponential,
                    node_edge_index->inbound_backward_cumulative_weights_exponential_size * sizeof(double),
                    cudaMemcpyHostToDevice));
            temp_node_edge_index.inbound_backward_cumulative_weights_exponential = d_inbound_backward_weights;
        }

        // Make sure use_gpu is set to true
        temp_node_edge_index.use_gpu = true;
    }

    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(device_node_edge_index, &temp_node_edge_index, sizeof(NodeEdgeIndexStore), cudaMemcpyHostToDevice
        ));

    temp_node_edge_index.owns_data = false;

    return device_node_edge_index;
}

#endif

HOST void node_edge_index::rebuild(NodeEdgeIndexStore *node_edge_index, const EdgeDataStore *edge_data, const bool is_directed) {
    // Get sizes
    const size_t num_edges = edge_data->timestamps_size;

    // Step 2: Allocate and compute node edge offsets
    allocate_node_edge_offsets(node_edge_index, edge_data->active_node_ids_size, is_directed);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_edge_offsets_cuda(node_edge_index, edge_data, is_directed);
    } else
    #endif
    {
        compute_node_edge_offsets_std(node_edge_index, edge_data, is_directed);
    }

    // Step 3: Allocate and compute node edge indices
    allocate_node_edge_indices(node_edge_index, is_directed);

    // Create buffer for outbound edge indices
    size_t outbound_edge_indices_len = is_directed ? num_edges : num_edges * 2;
    EdgeWithEndpointType *outbound_edge_indices_buffer = nullptr;
    allocate_memory(&outbound_edge_indices_buffer, outbound_edge_indices_len, node_edge_index->use_gpu);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_edge_indices_cuda(node_edge_index, edge_data, outbound_edge_indices_buffer, is_directed);
    } else
    #endif
    {
        compute_node_edge_indices_std(node_edge_index, edge_data, outbound_edge_indices_buffer, is_directed);
    }

    // Clean up edge indices buffer
    clear_memory(&outbound_edge_indices_buffer, node_edge_index->use_gpu);

    // Step 4: Compute node timestamp offsets
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_timestamp_offsets_cuda(node_edge_index, edge_data, edge_data->active_node_ids_size,
                                            is_directed);
    } else
    #endif
    {
        compute_node_timestamp_offsets_std(node_edge_index, edge_data, edge_data->active_node_ids_size,
                                           is_directed);
    }

    // Step 5: Allocate and compute node timestamp indices
    allocate_node_timestamp_indices(node_edge_index, is_directed);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_timestamp_indices_cuda(node_edge_index, edge_data, edge_data->active_node_ids_size,
                                            is_directed);
    } else
    #endif
    {
        compute_node_timestamp_indices_std(node_edge_index, edge_data, edge_data->active_node_ids_size,
                                           is_directed);
    }
}
