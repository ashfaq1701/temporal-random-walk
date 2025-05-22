#include "node_edge_index.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include "../common/cuda_sort.cuh"
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
    const bool is_directed
) {
    const size_t edges_size = edge_data->timestamps_size;
    const size_t buffer_size = is_directed ? edges_size : edges_size * 2;

    const int* sources = edge_data->sources;
    const int* targets = edge_data->targets;
    size_t* outbound_indices = node_edge_index->outbound_indices;

    // === Step 1: Initialize outbound_indices ===
    #pragma omp parallel for
    for (size_t i = 0; i < edges_size; ++i) {
        if (is_directed) {
            outbound_indices[i] = i;
        } else {
            outbound_indices[i * 2]     = i;  // source endpoint
            outbound_indices[i * 2 + 1] = i;  // target endpoint
        }
    }

    // === Step 2: Generate node keys for sorting ===
    std::vector<int> node_keys(buffer_size);

    #pragma omp parallel for
    for (size_t i = 0; i < buffer_size; ++i) {
        const size_t edge_id = outbound_indices[i];
        const bool is_source = is_directed || (i % 2 == 0);
        node_keys[i] = is_source ? sources[edge_id] : targets[edge_id];
    }

    // === Step 3: Build a permutation array for indirect stable sort ===
    std::vector<size_t> indices(buffer_size);
    #pragma omp parallel for
    for (size_t i = 0; i < buffer_size; ++i) {
        indices[i] = i;
    }

    // === Step 4: Stable sort the permutation array by node ID ===
    parallel::stable_sort(
        indices.begin(),
        indices.end(),
        [&node_keys](const size_t a, const size_t b) {
            return node_keys[a] < node_keys[b];
        }
    );

    // === Step 5: Apply permutation to reorder outbound_indices ===
    std::vector<size_t> sorted_outbound(buffer_size);
    #pragma omp parallel for
    for (size_t i = 0; i < buffer_size; ++i) {
        sorted_outbound[i] = outbound_indices[indices[i]];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < buffer_size; ++i) {
        outbound_indices[i] = sorted_outbound[i];
    }

    // === Step 6: Handle inbound_indices (only for directed graphs) ===
    if (is_directed) {
        size_t* inbound_indices = node_edge_index->inbound_indices;

        // Fill with 0..edges_size-1
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_indices[i] = i;
        }

        // Stable sort directly by target node ID
        parallel::stable_sort(
            inbound_indices,
            inbound_indices + edges_size,
            [targets](const size_t a, const size_t b) {
                return targets[a] < targets[b];
            }
        );
    }
}

HOST void node_edge_index::compute_node_timestamp_offsets_and_indices_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const size_t node_count,
    const bool is_directed,
    const int* outbound_node_ids,
    const int* inbound_node_ids
) {
    const int64_t* timestamps = edge_data->timestamps;

    const size_t* outbound_indices = node_edge_index->outbound_indices;
    const size_t* inbound_indices = node_edge_index->inbound_indices;

    size_t* outbound_group_indices = node_edge_index->outbound_timestamp_group_indices;
    size_t* inbound_group_indices = node_edge_index->inbound_timestamp_group_indices;

    size_t* outbound_group_offsets = node_edge_index->outbound_timestamp_group_offsets;
    size_t* inbound_group_offsets = node_edge_index->inbound_timestamp_group_offsets;

    const size_t num_outbound = node_edge_index->outbound_indices_size;
    const size_t num_inbound = node_edge_index->inbound_indices_size;

    // === OUTBOUND ===
    {
        std::vector<size_t> flags(num_outbound, 0);

        // Step 1: Mark group starts
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (i == 0) {
                flags[i] = 1;
                continue;
            }

            const int node_curr = outbound_node_ids[i];
            const int node_prev = outbound_node_ids[i - 1];

            const int64_t ts_curr = timestamps[outbound_indices[i]];
            const int64_t ts_prev = timestamps[outbound_indices[i - 1]];

            flags[i] = (node_curr != node_prev || ts_curr != ts_prev) ? 1 : 0;
        }

        // Step 2: Scan to compute write positions
        std::vector<size_t> flag_scan(num_outbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_outbound);

        // Step 3: Write group indices
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (flags[i]) {
                outbound_group_indices[flag_scan[i]] = i;
            }
        }

        // Step 4: Count groups per node
        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (!flags[i]) continue;
            const int node = outbound_node_ids[i];
            #pragma omp atomic
            group_counts[node]++;
        }

        // Step 5: Compute group_offsets[1..], set group_offsets[0] = 0
        outbound_group_offsets[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);
        std::copy(group_counts.begin(), group_counts.end(), outbound_group_offsets + 1);
    }

    // === INBOUND ===
    if (is_directed) {
        std::vector<size_t> flags(num_inbound, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (i == 0) {
                flags[i] = 1;
                continue;
            }

            const int node_curr = inbound_node_ids[i];
            const int node_prev = inbound_node_ids[i - 1];

            const int64_t ts_curr = timestamps[inbound_indices[i]];
            const int64_t ts_prev = timestamps[inbound_indices[i - 1]];

            flags[i] = (node_curr != node_prev || ts_curr != ts_prev) ? 1 : 0;
        }

        std::vector<size_t> flag_scan(num_inbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_inbound);

        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (flags[i]) {
                inbound_group_indices[flag_scan[i]] = i;
            }
        }

        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (!flags[i]) continue;
            const int node = inbound_node_ids[i];
            #pragma omp atomic
            group_counts[node]++;
        }

        inbound_group_offsets[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);
        std::copy(group_counts.begin(), group_counts.end(), inbound_group_offsets + 1);
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
        const thrust::device_ptr<size_t> d_inbound_offsets(inbound_offsets_ptr);
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
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const bool is_directed,
    const size_t outbound_buffer_size,
    int* outbound_node_ids,
    int* inbound_node_ids
) {
    const size_t edges_size = edge_data->timestamps_size;

    const int* sources = edge_data->sources;
    const int* targets = edge_data->targets;
    size_t* outbound_indices = node_edge_index->outbound_indices;

    // === Step 1: Initialize outbound_indices ===
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(edges_size),
        [outbound_indices, is_directed] DEVICE (const size_t i) {
            if (is_directed) {
                outbound_indices[i] = i;
            } else {
                outbound_indices[i * 2]     = i;  // source endpoint
                outbound_indices[i * 2 + 1] = i;  // target endpoint
            }
        }
    );
    CUDA_KERNEL_CHECK("Initialized outbound_indices");

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<thrust::device_ptr<int>::difference_type>(0),
        thrust::make_counting_iterator<thrust::device_ptr<int>::difference_type>(static_cast<long>(outbound_buffer_size)),
        [outbound_node_ids, outbound_indices, sources, targets, is_directed] DEVICE (const thrust::device_ptr<int>::difference_type i) {
            const size_t edge_id = outbound_indices[i];
            const bool is_source = is_directed || (i % 2 == 0);
            outbound_node_ids[i] = is_source ? sources[edge_id] : targets[edge_id];
        }
    );
    CUDA_KERNEL_CHECK("Generated node keys");

    // === Step 3: Build a permutation array for indirect stable sort ===
    thrust::device_vector<size_t> indices(outbound_buffer_size);
    thrust::sequence(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end()
    );
    CUDA_KERNEL_CHECK("Created indices array");

    // === Step 4: Stable sort the permutation array by node ID ===
    cub_radix_sort_keys_and_values(
        thrust::raw_pointer_cast(outbound_node_ids),
        thrust::raw_pointer_cast(indices.data()),
        outbound_buffer_size);
    CUDA_KERNEL_CHECK("Sorted indices by node keys");

    // === Step 5: Apply permutation to reorder outbound_indices ===
    thrust::device_vector<size_t> sorted_outbound(outbound_buffer_size);
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<thrust::device_ptr<int>::difference_type>(0),
        thrust::make_counting_iterator<thrust::device_ptr<int>::difference_type>(static_cast<long>(outbound_buffer_size)),
        [sorted_outbound = sorted_outbound.data(), outbound_indices, indices = indices.data()] DEVICE (const thrust::device_ptr<int>::difference_type i) {
            sorted_outbound[i] = outbound_indices[indices[i]];
        }
    );
    CUDA_KERNEL_CHECK("Applied permutation");

    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_outbound.begin(),
        sorted_outbound.end(),
        outbound_indices
    );
    CUDA_KERNEL_CHECK("Copied sorted results");

    // === Step 6: Handle inbound_indices (only for directed graphs) ===
    if (is_directed) {
        size_t* inbound_indices = node_edge_index->inbound_indices;

        // Fill with 0..edges_size-1
        thrust::sequence(
            DEVICE_EXECUTION_POLICY,
            inbound_indices,
            inbound_indices + edges_size
        );
        CUDA_KERNEL_CHECK("Initialized inbound_indices");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            inbound_node_ids,
            targets,
            edges_size,
            cudaMemcpyDeviceToDevice
        ));

        cub_radix_sort_keys_and_values(
            inbound_node_ids,
            inbound_indices,
            edges_size);
        CUDA_KERNEL_CHECK("Sorted inbound_indices by target node");
    }
}

HOST void node_edge_index::compute_node_timestamp_offsets_and_indices_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    const size_t node_count,
    const bool is_directed,
    const int* outbound_node_ids,
    const int* inbound_node_ids
) {
    int64_t* timestamps_ptr = edge_data->timestamps;

    // === OUTBOUND ===
    {
        const size_t num_edges = node_edge_index->outbound_indices_size;
        size_t* indices = node_edge_index->outbound_indices;
        size_t* group_indices = node_edge_index->outbound_timestamp_group_indices;
        size_t* group_offsets = node_edge_index->outbound_timestamp_group_offsets;

        // Step 1: Mark group starts
        thrust::device_vector<int> flags(num_edges, 0);
        int* flags_ptr = thrust::raw_pointer_cast(flags.data());

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags_ptr,
            [outbound_node_ids, indices, timestamps_ptr] DEVICE(size_t i) -> int {
                if (i == 0) return 1;
                const int curr_node = outbound_node_ids[i];
                const int prev_node = outbound_node_ids[i - 1];
                const int64_t curr_ts = timestamps_ptr[indices[i]];
                const int64_t prev_ts = timestamps_ptr[indices[i - 1]];
                return (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
            }
        );

        // Step 2: Write group_indices using copy_if
        thrust::copy_if(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags.begin(),
            group_indices,
            [] DEVICE(const int flag) { return flag != 0; }
        );

        // Step 3: Count group starts per node
        thrust::device_vector<size_t> group_counts(node_count, 0);
        size_t* group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, outbound_node_ids, group_counts_ptr] DEVICE(const size_t i) {
                if (flags_ptr[i]) {
                    atomicAdd(reinterpret_cast<unsigned int*>(&group_counts_ptr[outbound_node_ids[i]]), 1);
                }
            }
        );

        // Step 4: Inclusive scan → group offsets
        CUDA_CHECK_AND_CLEAR(cudaMemset(group_offsets, 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            group_offsets + 1
        );
        CUDA_KERNEL_CHECK("Completed outbound timestamp group offsets/indices");
    }

    // === INBOUND ===
    if (is_directed) {
        const size_t num_edges = node_edge_index->inbound_indices_size;
        size_t* indices = node_edge_index->inbound_indices;
        size_t* group_indices = node_edge_index->inbound_timestamp_group_indices;
        size_t* group_offsets = node_edge_index->inbound_timestamp_group_offsets;

        thrust::device_vector<int> flags(num_edges, 0);
        int* flags_ptr = thrust::raw_pointer_cast(flags.data());

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags_ptr,
            [inbound_node_ids, indices, timestamps_ptr] DEVICE(size_t i) -> int {
                if (i == 0) return 1;
                const int curr_node = inbound_node_ids[i];
                const int prev_node = inbound_node_ids[i - 1];
                const int64_t curr_ts = timestamps_ptr[indices[i]];
                const int64_t prev_ts = timestamps_ptr[indices[i - 1]];
                return (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
            }
        );

        thrust::copy_if(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags.begin(),
            group_indices,
            [] DEVICE(const int flag) { return flag != 0; }
        );

        thrust::device_vector<size_t> group_counts(node_count, 0);
        size_t* group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, inbound_node_ids, group_counts_ptr] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    atomicAdd(reinterpret_cast<unsigned int*>(&group_counts_ptr[inbound_node_ids[i]]), 1);
                }
            }
        );

        CUDA_CHECK_AND_CLEAR(cudaMemset(group_offsets, 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            group_offsets + 1
        );
        CUDA_KERNEL_CHECK("Completed inbound timestamp group offsets/indices");
    }
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
        MemoryView<size_t> outbound_offsets = get_timestamp_offset_vector(node_edge_index, true, false);

        double* d_forward_weights = nullptr;
        double* d_backward_weights = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_forward_weights, outbound_groups_size * sizeof(double)));
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, outbound_groups_size * sizeof(double)));

        int64_t* timestamps_ptr = edge_data->timestamps;
        size_t* outbound_indices_ptr = node_edge_index->outbound_indices;
        size_t* outbound_group_indices_ptr = node_edge_index->outbound_timestamp_group_indices;
        size_t* outbound_offsets_ptr = outbound_offsets.data;

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, outbound_indices_ptr, outbound_group_indices_ptr,
             outbound_offsets_ptr, d_forward_weights, d_backward_weights, timescale_bound]
            DEVICE(const size_t node) {
                const size_t out_start = outbound_offsets_ptr[node];
                const size_t out_end = outbound_offsets_ptr[node + 1];

                if (out_start >= out_end) return;

                const size_t first_group_start = outbound_group_indices_ptr[out_start];
                const size_t last_group_start = outbound_group_indices_ptr[out_end - 1];
                const int64_t min_ts = timestamps_ptr[outbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[outbound_indices_ptr[last_group_start]];
                const double time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? timescale_bound / time_diff
                    : 1.0;

                double forward_sum = 0.0;
                double backward_sum = 0.0;

                // Compute weights
                for (size_t pos = out_start; pos < out_end; ++pos) {
                    const size_t edge_start = outbound_group_indices_ptr[pos];
                    const int64_t group_ts = timestamps_ptr[outbound_indices_ptr[edge_start]];

                    const double tf = (max_ts - group_ts) * time_scale;
                    const double tb = (group_ts - min_ts) * time_scale;

                    const double fw = exp(tf);
                    const double bw = exp(tb);

                    d_forward_weights[pos] = fw;
                    d_backward_weights[pos] = bw;

                    forward_sum += fw;
                    backward_sum += bw;
                }

                // Normalize and compute cumulative weights
                double fsum = 0.0;
                double bsum = 0.0;
                for (size_t pos = out_start; pos < out_end; ++pos) {
                    d_forward_weights[pos] /= forward_sum;
                    d_backward_weights[pos] /= backward_sum;

                    fsum += d_forward_weights[pos];
                    bsum += d_backward_weights[pos];

                    d_forward_weights[pos] = fsum;
                    d_backward_weights[pos] = bsum;
                }
            }
        );
        CUDA_KERNEL_CHECK("After thrust for_each outbound weights in update_temporal_weights_cuda");

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

        CUDA_CHECK_AND_CLEAR(cudaFree(d_forward_weights));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_backward_weights));
    }

    // Process inbound weights (only backward)
    if (node_edge_index->inbound_offsets_size > 0) {
        MemoryView<size_t> inbound_offsets = get_timestamp_offset_vector(node_edge_index, false, true);
        const size_t inbound_groups_size = node_edge_index->inbound_timestamp_group_indices_size;

        double* d_backward_weights = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, inbound_groups_size * sizeof(double)));

        int64_t* timestamps_ptr = edge_data->timestamps;
        size_t* inbound_indices_ptr = node_edge_index->inbound_indices;
        size_t* inbound_group_indices_ptr = node_edge_index->inbound_timestamp_group_indices;
        size_t* inbound_offsets_ptr = inbound_offsets.data;

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, inbound_indices_ptr, inbound_group_indices_ptr,
             inbound_offsets_ptr, d_backward_weights, timescale_bound]
            DEVICE(const size_t node) {
                const size_t in_start = inbound_offsets_ptr[node];
                const size_t in_end = inbound_offsets_ptr[node + 1];

                if (in_start >= in_end) return;

                const size_t first_group_start = inbound_group_indices_ptr[in_start];
                const size_t last_group_start = inbound_group_indices_ptr[in_end - 1];
                const int64_t min_ts = timestamps_ptr[inbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[inbound_indices_ptr[last_group_start]];
                const double time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? timescale_bound / time_diff
                    : 1.0;

                double backward_sum = 0.0;

                // Compute weights
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    const size_t edge_start = inbound_group_indices_ptr[pos];
                    const int64_t group_ts = timestamps_ptr[inbound_indices_ptr[edge_start]];

                    const double tb = (group_ts - min_ts) * time_scale;
                    const double bw = exp(tb);

                    d_backward_weights[pos] = bw;
                    backward_sum += bw;
                }

                // Normalize and compute cumulative weights
                double bsum = 0.0;
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    d_backward_weights[pos] /= backward_sum;
                    bsum += d_backward_weights[pos];
                    d_backward_weights[pos] = bsum;
                }
            }
        );
        CUDA_KERNEL_CHECK("After thrust for_each inbound weights in update_temporal_weights_cuda");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            node_edge_index->inbound_backward_cumulative_weights_exponential,
            d_backward_weights,
            inbound_groups_size * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));

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

HOST void node_edge_index::rebuild(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    const bool is_directed
) {
    // Step 1: Allocate and compute node edge offsets
    allocate_node_edge_offsets(node_edge_index, edge_data->active_node_ids_size, is_directed);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_edge_offsets_cuda(node_edge_index, edge_data, is_directed);
    } else
    #endif
    {
        compute_node_edge_offsets_std(node_edge_index, edge_data, is_directed);
    }

    // Step 2: Allocate and compute node edge indices
    allocate_node_edge_indices(node_edge_index, is_directed);

    const size_t num_edges = edge_data->active_node_ids_size;
    const size_t outbound_buffer_size = is_directed ? num_edges : num_edges * 2;

    int* outbound_node_ids = nullptr;
    allocate_memory(&outbound_node_ids, outbound_buffer_size, node_edge_index->use_gpu);

    int* inbound_node_ids = nullptr;
    allocate_memory(&inbound_node_ids, num_edges, node_edge_index->use_gpu);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_edge_indices_cuda(
            node_edge_index,
            edge_data,
            is_directed,
            outbound_buffer_size,
            outbound_node_ids,
            inbound_node_ids
        );
    } else
    #endif
    {
        compute_node_edge_indices_std(
            node_edge_index,
            edge_data,
            is_directed
        );
    }

    // Step 3: Allocate timestamp indices
    allocate_node_timestamp_indices(node_edge_index, is_directed);

    // Step 4 + 5: Compute timestamp group offsets AND group indices
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_timestamp_offsets_and_indices_cuda(
            node_edge_index,
            edge_data,
            edge_data->active_node_ids_size,
            is_directed,
            outbound_node_ids,
            inbound_node_ids
        );
    } else
    #endif
    {
        compute_node_timestamp_offsets_and_indices_std(
            node_edge_index,
            edge_data,
            edge_data->active_node_ids_size,
            is_directed,
            outbound_node_ids,
            inbound_node_ids
        );
    }

    // Clean up temporary buffers
    clear_memory(&outbound_node_ids, node_edge_index->use_gpu);
    clear_memory(&inbound_node_ids, node_edge_index->use_gpu);
}
