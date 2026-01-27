#include "node_edge_index.cuh"

#include <cmath>

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include "../common/cuda_sort.cuh"
#endif

#include <omp.h>
#include <cmath>
#include <algorithm>
#include "../utils/omp_utils.cuh"
#include "../common/nvtx_utils.h"
#include "../common/parallel_algorithms.cuh"
#include "../common/cuda_config.cuh"

/**
 * Common Functions
 */

HOST void node_edge_index::clear(NodeEdgeIndexStore *node_edge_index) {
    // Clear edge CSR structures
    clear_memory(&node_edge_index->node_group_outbound_offsets, node_edge_index->use_gpu);
    node_edge_index->node_group_outbound_offsets_size = 0;

    clear_memory(&node_edge_index->node_ts_sorted_outbound_indices, node_edge_index->use_gpu);
    node_edge_index->node_ts_sorted_outbound_indices_size = 0;

    clear_memory(&node_edge_index->count_ts_group_per_node_outbound, node_edge_index->use_gpu);
    node_edge_index->count_ts_group_per_node_outbound_size = 0;

    clear_memory(&node_edge_index->node_ts_group_outbound_offsets, node_edge_index->use_gpu);
    node_edge_index->node_ts_group_outbound_offsets_size = 0;

    // Clear inbound structures
    clear_memory(&node_edge_index->node_group_inbound_offsets, node_edge_index->use_gpu);
    node_edge_index->node_group_inbound_offsets_size = 0;

    clear_memory(&node_edge_index->node_ts_sorted_inbound_indices, node_edge_index->use_gpu);
    node_edge_index->node_ts_sorted_inbound_indices_size = 0;

    clear_memory(&node_edge_index->count_ts_group_per_node_inbound, node_edge_index->use_gpu);
    node_edge_index->count_ts_group_per_node_inbound_size = 0;

    clear_memory(&node_edge_index->node_ts_group_inbound_offsets, node_edge_index->use_gpu);
    node_edge_index->node_ts_group_inbound_offsets_size = 0;

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
        const size_t *offsets = forward ? node_edge_index->node_group_outbound_offsets : node_edge_index->node_group_inbound_offsets;
        size_t offsets_size = forward ? node_edge_index->node_group_outbound_offsets_size : node_edge_index->node_group_inbound_offsets_size;

        if (dense_node_id < 0 || dense_node_id >= offsets_size - 1) {
            return SizeRange{0, 0};
        }

        const size_t start = offsets[dense_node_id];
        const size_t end = offsets[dense_node_id + 1];

        return SizeRange{start, end};
    } else {
        if (dense_node_id < 0 || dense_node_id >= node_edge_index->node_group_outbound_offsets_size - 1) {
            return SizeRange{0, 0};
        }

        const size_t start = node_edge_index->node_group_outbound_offsets[dense_node_id];
        const size_t end = node_edge_index->node_group_outbound_offsets[dense_node_id + 1];

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
        group_offsets = node_edge_index->count_ts_group_per_node_inbound;
        group_offsets_size = node_edge_index->count_ts_group_per_node_inbound_size;
        group_indices = node_edge_index->node_ts_group_inbound_offsets;
        edge_offsets = node_edge_index->node_group_inbound_offsets;
    } else {
        group_offsets = node_edge_index->count_ts_group_per_node_outbound;
        group_offsets_size = node_edge_index->count_ts_group_per_node_outbound_size;
        group_indices = node_edge_index->node_ts_group_outbound_offsets;
        edge_offsets = node_edge_index->node_group_outbound_offsets;
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
            node_edge_index->count_ts_group_per_node_inbound,
            node_edge_index->count_ts_group_per_node_inbound_size
        };
    } else {
        return MemoryView<size_t>{
            node_edge_index->count_ts_group_per_node_outbound,
            node_edge_index->count_ts_group_per_node_outbound_size
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

HOST void node_edge_index::allocate_node_group_offsets(NodeEdgeIndexStore *node_edge_index, const size_t node_index_capacity,
                                            const bool is_directed) {
    allocate_memory(&node_edge_index->node_group_outbound_offsets, node_index_capacity + 1, node_edge_index->use_gpu);
    node_edge_index->node_group_outbound_offsets_size = node_index_capacity + 1;
    fill_memory(node_edge_index->node_group_outbound_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                node_edge_index->use_gpu);

    // For directed graphs, also allocate inbound structures
    if (is_directed) {
        allocate_memory(&node_edge_index->node_group_inbound_offsets, node_index_capacity + 1, node_edge_index->use_gpu);
        node_edge_index->node_group_inbound_offsets_size = node_index_capacity + 1;
        fill_memory(node_edge_index->node_group_inbound_offsets, node_index_capacity + 1, static_cast<size_t>(0),
                    node_edge_index->use_gpu);
    }
}

HOST void node_edge_index::allocate_node_ts_sorted_indices(NodeEdgeIndexStore *node_edge_index, const bool is_directed) {
    size_t num_outbound_edges = 0;

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // For GPU memory, we need to copy the value back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_outbound_edges,
            node_edge_index->node_group_outbound_offsets + (node_edge_index->node_group_outbound_offsets_size - 1),
            sizeof(size_t),
            cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        // For CPU memory, we can access it directly
        num_outbound_edges = node_edge_index->node_group_outbound_offsets[node_edge_index->node_group_outbound_offsets_size - 1];
    }

    // Allocate memory for outbound indices
    allocate_memory(&node_edge_index->node_ts_sorted_outbound_indices, num_outbound_edges, node_edge_index->use_gpu);
    node_edge_index->node_ts_sorted_outbound_indices_size = num_outbound_edges;

    // For directed graphs, also allocate inbound indices
    if (is_directed) {
        size_t num_inbound_edges = 0;

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&num_inbound_edges,
                node_edge_index->node_group_inbound_offsets + (node_edge_index->node_group_inbound_offsets_size - 1),
                sizeof(size_t),
                cudaMemcpyDeviceToHost));
        } else
        #endif
        {
            num_inbound_edges = node_edge_index->node_group_inbound_offsets[node_edge_index->node_group_inbound_offsets_size - 1];
        }

        allocate_memory(&node_edge_index->node_ts_sorted_inbound_indices, num_inbound_edges, node_edge_index->use_gpu);
        node_edge_index->node_ts_sorted_inbound_indices_size = num_inbound_edges;
    }
}

/**
 * Std implementations
 */
HOST void node_edge_index::compute_node_group_offsets_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const bool is_directed
) {
    const size_t num_edges = edge_data->timestamps_size;

    auto* outbound_offsets = node_edge_index->node_group_outbound_offsets;
    auto* inbound_offsets  = node_edge_index->node_group_inbound_offsets;
    const auto* sources    = edge_data->sources;
    const auto* targets    = edge_data->targets;

    const size_t offset_size = node_edge_index->node_group_outbound_offsets_size;

    // Step 1: Zero out offset arrays
    std::fill_n(outbound_offsets, offset_size, 0);
    if (is_directed) {
        std::fill_n(inbound_offsets, node_edge_index->node_group_inbound_offsets_size, 0);
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
        parallel_inclusive_scan(inbound_offsets + 1, node_edge_index->node_group_inbound_offsets_size - 1);
    }
}

HOST void node_edge_index::compute_node_ts_sorted_indices_std(
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
    size_t* outbound_indices = node_edge_index->node_ts_sorted_outbound_indices;

    // === Step 1: Initialize node_ts_sorted_outbound_indices ===
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
    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        const size_t edge_id = outbound_indices[i];
        const bool is_source = is_directed || (i % 2 == 0);
        outbound_node_ids[i] = is_source ? sources[edge_id] : targets[edge_id];
    }

    // === Step 3: Build a permutation array for indirect stable sort ===
    std::vector<size_t> indices(outbound_buffer_size);
    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        indices[i] = i;
    }

    // === Step 4: Stable sort the permutation array by node ID ===
    parallel::stable_sort(
        indices.begin(),
        indices.end(),
        [&outbound_node_ids](const size_t a, const size_t b) {
            return outbound_node_ids[a] < outbound_node_ids[b];
        }
    );

    // === Step 5: Apply permutation to reorder node_ts_sorted_outbound_indices and outbound_node_ids ===
    std::vector<size_t> sorted_outbound_indices(outbound_buffer_size);
    std::vector<int> sorted_outbound_node_ids(outbound_buffer_size);

    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        sorted_outbound_indices[i] = outbound_indices[indices[i]];
        sorted_outbound_node_ids[i] = outbound_node_ids[indices[i]];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        outbound_indices[i] = sorted_outbound_indices[i];
        outbound_node_ids[i] = sorted_outbound_node_ids[i];
    }

    // === Step 6: Handle node_ts_sorted_inbound_indices (only for directed graphs) ===
    if (is_directed) {
        size_t* inbound_indices = node_edge_index->node_ts_sorted_inbound_indices;

        // Step 1: Fill with 0..edges_size-1
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_indices[i] = i;
        }

        // Step 2: Fill inbound_node_ids = targets[i]
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_node_ids[i] = edge_data->targets[i];
        }

        // Step 3: Sort node_ts_sorted_inbound_indices by inbound_node_ids
        parallel::stable_sort(inbound_indices, inbound_indices + edges_size,
            [inbound_node_ids](size_t a, size_t b) {
                return inbound_node_ids[a] < inbound_node_ids[b];
            }
        );

        // Step 4: Permute inbound_node_ids to match sorted node_ts_sorted_inbound_indices
        std::vector<int> sorted_inbound_node_ids(edges_size);
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            sorted_inbound_node_ids[i] = inbound_node_ids[inbound_indices[i]];
        }

        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_node_ids[i] = sorted_inbound_node_ids[i];
        }
    }
}

HOST void node_edge_index::allocate_and_compute_node_ts_group_counts_and_offsets_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const size_t node_count,
    const bool is_directed,
    const int* outbound_node_ids,
    const int* inbound_node_ids
) {
    const int64_t* timestamps = edge_data->timestamps;

    const size_t* outbound_indices = node_edge_index->node_ts_sorted_outbound_indices;
    const size_t* inbound_indices = node_edge_index->node_ts_sorted_inbound_indices;

    const size_t num_outbound = node_edge_index->node_ts_sorted_outbound_indices_size;
    const size_t num_inbound = node_edge_index->node_ts_sorted_inbound_indices_size;

    // === OUTBOUND ===
    {
        std::vector<size_t> flags(num_outbound, 0);

        // Step 1: Mark group start flags
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (i == 0) {
                flags[i] = 1;
                continue;
            }
            const int curr_node = outbound_node_ids[i];
            const int prev_node = outbound_node_ids[i - 1];
            const int64_t curr_ts = timestamps[outbound_indices[i]];
            const int64_t prev_ts = timestamps[outbound_indices[i - 1]];
            flags[i] = (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
        }

        // Step 2: Compute number of groups
        size_t num_groups = 0;
        #pragma omp parallel for reduction(+:num_groups)
        for (size_t i = 0; i < num_outbound; ++i) {
            num_groups += flags[i];
        }

        resize_memory(
            &node_edge_index->node_ts_group_outbound_offsets,
            node_edge_index->node_ts_group_outbound_offsets_size,
            num_groups,
            node_edge_index->use_gpu
        );
        node_edge_index->node_ts_group_outbound_offsets_size = num_groups;

        size_t* group_indices_out = node_edge_index->node_ts_group_outbound_offsets;

        // Step 3: Write group start indices using exclusive scan over flags
        std::vector<size_t> flag_scan(num_outbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_outbound);

        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (flags[i]) {
                group_indices_out[flag_scan[i]] = i;
            }
        }

        // Step 4: Count groups per node
        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (!flags[i]) continue;
            const int node = outbound_node_ids[i];

            if (node >= 0 && node < node_count) {
                #pragma omp atomic
                group_counts[node]++;
            }
        }

        // Step 5: Allocate and compute group offsets
        resize_memory(
            &node_edge_index->count_ts_group_per_node_outbound,
            node_edge_index->count_ts_group_per_node_outbound_size,
            node_count + 1,
            node_edge_index->use_gpu
        );
        node_edge_index->count_ts_group_per_node_outbound_size = node_count + 1;

        node_edge_index->count_ts_group_per_node_outbound[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);

        #pragma omp parallel for
        for (size_t i = 0; i < node_count; ++i) {
            node_edge_index->count_ts_group_per_node_outbound[i + 1] = group_counts[i];
        }
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
            const int curr_node = inbound_node_ids[i];
            const int prev_node = inbound_node_ids[i - 1];
            const int64_t curr_ts = timestamps[inbound_indices[i]];
            const int64_t prev_ts = timestamps[inbound_indices[i - 1]];
            flags[i] = (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
        }

        size_t num_groups = 0;
        #pragma omp parallel for reduction(+:num_groups)
        for (size_t i = 0; i < num_inbound; ++i) {
            num_groups += flags[i];
        }

        resize_memory(
            &node_edge_index->node_ts_group_inbound_offsets,
            node_edge_index->node_ts_group_inbound_offsets_size,
            num_groups,
            node_edge_index->use_gpu
        );
        node_edge_index->node_ts_group_inbound_offsets_size = num_groups;

        size_t* group_indices_out = node_edge_index->node_ts_group_inbound_offsets;

        std::vector<size_t> flag_scan(num_inbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_inbound);

        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (flags[i]) {
                group_indices_out[flag_scan[i]] = i;
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

        resize_memory(
            &node_edge_index->count_ts_group_per_node_inbound,
            node_edge_index->count_ts_group_per_node_inbound_size,
            node_count + 1,
            node_edge_index->use_gpu
        );
        node_edge_index->count_ts_group_per_node_inbound_size = node_count + 1;

        node_edge_index->count_ts_group_per_node_inbound[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);

        #pragma omp parallel for
        for (size_t i = 0; i < node_count; ++i) {
            node_edge_index->count_ts_group_per_node_inbound[i + 1] = group_counts[i];
        }
    }
}


HOST void node_edge_index::update_temporal_weights_std(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const double timescale_bound
) {
    const size_t node_index_capacity = node_edge_index->node_group_outbound_offsets_size - 1;
    const size_t outbound_groups_size = node_edge_index->node_ts_group_outbound_offsets_size;

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

    const bool is_directed = node_edge_index->node_group_inbound_offsets_size > 0;

    if (is_directed) {
        const size_t inbound_groups_size = node_edge_index->node_ts_group_inbound_offsets_size;
        resize_memory(
            &node_edge_index->inbound_backward_cumulative_weights_exponential,
            node_edge_index->inbound_backward_cumulative_weights_exponential_size,
            inbound_groups_size,
            node_edge_index->use_gpu
        );
        node_edge_index->inbound_backward_cumulative_weights_exponential_size = inbound_groups_size;
    }

    // Process outbound weights
    {
        auto outbound_offsets = get_timestamp_offset_vector(node_edge_index, true, false);

        // Step 1: Create node mapping for each group position
        std::vector<size_t> group_to_node(outbound_groups_size);

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            for (size_t pos = out_start; pos < out_end; ++pos) {
                group_to_node[pos] = node;
            }
        }

        // Step 2: Compute min/max timestamps and time scale per node
        std::vector<int64_t> node_min_ts(node_index_capacity);
        std::vector<int64_t> node_max_ts(node_index_capacity);
        std::vector<double> node_time_scale(node_index_capacity);

        const auto* ts_group_indices = node_edge_index->node_ts_group_outbound_offsets;
        const auto* edge_indices = node_edge_index->node_ts_sorted_outbound_indices;
        const auto* timestamps = edge_data->timestamps;

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            if (out_start >= out_end) {
                node_min_ts[node] = 0;
                node_max_ts[node] = 0;
                node_time_scale[node] = 1.0;
                continue;
            }

            const int64_t min_ts = timestamps[edge_indices[ts_group_indices[out_start]]];
            const int64_t max_ts = timestamps[edge_indices[ts_group_indices[out_end - 1]]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

            node_min_ts[node] = min_ts;
            node_max_ts[node] = max_ts;
            node_time_scale[node] = time_scale;
        }

        // Step 3: Compute raw weights in parallel
        std::vector<double> raw_forward_weights(outbound_groups_size);
        std::vector<double> raw_backward_weights(outbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const size_t edge_start = ts_group_indices[pos];
            const int64_t group_ts = timestamps[edge_indices[edge_start]];
            const int64_t min_ts = node_min_ts[node];
            const int64_t max_ts = node_max_ts[node];
            const double time_scale = node_time_scale[node];

            const double f_scaled = (timescale_bound > 0) ? static_cast<double>(max_ts - group_ts) * time_scale : static_cast<double>(max_ts - group_ts);
            const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);

            raw_forward_weights[pos] = std::exp(f_scaled);
            raw_backward_weights[pos] = std::exp(b_scaled);
        }

        // Step 4: Compute normalization sums per node
        std::vector<double> node_forward_sums(node_index_capacity, 0.0);
        std::vector<double> node_backward_sums(node_index_capacity, 0.0);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];

            #pragma omp atomic
            node_forward_sums[node] += raw_forward_weights[pos];

            #pragma omp atomic
            node_backward_sums[node] += raw_backward_weights[pos];
        }

        // Step 5: Normalize weights
        std::vector<double> normalized_forward_weights(outbound_groups_size);
        std::vector<double> normalized_backward_weights(outbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const double forward_sum = node_forward_sums[node];
            const double backward_sum = node_backward_sums[node];

            normalized_forward_weights[pos] = raw_forward_weights[pos] / forward_sum;
            normalized_backward_weights[pos] = raw_backward_weights[pos] / backward_sum;
        }

        // Step 6: Compute cumulative sums per node
        auto* f_weights = node_edge_index->outbound_forward_cumulative_weights_exponential;
        auto* b_weights = node_edge_index->outbound_backward_cumulative_weights_exponential;

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            if (out_start >= out_end) continue;

            double f_cumsum = 0.0;
            double b_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                f_cumsum += normalized_forward_weights[pos];
                b_cumsum += normalized_backward_weights[pos];
                f_weights[pos] = f_cumsum;
                b_weights[pos] = b_cumsum;
            }
        }
    }

    // Process inbound weights (only backward)
    if (is_directed) {
        auto inbound_offsets = get_timestamp_offset_vector(node_edge_index, false, true);
        const size_t inbound_groups_size = node_edge_index->node_ts_group_inbound_offsets_size;

        // Step 1: Create node mapping for each group position
        std::vector<size_t> group_to_node(inbound_groups_size);

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            for (size_t pos = in_start; pos < in_end; ++pos) {
                group_to_node[pos] = node;
            }
        }

        // Step 2: Compute min/max timestamps and time scale per node
        std::vector<int64_t> node_min_ts(node_index_capacity);
        std::vector<int64_t> node_max_ts(node_index_capacity);
        std::vector<double> node_time_scale(node_index_capacity);

        const auto* ts_group_indices = node_edge_index->node_ts_group_inbound_offsets;
        const auto* edge_indices = node_edge_index->node_ts_sorted_inbound_indices;
        const auto* timestamps = edge_data->timestamps;

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            if (in_start >= in_end) {
                node_min_ts[node] = 0;
                node_max_ts[node] = 0;
                node_time_scale[node] = 1.0;
                continue;
            }

            const int64_t min_ts = timestamps[edge_indices[ts_group_indices[in_start]]];
            const int64_t max_ts = timestamps[edge_indices[ts_group_indices[in_end - 1]]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

            node_min_ts[node] = min_ts;
            node_max_ts[node] = max_ts;
            node_time_scale[node] = time_scale;
        }

        // Step 3: Compute raw weights in parallel
        std::vector<double> raw_backward_weights(inbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const size_t edge_start = ts_group_indices[pos];
            const int64_t group_ts = timestamps[edge_indices[edge_start]];
            const int64_t min_ts = node_min_ts[node];
            const double time_scale = node_time_scale[node];

            const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);
            raw_backward_weights[pos] = std::exp(b_scaled);
        }

        // Step 4: Compute normalization sums per node
        std::vector<double> node_backward_sums(node_index_capacity, 0.0);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];

            #pragma omp atomic
            node_backward_sums[node] += raw_backward_weights[pos];
        }

        // Step 5: Normalize weights
        std::vector<double> normalized_backward_weights(inbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const double backward_sum = node_backward_sums[node];
            normalized_backward_weights[pos] = raw_backward_weights[pos] / backward_sum;
        }

        // Step 6: Compute cumulative sums per node
        auto* b_weights = node_edge_index->inbound_backward_cumulative_weights_exponential;

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            if (in_start >= in_end) continue;

            double b_cumsum = 0.0;
            for (size_t pos = in_start; pos < in_end; ++pos) {
                b_cumsum += normalized_backward_weights[pos];
                b_weights[pos] = b_cumsum;
            }
        }
    }
}

/**
 * Cuda implementations
 */
#ifdef HAS_CUDA

HOST void node_edge_index::compute_node_group_offsets_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    bool is_directed
) {
    const size_t num_edges = edge_data->timestamps_size;

    // Get raw pointers to work with
    size_t *outbound_offsets_ptr = node_edge_index->node_group_outbound_offsets;
    size_t *inbound_offsets_ptr = is_directed ? node_edge_index->node_group_inbound_offsets : nullptr;
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
    CUDA_KERNEL_CHECK("After thrust for_each in compute_node_group_offsets_cuda");

    // Calculate prefix sums for outbound edge offsets
    thrust::device_ptr<size_t> d_outbound_offsets(outbound_offsets_ptr);
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_outbound_offsets + 1,
        d_outbound_offsets + static_cast<long>(node_edge_index->node_group_outbound_offsets_size),
        d_outbound_offsets + 1
    );
    CUDA_KERNEL_CHECK("After thrust inclusive_scan outbound in compute_node_group_offsets_cuda");

    // Calculate prefix sums for inbound edge offsets (if directed)
    if (is_directed) {
        const thrust::device_ptr<size_t> d_inbound_offsets(inbound_offsets_ptr);
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            d_inbound_offsets + 1,
            d_inbound_offsets + static_cast<long>(node_edge_index->node_group_inbound_offsets_size),
            d_inbound_offsets + 1
        );
        CUDA_KERNEL_CHECK("After thrust inclusive_scan inbound in compute_node_group_offsets_cuda");
    }
}

HOST void node_edge_index::compute_node_ts_sorted_indices_cuda(
    NodeEdgeIndexStore* node_edge_index,
    const EdgeDataStore* edge_data,
    const bool is_directed,
    const size_t outbound_buffer_size,
    int* outbound_node_ids,
    int* inbound_node_ids
) {
    NvtxRange r("node_index_rebuild");

    const size_t edges_size = edge_data->timestamps_size;

    const int* sources = edge_data->sources;
    const int* targets = edge_data->targets;
    size_t* outbound_indices = node_edge_index->node_ts_sorted_outbound_indices;

    // === Step 1: Initialize node_ts_sorted_outbound_indices ===
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
    CUDA_KERNEL_CHECK("Initialized node_ts_sorted_outbound_indices");

    // === Step 2: Fill outbound_node_ids
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(outbound_buffer_size),
        [outbound_node_ids, outbound_indices, sources, targets, is_directed] DEVICE (const size_t i) {
            const size_t edge_id = outbound_indices[i];
            const bool is_source = is_directed || (i % 2 == 0);
            outbound_node_ids[i] = is_source ? sources[edge_id] : targets[edge_id];
        }
    );
    CUDA_KERNEL_CHECK("Generated outbound_node_ids");

    // === Step 3: Build permutation array
    thrust::device_vector<size_t> indices(outbound_buffer_size);
    thrust::sequence(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end()
    );
    CUDA_KERNEL_CHECK("Generated permutation indices");

    // === Step 4: Sort indices by outbound_node_ids using your CUB wrapper
    cub_radix_sort_values_by_keys(
        outbound_node_ids,                         // keys (sorted in-place)
        thrust::raw_pointer_cast(indices.data()),  // values (permutation)
        outbound_buffer_size
    );
    CUDA_KERNEL_CHECK("Sorted indices by node keys");

    // === Step 5: Apply permutation to node_ts_sorted_outbound_indices and outbound_node_ids
    thrust::device_vector<size_t> sorted_outbound_indices(outbound_buffer_size);
    thrust::device_vector<int> sorted_outbound_node_ids(outbound_buffer_size);

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(outbound_buffer_size),
        [sorted_outbound_indices = sorted_outbound_indices.data(),
         sorted_outbound_node_ids = sorted_outbound_node_ids.data(),
         outbound_indices,
         outbound_node_ids,
         indices = indices.data()] DEVICE (const size_t i) {
            const auto idx = static_cast<long>(i);
            const size_t sorted_idx = indices[idx];
            sorted_outbound_indices[idx] = outbound_indices[sorted_idx];
            sorted_outbound_node_ids[idx] = outbound_node_ids[sorted_idx];
        }
    );
    CUDA_KERNEL_CHECK("Applied permutation");

    // Copy results back
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_outbound_indices.begin(),
        sorted_outbound_indices.end(),
        outbound_indices
    );
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_outbound_node_ids.begin(),
        sorted_outbound_node_ids.end(),
        outbound_node_ids
    );
    CUDA_KERNEL_CHECK("Copied sorted outbound data");

    // === Step 6: Handle node_ts_sorted_inbound_indices (only for directed graphs)
    if (is_directed) {
        size_t* inbound_indices = node_edge_index->node_ts_sorted_inbound_indices;

        // Step 1: Fill with 0..edges_size-1
        thrust::sequence(
            DEVICE_EXECUTION_POLICY,
            inbound_indices,
            inbound_indices + edges_size
        );
        CUDA_KERNEL_CHECK("Initialized node_ts_sorted_inbound_indices");

        // Step 2: Fill inbound_node_ids = targets[i]
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            inbound_node_ids,
            targets,
            sizeof(int) * edges_size,
            cudaMemcpyDeviceToDevice
        ));

        // Step 3: Sort node_ts_sorted_inbound_indices by inbound_node_ids
        cub_radix_sort_values_by_keys(
            inbound_node_ids,
            inbound_indices,
            edges_size
        );
        CUDA_KERNEL_CHECK("Sorted node_ts_sorted_inbound_indices");

        // Step 4: Permute inbound_node_ids to match sorted indices
        thrust::device_vector<int> sorted_inbound_node_ids(edges_size);
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(edges_size),
            [sorted_inbound_node_ids = sorted_inbound_node_ids.data(),
             inbound_node_ids, inbound_indices] DEVICE (const size_t i) {
                const auto idx = static_cast<long>(i);
                sorted_inbound_node_ids[idx] = inbound_node_ids[inbound_indices[idx]];
            }
        );
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            sorted_inbound_node_ids.begin(),
            sorted_inbound_node_ids.end(),
            inbound_node_ids
        );
        CUDA_KERNEL_CHECK("Copied sorted inbound_node_ids");
    }
}

HOST void node_edge_index::allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
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
        const size_t num_edges = node_edge_index->node_ts_sorted_outbound_indices_size;
        size_t* indices = node_edge_index->node_ts_sorted_outbound_indices;

        // Step 1: Mark group starts
        thrust::device_vector<int> flags(num_edges, 0);
        auto flags_ptr = thrust::raw_pointer_cast(flags.data());

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags_ptr,
            [outbound_node_ids, indices, timestamps_ptr] DEVICE(const size_t i) -> int {
                if (i == 0) return 1;
                const int curr_node = outbound_node_ids[i];
                const int prev_node = outbound_node_ids[i - 1];
                const int64_t curr_ts = timestamps_ptr[indices[i]];
                const int64_t prev_ts = timestamps_ptr[indices[i - 1]];
                return (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
            }
        );

        // Step 2: Compute group count
        const size_t num_groups = thrust::reduce(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
        );

        resize_memory(
            &node_edge_index->node_ts_group_outbound_offsets,
            node_edge_index->node_ts_group_outbound_offsets_size,
            num_groups,
            node_edge_index->use_gpu
        );
        node_edge_index->node_ts_group_outbound_offsets_size = num_groups;

        size_t* group_indices_out = node_edge_index->node_ts_group_outbound_offsets;

        // Step 3: Compute exclusive scan over flags
        thrust::device_vector<size_t> flag_scan(num_edges + 1, 0);
        thrust::exclusive_scan(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            flag_scan.begin()
        );

        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());

        // Step 4: Write group_indices_out
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    group_indices_out[flag_scan_ptr[i]] = i;
                }
            }
        );

        // Step 5: Count groups per node
        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, outbound_node_ids, group_counts_ptr, node_count] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    const int node = outbound_node_ids[i];
                    if (node >= 0 && node < node_count) {
                        atomicAdd(&group_counts_ptr[node], 1u);
                    }
                }
            }
        );

        // Step 6: Allocate and compute offsets
        resize_memory(
            &node_edge_index->count_ts_group_per_node_outbound,
            node_edge_index->count_ts_group_per_node_outbound_size,
            node_count + 1,
            node_edge_index->use_gpu
        );
        node_edge_index->count_ts_group_per_node_outbound_size = node_count + 1;

        CUDA_CHECK_AND_CLEAR(cudaMemset(node_edge_index->count_ts_group_per_node_outbound, 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            node_edge_index->count_ts_group_per_node_outbound + 1
        );
    }

    // === INBOUND ===
    if (is_directed) {
        const size_t num_edges = node_edge_index->node_ts_sorted_inbound_indices_size;
        size_t* indices = node_edge_index->node_ts_sorted_inbound_indices;

        thrust::device_vector<int> flags(num_edges, 0);
        auto flags_ptr = thrust::raw_pointer_cast(flags.data());

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

        const size_t num_groups = thrust::reduce(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
        );

        resize_memory(
            &node_edge_index->node_ts_group_inbound_offsets,
            node_edge_index->node_ts_group_inbound_offsets_size,
            num_groups,
            node_edge_index->use_gpu
        );
        node_edge_index->node_ts_group_inbound_offsets_size = num_groups;

        size_t* group_indices_out = node_edge_index->node_ts_group_inbound_offsets;

        thrust::device_vector<size_t> flag_scan(num_edges + 1, 0);
        thrust::exclusive_scan(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            flag_scan.begin()
        );

        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out] DEVICE(const size_t i) {
                if (flags_ptr[i]) {
                    group_indices_out[flag_scan_ptr[i]] = i;
                }
            }
        );

        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, inbound_node_ids, group_counts_ptr, node_count] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    const int node = inbound_node_ids[i];
                    if (node >= 0 && node < node_count) {
                        atomicAdd(&group_counts_ptr[node], 1u);
                    }
                }
            }
        );

        resize_memory(
            &node_edge_index->count_ts_group_per_node_inbound,
            node_edge_index->count_ts_group_per_node_inbound_size,
            node_count + 1,
            node_edge_index->use_gpu
        );
        node_edge_index->count_ts_group_per_node_inbound_size = node_count + 1;

        CUDA_CHECK_AND_CLEAR(cudaMemset(node_edge_index->count_ts_group_per_node_inbound, 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            node_edge_index->count_ts_group_per_node_inbound + 1
        );
    }
}

HOST void node_edge_index::update_temporal_weights_cuda(
    NodeEdgeIndexStore *node_edge_index,
    const EdgeDataStore *edge_data,
    double timescale_bound
) {
    // Get the number of nodes and timestamp groups
    size_t node_index_capacity = node_edge_index->node_group_outbound_offsets_size - 1;
    const size_t outbound_groups_size = node_edge_index->node_ts_group_outbound_offsets_size;

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
    if (node_edge_index->node_group_inbound_offsets_size > 0) {
        const size_t inbound_groups_size = node_edge_index->node_ts_group_inbound_offsets_size;
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

        // Step 1: Create node mapping for each group position
        thrust::device_vector<size_t> group_to_node(outbound_groups_size);
        auto group_to_node_ptr = thrust::raw_pointer_cast(group_to_node.data());

        size_t* outbound_offsets_ptr = outbound_offsets.data;

        thrust::upper_bound(
            DEVICE_EXECUTION_POLICY,
            outbound_offsets_ptr,
            outbound_offsets_ptr + node_index_capacity + 1,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            group_to_node.begin()
        );

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            group_to_node.begin(),
            [] DEVICE(const size_t x) { return x - 1; }
        );

        // Step 2: Compute min/max timestamps per node
        thrust::device_vector<int64_t> node_min_ts(node_index_capacity);
        thrust::device_vector<int64_t> node_max_ts(node_index_capacity);
        thrust::device_vector<double> node_time_scale(node_index_capacity);

        auto node_min_ts_ptr = thrust::raw_pointer_cast(node_min_ts.data());
        auto node_max_ts_ptr = thrust::raw_pointer_cast(node_max_ts.data());
        auto node_time_scale_ptr = thrust::raw_pointer_cast(node_time_scale.data());

        int64_t* timestamps_ptr = edge_data->timestamps;
        size_t* outbound_indices_ptr = node_edge_index->node_ts_sorted_outbound_indices;
        size_t* outbound_group_indices_ptr = node_edge_index->node_ts_group_outbound_offsets;

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, outbound_indices_ptr, outbound_group_indices_ptr,
             outbound_offsets_ptr, node_min_ts_ptr, node_max_ts_ptr, node_time_scale_ptr, timescale_bound]
            DEVICE(const size_t node) {
                const size_t out_start = outbound_offsets_ptr[node];
                const size_t out_end = outbound_offsets_ptr[node + 1];

                if (out_start >= out_end) {
                    node_min_ts_ptr[node] = 0;
                    node_max_ts_ptr[node] = 0;
                    node_time_scale_ptr[node] = 1.0;
                    return;
                }

                const size_t first_group_start = outbound_group_indices_ptr[out_start];
                const size_t last_group_start = outbound_group_indices_ptr[out_end - 1];
                const int64_t min_ts = timestamps_ptr[outbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[outbound_indices_ptr[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? timescale_bound / time_diff
                    : 1.0;

                node_min_ts_ptr[node] = min_ts;
                node_max_ts_ptr[node] = max_ts;
                node_time_scale_ptr[node] = time_scale;
            }
        );

        // Step 3: Compute raw weights in parallel
        thrust::device_vector<double> raw_forward_weights(outbound_groups_size);
        thrust::device_vector<double> raw_backward_weights(outbound_groups_size);

        auto raw_forward_weights_ptr = thrust::raw_pointer_cast(raw_forward_weights.data());
        auto raw_backward_weights_ptr = thrust::raw_pointer_cast(raw_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [timestamps_ptr, outbound_indices_ptr, outbound_group_indices_ptr,
             group_to_node_ptr, node_min_ts_ptr, node_max_ts_ptr, node_time_scale_ptr,
             raw_forward_weights_ptr, raw_backward_weights_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const size_t edge_start = outbound_group_indices_ptr[pos];
                const int64_t group_ts = timestamps_ptr[outbound_indices_ptr[edge_start]];
                const int64_t min_ts = node_min_ts_ptr[node];
                const int64_t max_ts = node_max_ts_ptr[node];
                const double time_scale = node_time_scale_ptr[node];

                const double tf = static_cast<double>(max_ts - group_ts) * time_scale;
                const double tb = static_cast<double>(group_ts - min_ts) * time_scale;

                raw_forward_weights_ptr[pos] = exp(tf);
                raw_backward_weights_ptr[pos] = exp(tb);
            }
        );

        // Step 4: Compute normalization sums per node using segmented reduce
        thrust::device_vector<double> node_forward_sums(node_index_capacity);
        thrust::device_vector<double> node_backward_sums(node_index_capacity);

        auto node_forward_sums_ptr = thrust::raw_pointer_cast(node_forward_sums.data());
        auto node_backward_sums_ptr = thrust::raw_pointer_cast(node_backward_sums.data());

        // Initialize sums to zero
        thrust::fill(node_forward_sums.begin(), node_forward_sums.end(), 0.0);
        thrust::fill(node_backward_sums.begin(), node_backward_sums.end(), 0.0);

        // Accumulate sums using atomic operations
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [group_to_node_ptr, raw_forward_weights_ptr, raw_backward_weights_ptr,
             node_forward_sums_ptr, node_backward_sums_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                atomicAdd(&node_forward_sums_ptr[node], raw_forward_weights_ptr[pos]);
                atomicAdd(&node_backward_sums_ptr[node], raw_backward_weights_ptr[pos]);
            }
        );

        // Step 5: Normalize weights and compute cumulative sums
        thrust::device_vector<double> normalized_forward_weights(outbound_groups_size);
        thrust::device_vector<double> normalized_backward_weights(outbound_groups_size);

        auto normalized_forward_weights_ptr = thrust::raw_pointer_cast(normalized_forward_weights.data());
        auto normalized_backward_weights_ptr = thrust::raw_pointer_cast(normalized_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [group_to_node_ptr, raw_forward_weights_ptr, raw_backward_weights_ptr,
             node_forward_sums_ptr, node_backward_sums_ptr,
             normalized_forward_weights_ptr, normalized_backward_weights_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const double forward_sum = node_forward_sums_ptr[node];
                const double backward_sum = node_backward_sums_ptr[node];

                normalized_forward_weights_ptr[pos] = raw_forward_weights_ptr[pos] / forward_sum;
                normalized_backward_weights_ptr[pos] = raw_backward_weights_ptr[pos] / backward_sum;
            }
        );

        // Step 6: Compute cumulative sums per node
        double* final_forward_weights = node_edge_index->outbound_forward_cumulative_weights_exponential;
        double* final_backward_weights = node_edge_index->outbound_backward_cumulative_weights_exponential;

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_forward_weights_ptr, final_forward_weights);

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_backward_weights_ptr, final_backward_weights
        );
        CUDA_KERNEL_CHECK("After outbound weights processing in update_temporal_weights_cuda");
    }

    // Process inbound weights (only backward)
    if (node_edge_index->node_group_inbound_offsets_size > 0) {
        node_index_capacity = node_edge_index->node_group_inbound_offsets_size - 1;

        MemoryView<size_t> inbound_offsets = get_timestamp_offset_vector(node_edge_index, false, true);
        const size_t inbound_groups_size = node_edge_index->node_ts_group_inbound_offsets_size;

        // Step 1: Create node mapping for each group position
        thrust::device_vector<size_t> group_to_node(inbound_groups_size);
        auto group_to_node_ptr = thrust::raw_pointer_cast(group_to_node.data());

        size_t* inbound_offsets_ptr = inbound_offsets.data;

        thrust::upper_bound(
            DEVICE_EXECUTION_POLICY,
            inbound_offsets_ptr,
            inbound_offsets_ptr + node_index_capacity + 1,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            group_to_node.begin()
        );

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            group_to_node.begin(),
            [] DEVICE(const size_t x) { return x - 1; }
        );

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [inbound_offsets_ptr, group_to_node_ptr] DEVICE(const size_t node) {
                const size_t in_start = inbound_offsets_ptr[node];
                const size_t in_end = inbound_offsets_ptr[node + 1];

                for (size_t pos = in_start; pos < in_end; ++pos) {
                    group_to_node_ptr[pos] = node;
                }
            }
        );

        // Step 2: Compute min/max timestamps per node
        thrust::device_vector<int64_t> node_min_ts(node_index_capacity);
        thrust::device_vector<int64_t> node_max_ts(node_index_capacity);
        thrust::device_vector<double> node_time_scale(node_index_capacity);

        auto node_min_ts_ptr = thrust::raw_pointer_cast(node_min_ts.data());
        auto node_max_ts_ptr = thrust::raw_pointer_cast(node_max_ts.data());
        auto node_time_scale_ptr = thrust::raw_pointer_cast(node_time_scale.data());

        int64_t* timestamps_ptr = edge_data->timestamps;
        size_t* inbound_indices_ptr = node_edge_index->node_ts_sorted_inbound_indices;
        size_t* inbound_group_indices_ptr = node_edge_index->node_ts_group_inbound_offsets;

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [timestamps_ptr, inbound_indices_ptr, inbound_group_indices_ptr,
             inbound_offsets_ptr, node_min_ts_ptr, node_max_ts_ptr, node_time_scale_ptr, timescale_bound]
            DEVICE(const size_t node) {
                const size_t in_start = inbound_offsets_ptr[node];
                const size_t in_end = inbound_offsets_ptr[node + 1];

                if (in_start >= in_end) {
                    node_min_ts_ptr[node] = 0;
                    node_max_ts_ptr[node] = 0;
                    node_time_scale_ptr[node] = 1.0;
                    return;
                }

                const size_t first_group_start = inbound_group_indices_ptr[in_start];
                const size_t last_group_start = inbound_group_indices_ptr[in_end - 1];
                const int64_t min_ts = timestamps_ptr[inbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[inbound_indices_ptr[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? timescale_bound / time_diff
                    : 1.0;

                node_min_ts_ptr[node] = min_ts;
                node_max_ts_ptr[node] = max_ts;
                node_time_scale_ptr[node] = time_scale;
            }
        );

        // Step 3: Compute raw weights in parallel
        thrust::device_vector<double> raw_backward_weights(inbound_groups_size);
        auto raw_backward_weights_ptr = thrust::raw_pointer_cast(raw_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [timestamps_ptr, inbound_indices_ptr, inbound_group_indices_ptr,
             group_to_node_ptr, node_min_ts_ptr, node_time_scale_ptr, raw_backward_weights_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const size_t edge_start = inbound_group_indices_ptr[pos];
                const int64_t group_ts = timestamps_ptr[inbound_indices_ptr[edge_start]];
                const int64_t min_ts = node_min_ts_ptr[node];
                const double time_scale = node_time_scale_ptr[node];

                const double tb = static_cast<double>(group_ts - min_ts) * time_scale;
                raw_backward_weights_ptr[pos] = exp(tb);
            }
        );

        // Step 4: Compute normalization sums per node
        thrust::device_vector<double> node_backward_sums(node_index_capacity);
        auto node_backward_sums_ptr = thrust::raw_pointer_cast(node_backward_sums.data());

        thrust::fill(node_backward_sums.begin(), node_backward_sums.end(), 0.0);

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [group_to_node_ptr, raw_backward_weights_ptr, node_backward_sums_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                atomicAdd(&node_backward_sums_ptr[node], raw_backward_weights_ptr[pos]);
            }
        );

        // Step 5: Normalize weights
        thrust::device_vector<double> normalized_backward_weights(inbound_groups_size);
        auto normalized_backward_weights_ptr = thrust::raw_pointer_cast(normalized_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [group_to_node_ptr, raw_backward_weights_ptr, node_backward_sums_ptr, normalized_backward_weights_ptr]
            DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const double backward_sum = node_backward_sums_ptr[node];
                normalized_backward_weights_ptr[pos] = raw_backward_weights_ptr[pos] / backward_sum;
            }
        );

        // Step 6: Compute cumulative sums per node
        double* final_backward_weights = node_edge_index->inbound_backward_cumulative_weights_exponential;

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_backward_weights_ptr, final_backward_weights
        );

        CUDA_KERNEL_CHECK("After inbound weights processing in update_temporal_weights_cuda");
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
        if (node_edge_index->node_group_outbound_offsets) {
            size_t *d_outbound_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_offsets, node_edge_index->node_group_outbound_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_offsets, node_edge_index->node_group_outbound_offsets, node_edge_index->
                    node_group_outbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.node_group_outbound_offsets = d_outbound_offsets;
        }

        if (node_edge_index->node_group_inbound_offsets) {
            size_t *d_inbound_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_offsets, node_edge_index->node_group_inbound_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_offsets, node_edge_index->node_group_inbound_offsets, node_edge_index->
                    node_group_inbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.node_group_inbound_offsets = d_inbound_offsets;
        }

        if (node_edge_index->node_ts_sorted_outbound_indices) {
            size_t *d_outbound_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_indices, node_edge_index->node_ts_sorted_outbound_indices_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_indices, node_edge_index->node_ts_sorted_outbound_indices, node_edge_index->
                    node_ts_sorted_outbound_indices_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.node_ts_sorted_outbound_indices = d_outbound_indices;
        }

        if (node_edge_index->node_ts_sorted_inbound_indices) {
            size_t *d_inbound_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_indices, node_edge_index->node_ts_sorted_inbound_indices_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_indices, node_edge_index->node_ts_sorted_inbound_indices, node_edge_index->
                    node_ts_sorted_inbound_indices_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_node_edge_index.node_ts_sorted_inbound_indices = d_inbound_indices;
        }

        if (node_edge_index->count_ts_group_per_node_outbound) {
            size_t *d_outbound_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_timestamp_group_offsets, node_edge_index->
                    count_ts_group_per_node_outbound_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_timestamp_group_offsets, node_edge_index->count_ts_group_per_node_outbound,
                    node_edge_index->count_ts_group_per_node_outbound_size * sizeof(size_t), cudaMemcpyHostToDevice
                ));
            temp_node_edge_index.count_ts_group_per_node_outbound = d_outbound_timestamp_group_offsets;
        }

        if (node_edge_index->count_ts_group_per_node_inbound) {
            size_t *d_inbound_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_timestamp_group_offsets, node_edge_index->count_ts_group_per_node_inbound_size
                    * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_timestamp_group_offsets, node_edge_index->count_ts_group_per_node_inbound,
                    node_edge_index->count_ts_group_per_node_inbound_size * sizeof(size_t), cudaMemcpyHostToDevice))
            ;
            temp_node_edge_index.count_ts_group_per_node_inbound = d_inbound_timestamp_group_offsets;
        }

        if (node_edge_index->node_ts_group_outbound_offsets) {
            size_t *d_outbound_timestamp_group_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_outbound_timestamp_group_indices, node_edge_index->
                    node_ts_group_outbound_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_outbound_timestamp_group_indices, node_edge_index->node_ts_group_outbound_offsets,
                    node_edge_index->node_ts_group_outbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice
                ));
            temp_node_edge_index.node_ts_group_outbound_offsets = d_outbound_timestamp_group_indices;
        }

        if (node_edge_index->node_ts_group_inbound_offsets) {
            size_t *d_inbound_timestamp_group_indices;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_inbound_timestamp_group_indices, node_edge_index->node_ts_group_inbound_offsets_size
                    * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_inbound_timestamp_group_indices, node_edge_index->node_ts_group_inbound_offsets,
                    node_edge_index->node_ts_group_inbound_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice))
            ;
            temp_node_edge_index.node_ts_group_inbound_offsets = d_inbound_timestamp_group_indices;
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
    allocate_node_group_offsets(node_edge_index, edge_data->active_node_ids_size, is_directed);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_group_offsets_cuda(node_edge_index, edge_data, is_directed);
    } else
    #endif
    {
        compute_node_group_offsets_std(node_edge_index, edge_data, is_directed);
    }

    // Step 2: Allocate and compute node edge indices
    allocate_node_ts_sorted_indices(node_edge_index, is_directed);

    const size_t num_edges = edge_data->timestamps_size;
    const size_t outbound_buffer_size = is_directed ? num_edges : num_edges * 2;

    int* outbound_node_ids = nullptr;
    allocate_memory(&outbound_node_ids, outbound_buffer_size, node_edge_index->use_gpu);

    int* inbound_node_ids = nullptr;
    allocate_memory(&inbound_node_ids, num_edges, node_edge_index->use_gpu);

    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        compute_node_ts_sorted_indices_cuda(
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
        compute_node_ts_sorted_indices_std(
            node_edge_index,
            edge_data,
            is_directed,
            outbound_buffer_size,
            outbound_node_ids,
            inbound_node_ids
        );
    }

    // Step 3 + 4: Compute timestamp group offsets AND group indices
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
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
        allocate_and_compute_node_ts_group_counts_and_offsets_std(
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

HOST size_t node_edge_index::get_memory_used(NodeEdgeIndexStore* node_edge_index) {
    size_t total_memory = 0;

    // Node group offset arrays
    total_memory += node_edge_index->node_group_outbound_offsets_size * sizeof(size_t);
    total_memory += node_edge_index->node_group_inbound_offsets_size * sizeof(size_t);

    // Node timestamp-sorted indices arrays
    total_memory += node_edge_index->node_ts_sorted_outbound_indices_size * sizeof(size_t);
    total_memory += node_edge_index->node_ts_sorted_inbound_indices_size * sizeof(size_t);

    // Timestamp group counts per node
    total_memory += node_edge_index->count_ts_group_per_node_outbound_size * sizeof(size_t);
    total_memory += node_edge_index->count_ts_group_per_node_inbound_size * sizeof(size_t);

    // Node timestamp group offset arrays
    total_memory += node_edge_index->node_ts_group_outbound_offsets_size * sizeof(size_t);
    total_memory += node_edge_index->node_ts_group_inbound_offsets_size * sizeof(size_t);

    // Cumulative weight arrays (if allocated for weight computation)
    total_memory += node_edge_index->outbound_forward_cumulative_weights_exponential_size * sizeof(double);
    total_memory += node_edge_index->outbound_backward_cumulative_weights_exponential_size * sizeof(double);
    total_memory += node_edge_index->inbound_backward_cumulative_weights_exponential_size * sizeof(double);

    return total_memory;
}
