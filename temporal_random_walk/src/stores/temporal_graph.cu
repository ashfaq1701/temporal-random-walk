#include "temporal_graph.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#endif

HOST void temporal_graph::update_temporal_weights(const TemporalGraphStore *graph) {
#ifdef HAS_CUDA
    if (graph->use_gpu) {
        edge_data::update_temporal_weights_cuda(graph->edge_data, graph->timescale_bound);
        node_edge_index::update_temporal_weights_cuda(graph->node_edge_index, graph->edge_data, graph->timescale_bound);
    } else
#endif
    {
        edge_data::update_temporal_weights_std(graph->edge_data, graph->timescale_bound);
        node_edge_index::update_temporal_weights_std(graph->node_edge_index, graph->edge_data, graph->timescale_bound);
    }
}

HOST DEVICE size_t temporal_graph::get_total_edges(const TemporalGraphStore *graph) {
    return edge_data::size(graph->edge_data);
}

HOST size_t temporal_graph::get_node_count(const TemporalGraphStore *graph) {
    return edge_data::active_node_count(graph->edge_data);
}

HOST int64_t temporal_graph::get_latest_timestamp(const TemporalGraphStore *graph) {
    return graph->latest_timestamp;
}

HOST DataBlock<int> temporal_graph::get_node_ids(const TemporalGraphStore *graph) {
    return edge_data::get_active_node_ids(graph->edge_data);
}

HOST DataBlock<Edge> temporal_graph::get_edges(const TemporalGraphStore *graph) {
    return edge_data::get_edges(graph->edge_data);
}

/**
 * Std implementations
 */

HOST void temporal_graph::sort_and_merge_edges_std(TemporalGraphStore *graph, const size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int *sources = graph->edge_data->sources;
    int *targets = graph->edge_data->targets;
    int64_t *timestamps = graph->edge_data->timestamps;

    // === Step 1: Create index array for new edges ===
    auto *indices = new size_t[new_edges_count];
    for (size_t i = 0; i < new_edges_count; ++i) {
        indices[i] = start_idx + i;
    }

    // === Step 2: Sort new edge indices by timestamp ===
    std::sort(indices, indices + new_edges_count,
              [timestamps](size_t i, size_t j) {
                  return timestamps[i] < timestamps[j];
              });

    // === Step 3: Allocate temp arrays for sorted new edges ===
    auto sorted_sources = new int[new_edges_count];
    auto sorted_targets = new int[new_edges_count];
    auto *sorted_timestamps = new int64_t[new_edges_count];

    for (size_t i = 0; i < new_edges_count; ++i) {
        const size_t idx = indices[i];
        sorted_sources[i] = sources[idx];
        sorted_targets[i] = targets[idx];
        sorted_timestamps[i] = timestamps[idx];
    }

    delete[] indices;

    // === Step 4: Allocate merge output arrays ===
    auto merged_sources = new int[total_size];
    auto merged_targets = new int[total_size];
    auto *merged_timestamps = new int64_t[total_size];

    // === Step 5: Merge old and new sorted edges ===
    size_t i = 0; // index in old edges
    size_t j = 0; // index in sorted new edges
    size_t k = 0; // output index

    while (i < start_idx && j < new_edges_count) {
        if (timestamps[i] <= sorted_timestamps[j]) {
            merged_sources[k] = sources[i];
            merged_targets[k] = targets[i];
            merged_timestamps[k] = timestamps[i];
            ++i;
        } else {
            merged_sources[k] = sorted_sources[j];
            merged_targets[k] = sorted_targets[j];
            merged_timestamps[k] = sorted_timestamps[j];
            ++j;
        }
        ++k;
    }

    while (i < start_idx) {
        merged_sources[k] = sources[i];
        merged_targets[k] = targets[i];
        merged_timestamps[k] = timestamps[i];
        ++i;
        ++k;
    }

    while (j < new_edges_count) {
        merged_sources[k] = sorted_sources[j];
        merged_targets[k] = sorted_targets[j];
        merged_timestamps[k] = sorted_timestamps[j];
        ++j;
        ++k;
    }

    // === Step 6: Copy merged arrays back to edge_data ===
    for (size_t m = 0; m < total_size; ++m) {
        sources[m] = merged_sources[m];
        targets[m] = merged_targets[m];
        timestamps[m] = merged_timestamps[m];
    }

    // === Step 7: Cleanup ===
    delete[] sorted_sources;
    delete[] sorted_targets;
    delete[] sorted_timestamps;
    delete[] merged_sources;
    delete[] merged_targets;
    delete[] merged_timestamps;
}

HOST void temporal_graph::delete_old_edges_std(TemporalGraphStore *graph) {
    if (graph->max_time_capacity <= 0 || edge_data::empty(graph->edge_data)) return;

    const int64_t cutoff_time = graph->latest_timestamp - graph->max_time_capacity;
    const auto it = std::upper_bound(
        graph->edge_data->timestamps,
        graph->edge_data->timestamps + graph->edge_data->timestamps_size,
        cutoff_time);
    if (it == graph->edge_data->timestamps) return;

    const int delete_count = static_cast<int>(it - graph->edge_data->timestamps);
    const size_t remaining = edge_data::size(graph->edge_data) - delete_count;

    if (remaining > 0) {
        remove_first_n_memory(
            &graph->edge_data->sources,
            graph->edge_data->sources_size,
            delete_count,
            graph->use_gpu);

        remove_first_n_memory(
            &graph->edge_data->targets,
            graph->edge_data->targets_size,
            delete_count,
            graph->use_gpu);

        remove_first_n_memory(
            &graph->edge_data->timestamps,
            graph->edge_data->timestamps_size,
            delete_count,
            graph->use_gpu);
    }

    edge_data::set_size(graph->edge_data, remaining);
}

HOST void temporal_graph::add_multiple_edges_std(
    TemporalGraphStore *graph,
    const int *sources,
    const int *targets,
    const int64_t *timestamps,
    const size_t num_new_edges) {

    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(graph->edge_data);

    // Find maximum timestamp in the new edges efficiently
    int64_t max_timestamp = graph->latest_timestamp;

    #pragma omp parallel for reduction(max:max_timestamp)
    for (size_t i = 0; i < num_new_edges; i++) {
        max_timestamp = std::max(max_timestamp, timestamps[i]);
    }

    graph->latest_timestamp = max_timestamp;

    // Add edges to edge data
    edge_data::add_edges(graph->edge_data, sources, targets, timestamps, num_new_edges);

    // Sort and merge new edges
    sort_and_merge_edges_std(graph, start_idx);

    // Handle time window
    if (graph->max_time_capacity > 0) {
        delete_old_edges_std(graph);
    }

    // Populate active node ids
    edge_data::populate_active_nodes_std(graph->edge_data);

    // Update timestamp groups
    edge_data::update_timestamp_groups_std(graph->edge_data);

    // Rebuild edge indices
    node_edge_index::rebuild(graph->node_edge_index, graph->edge_data, graph->is_directed);

    // Update temporal weights if enabled
    if (graph->enable_weight_computation) {
        update_temporal_weights(graph);
    }
}

HOST size_t temporal_graph::count_timestamps_less_than_std(const TemporalGraphStore *graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = std::lower_bound(
        graph->edge_data->unique_timestamps,
        graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size,
        timestamp);
    return it - graph->edge_data->unique_timestamps;
}

HOST size_t temporal_graph::count_timestamps_greater_than_std(const TemporalGraphStore *graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = std::upper_bound(
        graph->edge_data->unique_timestamps,
        graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size,
        timestamp);
    return (graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size) - it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_std(TemporalGraphStore *graph, const int node_id,
                                                       const int64_t timestamp) {
    size_t *timestamp_group_offsets;
    size_t *timestamp_group_indices;
    size_t *edge_indices;

    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    if (graph->is_directed) {
        timestamp_group_offsets = graph->node_edge_index->inbound_timestamp_group_offsets;
        timestamp_group_indices = graph->node_edge_index->inbound_timestamp_group_indices;
        edge_indices = graph->node_edge_index->inbound_indices;
    } else {
        timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
        timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
        edge_indices = graph->node_edge_index->outbound_indices;
    }

    const size_t group_start = timestamp_group_offsets[node_id];
    const size_t group_end = timestamp_group_offsets[node_id + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::lower_bound(
        timestamp_group_indices + static_cast<int>(group_start),
        timestamp_group_indices + static_cast<int>(group_end),
        timestamp,
        [graph, edge_indices](const size_t group_pos, const int64_t ts) {
            return graph->edge_data->timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_std(TemporalGraphStore *graph, const int node_id,
                                                          const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    // Used for forward walks
    const size_t *timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t *timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t *edge_indices = graph->node_edge_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[node_id];
    const size_t group_end = timestamp_group_offsets[node_id + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    const auto it = std::upper_bound(
        timestamp_group_indices + static_cast<int>(group_start),
        timestamp_group_indices + static_cast<int>(group_end),
        timestamp,
        [graph, edge_indices](const int64_t ts, const size_t group_pos) {
            return ts < graph->edge_data->timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, timestamp_group_indices + static_cast<int>(group_end));
}

/**
 * CUDA implementations
 */

#ifdef HAS_CUDA

HOST void temporal_graph::sort_and_merge_edges_cuda(TemporalGraphStore *graph, const size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int *d_sources = graph->edge_data->sources;
    int *d_targets = graph->edge_data->targets;
    int64_t *d_timestamps = graph->edge_data->timestamps;

    // === Step 1: Create index array for new edges ===
    size_t *d_indices = nullptr;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_indices, new_edges_count * sizeof(size_t)));
    thrust::sequence(DEVICE_EXECUTION_POLICY, d_indices, d_indices + new_edges_count, start_idx);

    // === Step 2: Sort new edge indices by timestamp ===
    thrust::sort(
        DEVICE_EXECUTION_POLICY,
        d_indices,
        d_indices + new_edges_count,
        [d_timestamps] __device__ (size_t i, size_t j) {
            return d_timestamps[i] < d_timestamps[j];
        });
    CUDA_KERNEL_CHECK("After thrust sort in sort_and_merge_edges_cuda");

    // === Step 3: Allocate temp arrays for sorted edges ===
    int *d_sorted_sources = nullptr, *d_sorted_targets = nullptr;
    int64_t *d_sorted_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sorted_sources, new_edges_count * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sorted_targets, new_edges_count * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sorted_timestamps, new_edges_count * sizeof(int64_t)));

    // === Step 4: Gather sorted new edge data ===
    thrust::gather(DEVICE_EXECUTION_POLICY,
                   d_indices, d_indices + new_edges_count,
                   d_sources, d_sorted_sources);
    CUDA_KERNEL_CHECK("After thrust gather sources in sort_and_merge_edges_cuda");

    thrust::gather(DEVICE_EXECUTION_POLICY,
                   d_indices, d_indices + new_edges_count,
                   d_targets, d_sorted_targets);
    CUDA_KERNEL_CHECK("After thrust gather targets in sort_and_merge_edges_cuda");

    thrust::gather(DEVICE_EXECUTION_POLICY,
                   d_indices, d_indices + new_edges_count,
                   d_timestamps, d_sorted_timestamps);
    CUDA_KERNEL_CHECK("After thrust gather timestamps in sort_and_merge_edges_cuda");

    clear_memory(&d_indices, true);

    // === Step 5: Allocate merge output arrays ===
    int *d_merged_sources = nullptr, *d_merged_targets = nullptr;
    int64_t *d_merged_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_sources, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_targets, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_timestamps, total_size * sizeof(int64_t)));

    // === Step 5: Merge timestamps + sources ===
    thrust::merge_by_key(
        DEVICE_EXECUTION_POLICY,
        d_timestamps, d_timestamps + start_idx, // keys1
        d_sorted_timestamps, d_sorted_timestamps + new_edges_count, // keys2
        d_sources, d_sorted_sources, // values
        d_merged_timestamps, d_merged_sources,
        thrust::less<int64_t>() // comparator
    );
    CUDA_KERNEL_CHECK("After first thrust merge_by_key in sort_and_merge_edges_cuda");

    // === Step 6: Merge timestamps + targets (reuse merged keys or discard) ===
    thrust::merge_by_key(
        DEVICE_EXECUTION_POLICY,
        d_timestamps, d_timestamps + start_idx,
        d_sorted_timestamps, d_sorted_timestamps + new_edges_count,
        d_targets, d_sorted_targets,
        /* out keys */ thrust::make_discard_iterator(), // don't need keys again
        d_merged_targets,
        thrust::less<int64_t>()
    );
    CUDA_KERNEL_CHECK("After second thrust merge_by_key in sort_and_merge_edges_cuda");

    // === Step 7: Copy merged arrays back to graph ===
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(d_sources, d_merged_sources, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(d_targets, d_merged_targets, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(d_timestamps, d_merged_timestamps, total_size * sizeof(int64_t), cudaMemcpyDeviceToDevice));

    // === Step 8: Cleanup ===
    clear_memory(&d_sorted_sources, true);
    clear_memory(&d_sorted_targets, true);
    clear_memory(&d_sorted_timestamps, true);
    clear_memory(&d_merged_sources, true);
    clear_memory(&d_merged_targets, true);
    clear_memory(&d_merged_timestamps, true);
}

HOST void temporal_graph::delete_old_edges_cuda(TemporalGraphStore *graph) {
    if (graph->max_time_capacity <= 0 || edge_data::empty(graph->edge_data)) return;

    const int64_t cutoff_time = graph->latest_timestamp - graph->max_time_capacity;

    // Find the index of the first timestamp greater than cutoff_time
    int64_t *timestamps_ptr = graph->edge_data->timestamps;
    const size_t timestamps_size = graph->edge_data->timestamps_size;

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamps_ptr),
        thrust::device_pointer_cast(timestamps_ptr + timestamps_size),
        cutoff_time
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in delete_old_edges_cuda");

    if (it == thrust::device_pointer_cast(timestamps_ptr)) return;

    const int delete_count = static_cast<int>(it - thrust::device_pointer_cast(timestamps_ptr));
    const size_t remaining = graph->edge_data->timestamps_size - delete_count;

    if (remaining > 0) {
        // Move edges using thrust::copy
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(graph->edge_data->sources + delete_count),
            thrust::device_pointer_cast(graph->edge_data->sources + graph->edge_data->sources_size),
            thrust::device_pointer_cast(graph->edge_data->sources)
        );
        CUDA_KERNEL_CHECK("After thrust copy sources in delete_old_edges_cuda");

        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(graph->edge_data->targets + delete_count),
            thrust::device_pointer_cast(graph->edge_data->targets + graph->edge_data->targets_size),
            thrust::device_pointer_cast(graph->edge_data->targets)
        );
        CUDA_KERNEL_CHECK("After thrust copy targets in delete_old_edges_cuda");

        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(graph->edge_data->timestamps + delete_count),
            thrust::device_pointer_cast(graph->edge_data->timestamps + graph->edge_data->timestamps_size),
            thrust::device_pointer_cast(graph->edge_data->timestamps)
        );
        CUDA_KERNEL_CHECK("After thrust copy timestamps in delete_old_edges_cuda");
    }

    // Update sizes
    edge_data::set_size(graph->edge_data, remaining);
}

HOST void temporal_graph::add_multiple_edges_cuda(
    TemporalGraphStore *graph,
    const int *sources,
    const int *targets,
    const int64_t *timestamps,
    const size_t num_new_edges) {
    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(graph->edge_data);

    // Allocate CUDA memory for sources, targets, and timestamps
    int *d_sources = nullptr;
    int *d_targets = nullptr;
    int64_t *d_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sources, num_new_edges * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_targets, num_new_edges * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_timestamps, num_new_edges * sizeof(int64_t)));

    // Copy data directly to device
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_sources, sources, num_new_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_targets, targets, num_new_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamps, timestamps, num_new_edges * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Find maximum timestamp using thrust::reduce
    const int64_t current_max = graph->latest_timestamp;
    const int64_t timestamps_max = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(d_timestamps),
        thrust::device_pointer_cast(d_timestamps + num_new_edges),
        current_max,
        thrust::maximum<int64_t>()
    );
    CUDA_KERNEL_CHECK("After thrust reduce in add_multiple_edges_cuda");

    // Update latest timestamp
    graph->latest_timestamp = timestamps_max;

    // Add edges to edge data
    edge_data::add_edges(graph->edge_data, d_sources, d_targets, d_timestamps, num_new_edges);

    // Sort and merge new edges
    sort_and_merge_edges_cuda(graph, start_idx);

    // Handle time window
    if (graph->max_time_capacity > 0) {
        delete_old_edges_cuda(graph);
    }

    // Populate active node ids
    edge_data::populate_active_nodes_cuda(graph->edge_data);

    // Update timestamp groups
    edge_data::update_timestamp_groups_cuda(graph->edge_data);

    // Rebuild edge indices
    node_edge_index::rebuild(graph->node_edge_index, graph->edge_data, graph->is_directed);

    // Update temporal weights if enabled
    if (graph->enable_weight_computation) {
        update_temporal_weights(graph);
    }

    // Clean up
    CUDA_CHECK_AND_CLEAR(cudaFree(d_sources));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_targets));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_timestamps));
}

HOST size_t temporal_graph::count_timestamps_less_than_cuda(const TemporalGraphStore *graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps),
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size),
        timestamp
    );
    CUDA_KERNEL_CHECK("After thrust lower_bound in count_timestamps_less_than_cuda");

    return it - thrust::device_pointer_cast(graph->edge_data->unique_timestamps);
}

HOST size_t temporal_graph::count_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps),
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size),
        timestamp
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_timestamps_greater_than_cuda");

    return thrust::device_pointer_cast(graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size) -
           it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_cuda(const TemporalGraphStore *graph, const int node_id,
                                                        const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    size_t *timestamp_group_offsets;
    size_t *timestamp_group_indices;
    size_t *edge_indices;

    if (graph->is_directed) {
        timestamp_group_offsets = graph->node_edge_index->inbound_timestamp_group_offsets;
        timestamp_group_indices = graph->node_edge_index->inbound_timestamp_group_indices;
        edge_indices = graph->node_edge_index->inbound_indices;
    } else {
        timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
        timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
        edge_indices = graph->node_edge_index->outbound_indices;
    }

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&group_start, timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&group_end, timestamp_group_offsets + node_id + 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    int64_t *timestamps_ptr = graph->edge_data->timestamps;

    // Binary search on group indices
    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const size_t group_pos, const int64_t ts) {
            return timestamps_ptr[edge_indices[group_pos]] < ts;
        });
    CUDA_KERNEL_CHECK("After thrust lower_bound in count_node_timestamps_less_than_cuda");

    return thrust::distance(thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_cuda(const TemporalGraphStore *graph, const int node_id,
                                                           const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    const size_t *timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t *timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t *edge_indices = graph->node_edge_index->outbound_indices;

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&group_start, timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&group_end, timestamp_group_offsets + (node_id + 1), sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    int64_t *timestamps_ptr = graph->edge_data->timestamps;

    // Binary search on group indices
    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const int64_t ts, const size_t group_pos) {
            return ts < timestamps_ptr[edge_indices[group_pos]];
        });
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_node_timestamps_greater_than_cuda");

    return thrust::distance(it, thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end));
}

HOST TemporalGraphStore* temporal_graph::to_device_ptr(const TemporalGraphStore *graph) {
    // Create a new TemporalGraph object on the device
    TemporalGraphStore *device_graph;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_graph, sizeof(TemporalGraphStore)));

    // Create a temporary copy to modify
    TemporalGraphStore temp_graph = *graph;

    // Copy substructures to device
    if (graph->edge_data) {
        temp_graph.edge_data = edge_data::to_device_ptr(graph->edge_data);
    }

    if (graph->node_edge_index) {
        temp_graph.node_edge_index = node_edge_index::to_device_ptr(graph->node_edge_index);
    }

    // Make sure use_gpu is set to true
    temp_graph.use_gpu = true;

    // Copy the updated struct to device
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_graph, &temp_graph, sizeof(TemporalGraphStore), cudaMemcpyHostToDevice));

    temp_graph.owns_data = false;

    return device_graph;
}

HOST void temporal_graph::free_device_pointers(TemporalGraphStore *d_graph) {
    if (!d_graph) return;

    // Copy the struct from device to host to access pointers
    TemporalGraphStore h_graph;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&h_graph, d_graph, sizeof(TemporalGraphStore), cudaMemcpyDeviceToHost));
    h_graph.owns_data = false;

    // Free only the nested device pointers (not their underlying data)
    if (h_graph.edge_data) clear_memory(&h_graph.edge_data, true);
    if (h_graph.node_edge_index) clear_memory(&h_graph.node_edge_index, true);

    clear_memory(&d_graph, true);
}

#endif
