#include "temporal_graph.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include "../common/cuda_sort.cuh"
#endif

#include "../common/nvtx_utils.h"
#include "../common/comparators.cuh"
#include "../common/parallel_algorithms.cuh"

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

HOST void temporal_graph::sort_and_merge_edges_std(TemporalGraphStore* graph, const size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* sources = graph->edge_data->sources;
    int* targets = graph->edge_data->targets;
    int64_t* timestamps = graph->edge_data->timestamps;

    // === Step 1: Build index arrays for old and new edges ===
    std::vector<int> old_indices(start_idx);
    std::vector<int> new_indices(new_edges_count);
    std::vector<int> merged_indices(total_size);

    #pragma omp parallel for
    for (int i = 0; i < start_idx; ++i)
        old_indices[i] = i;

    #pragma omp parallel for
    for (int i = 0; i < new_edges_count; ++i)
        new_indices[i] = static_cast<int>(start_idx) + i;

    // === Step 2: Sort new indices by timestamp ===
    parallel::sort(
        new_indices.begin(), new_indices.end(),
        TimestampComparator(timestamps));

    // === Step 3: Merge indices by timestamp ===
    parallel::merge(
        old_indices.begin(), old_indices.end(),
        new_indices.begin(), new_indices.end(),
        merged_indices.begin(),
        TimestampComparator(timestamps));

    // === Step 4: Allocate temporary arrays for merged output ===
    auto* merged_sources = new int[total_size];
    auto* merged_targets = new int[total_size];
    auto* merged_timestamps = new int64_t[total_size];

    // === Step 5: Gather edge data using merged indices ===
    #pragma omp parallel for
    for (int i = 0; i < total_size; ++i) {
        const int idx = merged_indices[i];
        merged_sources[i] = sources[idx];
        merged_targets[i] = targets[idx];
        merged_timestamps[i] = timestamps[idx];
    }

    // === Step 6: Copy merged data back to graph ===
    std::memcpy(sources, merged_sources, total_size * sizeof(int));
    std::memcpy(targets, merged_targets, total_size * sizeof(int));
    std::memcpy(timestamps, merged_timestamps, total_size * sizeof(int64_t));

    // === Step 7: Cleanup ===
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
        timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_inbound;
        timestamp_group_indices = graph->node_edge_index->node_ts_group_inbound_offsets;
        edge_indices = graph->node_edge_index->node_ts_sorted_inbound_indices;
    } else {
        timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_outbound;
        timestamp_group_indices = graph->node_edge_index->node_ts_group_outbound_offsets;
        edge_indices = graph->node_edge_index->node_ts_sorted_outbound_indices;
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
    const size_t *timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_outbound;
    size_t *timestamp_group_indices = graph->node_edge_index->node_ts_group_outbound_offsets;
    size_t *edge_indices = graph->node_edge_index->node_ts_sorted_outbound_indices;

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

NvtxRange r("ingestion_sort_merge");
HOST void temporal_graph::sort_and_merge_edges_cuda(TemporalGraphStore* graph, const size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* d_sources = graph->edge_data->sources;
    int* d_targets = graph->edge_data->targets;
    int64_t* d_timestamps = graph->edge_data->timestamps;

    // === Step 1: Prepare old and new index arrays ===
    thrust::device_vector<int> old_indices(static_cast<int>(start_idx));
    thrust::sequence(old_indices.begin(), old_indices.end(), 0);

    thrust::device_vector<int> new_indices(static_cast<int>(new_edges_count));
    thrust::sequence(new_indices.begin(), new_indices.end(), static_cast<int>(start_idx));

    // === Step 2: Sort new indices using CUB radix sort (direct keys) ===
    cub_radix_sort_values_by_keys(
        d_timestamps + start_idx,
        thrust::raw_pointer_cast(new_indices.data()),
        new_edges_count);

    // === Step 3: Merge with CUB ===
    thrust::device_vector<int> merged_indices(static_cast<int>(total_size));

    thrust::merge(
        old_indices.begin(), old_indices.end(),
        new_indices.begin(), new_indices.end(),
        merged_indices.begin(),
        TimestampComparator(d_timestamps)
    );

    // === Step 4: Allocate temporary merged arrays ===
    int* d_merged_sources = nullptr;
    int* d_merged_targets = nullptr;
    int64_t* d_merged_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_sources, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_targets, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_timestamps, total_size * sizeof(int64_t)));

    // === Step 5: Gather merged edge data ===
    // Sources
    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_sources),
        thrust::device_pointer_cast(d_merged_sources)
    );

    // Targets
    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_targets),
        thrust::device_pointer_cast(d_merged_targets)
    );

    // Timestamps
    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_timestamps),
        thrust::device_pointer_cast(d_merged_timestamps)
    );

    CUDA_KERNEL_CHECK("After thrust gather");

    // === Step 6: Copy back to graph ===
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_sources, d_merged_sources, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_targets, d_merged_targets, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamps, d_merged_timestamps, total_size * sizeof(int64_t), cudaMemcpyDeviceToDevice));

    // === Step 7: Cleanup ===
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
        timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_inbound;
        timestamp_group_indices = graph->node_edge_index->node_ts_group_inbound_offsets;
        edge_indices = graph->node_edge_index->node_ts_sorted_inbound_indices;
    } else {
        timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_outbound;
        timestamp_group_indices = graph->node_edge_index->node_ts_group_outbound_offsets;
        edge_indices = graph->node_edge_index->node_ts_sorted_outbound_indices;
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

    const size_t *timestamp_group_offsets = graph->node_edge_index->count_ts_group_per_node_outbound;
    size_t *timestamp_group_indices = graph->node_edge_index->node_ts_group_outbound_offsets;
    size_t *edge_indices = graph->node_edge_index->node_ts_sorted_outbound_indices;

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

HOST size_t temporal_graph::get_memory_used(TemporalGraphStore* graph) {
    return edge_data::get_memory_used(graph->edge_data) + node_edge_index::get_memory_used(graph->node_edge_index);
}
