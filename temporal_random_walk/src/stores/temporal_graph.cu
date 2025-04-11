#include "temporal_graph.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#endif

#include "../utils/random.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"
#include "../random/pickers.cuh"
#include "edge_data.cuh"
#include "node_edge_index.cuh"

HOST void temporal_graph::update_temporal_weights(const TemporalGraphStore* graph) {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        edge_data::update_temporal_weights_cuda(graph->edge_data, graph->timescale_bound);
        node_edge_index::update_temporal_weights_cuda(graph->node_edge_index, graph->edge_data, graph->timescale_bound);
    }
    else
    #endif
    {
        edge_data::update_temporal_weights_std(graph->edge_data, graph->timescale_bound);
        node_edge_index::update_temporal_weights_std(graph->node_edge_index, graph->edge_data, graph->timescale_bound);
    }
}

HOST DEVICE size_t temporal_graph::get_total_edges(const TemporalGraphStore* graph) {
    return edge_data::size(graph->edge_data);
}

HOST size_t temporal_graph::get_node_count(const TemporalGraphStore* graph) {
    return edge_data::active_node_count(graph->edge_data);
}

HOST int64_t temporal_graph::get_latest_timestamp(const TemporalGraphStore* graph) {
    return graph->latest_timestamp;
}

HOST DataBlock<int> temporal_graph::get_node_ids(const TemporalGraphStore* graph) {
    return edge_data::get_active_node_ids(graph->edge_data);
}

HOST DataBlock<Edge> temporal_graph::get_edges(const TemporalGraphStore* graph) {
    return edge_data::get_edges(graph->edge_data);
}

HOST void temporal_graph::add_multiple_edges_std(
    TemporalGraphStore* graph,
    const Edge* new_edges,
    const size_t num_new_edges) {

    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(graph->edge_data);

    // Extract sources, targets, and timestamps from new edges
    auto sources = new int[num_new_edges];
    auto targets = new int[num_new_edges];
    auto* timestamps = new int64_t[num_new_edges];

    for (size_t i = 0; i < num_new_edges; i++) {
        if (!graph->is_directed && new_edges[i].u > new_edges[i].i) {
            // For undirected graphs, ensure source < target
            sources[i] = new_edges[i].i;
            targets[i] = new_edges[i].u;
        } else {
            sources[i] = new_edges[i].u;
            targets[i] = new_edges[i].i;
        }
        timestamps[i] = new_edges[i].ts;

        // Update latest timestamp
        graph->latest_timestamp = std::max(graph->latest_timestamp, new_edges[i].ts);
    }

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

    // Clean up
    delete[] sources;
    delete[] targets;
    delete[] timestamps;
}

HOST void temporal_graph::sort_and_merge_edges_std(TemporalGraphStore* graph, size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* sources = graph->edge_data->sources;
    int* targets = graph->edge_data->targets;
    int64_t* timestamps = graph->edge_data->timestamps;

    // === Step 1: Create index array for new edges ===
    auto* indices = new size_t[new_edges_count];
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
    auto* sorted_timestamps = new int64_t[new_edges_count];

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
    auto* merged_timestamps = new int64_t[total_size];

    // === Step 5: Merge old and new sorted edges ===
    size_t i = 0;  // index in old edges
    size_t j = 0;  // index in sorted new edges
    size_t k = 0;  // output index

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
        ++i; ++k;
    }

    while (j < new_edges_count) {
        merged_sources[k] = sorted_sources[j];
        merged_targets[k] = sorted_targets[j];
        merged_timestamps[k] = sorted_timestamps[j];
        ++j; ++k;
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

HOST void temporal_graph::delete_old_edges_std(TemporalGraphStore* graph) {
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

HOST size_t temporal_graph::count_timestamps_less_than_std(const TemporalGraphStore* graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = std::lower_bound(
        graph->edge_data->unique_timestamps,
        graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size,
        timestamp);
    return it - graph->edge_data->unique_timestamps;
}

HOST size_t temporal_graph::count_timestamps_greater_than_std(const TemporalGraphStore* graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = std::upper_bound(
        graph->edge_data->unique_timestamps,
        graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size,
        timestamp);
    return (graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size) - it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_std(TemporalGraphStore* graph, const int node_id, const int64_t timestamp) {
    size_t* timestamp_group_offsets;
    size_t* timestamp_group_indices;
    size_t* edge_indices;

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
        [graph, edge_indices](size_t group_pos, int64_t ts)
        {
            return graph->edge_data->timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_std(TemporalGraphStore* graph, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    // Used for forward walks
    size_t* timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t* timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t* edge_indices = graph->node_edge_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[node_id];
    const size_t group_end = timestamp_group_offsets[node_id + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    const auto it = std::upper_bound(
        timestamp_group_indices + static_cast<int>(group_start),
        timestamp_group_indices + static_cast<int>(group_end),
        timestamp,
        [graph, edge_indices](int64_t ts, size_t group_pos)
        {
            return ts < graph->edge_data->timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, timestamp_group_indices + static_cast<int>(group_end));
}

#ifdef HAS_CUDA
HOST void temporal_graph::add_multiple_edges_cuda(
    TemporalGraphStore* graph,
    const Edge* new_edges,
    const size_t num_new_edges) {

    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(graph->edge_data);

    // Allocate CUDA memory for sources, targets, and timestamps
    int* d_sources = nullptr;
    int* d_targets = nullptr;
    int64_t* d_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sources, num_new_edges * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_targets, num_new_edges * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_timestamps, num_new_edges * sizeof(int64_t)));

    // Copy edges to device if they're not already there
    Edge* d_edges = nullptr;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_edges, num_new_edges * sizeof(Edge)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_edges, new_edges, num_new_edges * sizeof(Edge), cudaMemcpyHostToDevice));

    // Process edges in parallel and find maximum timestamp
    const int64_t host_latest_timestamp = graph->latest_timestamp;
    int64_t* d_latest_timestamp = nullptr;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_latest_timestamp, sizeof(int64_t)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_latest_timestamp, &host_latest_timestamp, sizeof(int64_t), cudaMemcpyHostToDevice));

    const bool is_directed = graph->is_directed;

    // Launch kernel to process edges in parallel
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_new_edges),
        [d_sources, d_targets, d_timestamps, d_edges, d_latest_timestamp, is_directed] DEVICE (const size_t i) {
            if (!is_directed && d_edges[i].u > d_edges[i].i) {
                // For undirected graphs, ensure source < target
                d_sources[i] = d_edges[i].i;
                d_targets[i] = d_edges[i].u;
            } else {
                d_sources[i] = d_edges[i].u;
                d_targets[i] = d_edges[i].i;
            }
            d_timestamps[i] = d_edges[i].ts;

            // Update latest timestamp using atomic max
            atomicMax(reinterpret_cast<unsigned long long*>(d_latest_timestamp),
                      static_cast<unsigned long long>(d_edges[i].ts));
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each in add_multiple_edges_cuda");

    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&graph->latest_timestamp, d_latest_timestamp, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_latest_timestamp));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_edges));

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

HOST void temporal_graph::sort_and_merge_edges_cuda(TemporalGraphStore* graph, const size_t start_idx) {
    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* d_sources = graph->edge_data->sources;
    int* d_targets = graph->edge_data->targets;
    int64_t* d_timestamps = graph->edge_data->timestamps;

    // === Step 1: Create index array for new edges ===
    size_t* d_indices = nullptr;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_indices, new_edges_count * sizeof(size_t)));
    thrust::sequence(thrust::device, d_indices, d_indices + new_edges_count, start_idx);

    // === Step 2: Sort new edge indices by timestamp ===
    thrust::sort(
        thrust::device,
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
    thrust::gather(thrust::device,
                   d_indices, d_indices + new_edges_count,
                   d_sources, d_sorted_sources);
    CUDA_KERNEL_CHECK("After thrust gather sources in sort_and_merge_edges_cuda");

    thrust::gather(thrust::device,
                   d_indices, d_indices + new_edges_count,
                   d_targets, d_sorted_targets);
    CUDA_KERNEL_CHECK("After thrust gather targets in sort_and_merge_edges_cuda");

    thrust::gather(thrust::device,
                   d_indices, d_indices + new_edges_count,
                   d_timestamps, d_sorted_timestamps);
    CUDA_KERNEL_CHECK("After thrust gather timestamps in sort_and_merge_edges_cuda");

    clear_memory(&d_indices, true);

    // === Step 5: Allocate merge output arrays ===
    int* d_merged_sources = nullptr, *d_merged_targets = nullptr;
    int64_t* d_merged_timestamps = nullptr;

    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_sources, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_targets, total_size * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_merged_timestamps, total_size * sizeof(int64_t)));

    // === Step 6: Merge old and new sorted edges ===
    const auto first1 = thrust::make_zip_iterator(
        thrust::make_tuple(d_sources, d_targets, d_timestamps));
    const auto last1 = first1 + start_idx;

    const auto first2 = thrust::make_zip_iterator(
        thrust::make_tuple(d_sorted_sources, d_sorted_targets, d_sorted_timestamps));
    const auto last2 = first2 + new_edges_count;

    const auto result = thrust::make_zip_iterator(
        thrust::make_tuple(d_merged_sources, d_merged_targets, d_merged_timestamps));

    thrust::merge(thrust::device,
                  first1, last1,
                  first2, last2,
                  result,
                  [] __device__ (const thrust::tuple<int, int, int64_t>& a,
                                 const thrust::tuple<int, int, int64_t>& b) {
                      return thrust::get<2>(a) < thrust::get<2>(b);
                  });
    CUDA_KERNEL_CHECK("After thrust merge in sort_and_merge_edges_cuda");

    // === Step 7: Copy merged arrays back to graph ===
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_sources, d_merged_sources, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_targets, d_merged_targets, total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamps, d_merged_timestamps, total_size * sizeof(int64_t), cudaMemcpyDeviceToDevice));

    // === Step 8: Cleanup ===
    clear_memory(&d_sorted_sources, true);
    clear_memory(&d_sorted_targets, true);
    clear_memory(&d_sorted_timestamps, true);
    clear_memory(&d_merged_sources, true);
    clear_memory(&d_merged_targets, true);
    clear_memory(&d_merged_timestamps, true);
}

HOST void temporal_graph::delete_old_edges_cuda(TemporalGraphStore* graph) {
    if (graph->max_time_capacity <= 0 || edge_data::empty(graph->edge_data)) return;

    const int64_t cutoff_time = graph->latest_timestamp - graph->max_time_capacity;

    // Find the index of the first timestamp greater than cutoff_time
    int64_t* timestamps_ptr = graph->edge_data->timestamps;
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

HOST size_t temporal_graph::count_timestamps_less_than_cuda(const TemporalGraphStore* graph, const int64_t timestamp) {
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

HOST size_t temporal_graph::count_timestamps_greater_than_cuda(const TemporalGraphStore* graph, const int64_t timestamp) {
    if (edge_data::empty(graph->edge_data)) return 0;

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps),
        thrust::device_pointer_cast(graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size),
        timestamp
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_timestamps_greater_than_cuda");

    return thrust::device_pointer_cast(graph->edge_data->unique_timestamps + graph->edge_data->unique_timestamps_size) - it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_cuda(const TemporalGraphStore* graph, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    size_t* timestamp_group_offsets;
    size_t* timestamp_group_indices;
    size_t* edge_indices;

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
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start, timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end, timestamp_group_offsets + node_id + 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    int64_t* timestamps_ptr = graph->edge_data->timestamps;

    // Binary search on group indices
    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const size_t group_pos, const int64_t ts)
        {
            return timestamps_ptr[edge_indices[group_pos]] < ts;
        });
    CUDA_KERNEL_CHECK("After thrust lower_bound in count_node_timestamps_less_than_cuda");

    return thrust::distance(thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_cuda(const TemporalGraphStore* graph, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return 0;
    }

    const size_t* timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t* timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t* edge_indices = graph->node_edge_index->outbound_indices;

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start, timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end, timestamp_group_offsets + (node_id + 1), sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    int64_t* timestamps_ptr = graph->edge_data->timestamps;

    // Binary search on group indices
    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const int64_t ts, const size_t group_pos)
        {
            return ts < timestamps_ptr[edge_indices[group_pos]];
        });
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_node_timestamps_greater_than_cuda");

    return thrust::distance(it, thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end));
}
#endif

HOST Edge temporal_graph::get_edge_at_host(
    const TemporalGraphStore* graph,
    const RandomPickerType picker_type,
    const int64_t timestamp,
    const bool forward) {

    if (edge_data::empty(graph->edge_data)) return Edge{-1, -1, -1};

    const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
    if (num_groups == 0) return Edge{-1, -1, -1};

    long group_idx;
    if (timestamp != -1) {
        if (forward) {
            const size_t first_group = edge_data::find_group_after_timestamp(graph->edge_data, timestamp);
            const size_t available_groups = num_groups - first_group;
            if (available_groups == 0) return Edge{-1, -1, -1};

            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_host(
                    picker_type, 0, static_cast<int>(available_groups), false);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = static_cast<long>(first_group + index);
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->edge_data->forward_cumulative_weights_exponential,
                    graph->edge_data->forward_cumulative_weights_exponential_size,
                    first_group,
                    num_groups);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        } else {
            const size_t last_group = edge_data::find_group_before_timestamp(graph->edge_data, timestamp);
            if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

            const size_t available_groups = last_group + 1;
            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_host(
                    picker_type, 0, static_cast<int>(available_groups), true);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->edge_data->backward_cumulative_weights_exponential,
                    graph->edge_data->backward_cumulative_weights_exponential_size,
                    0,
                    last_group + 1);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        if (random_pickers::is_index_based_picker(picker_type)) {
            const auto index = random_pickers::pick_using_index_based_picker_host(
                picker_type, 0, static_cast<int>(num_groups), !forward);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_idx = index;
        } else {
            if (forward) {
                group_idx = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->edge_data->forward_cumulative_weights_exponential,
                    graph->edge_data->forward_cumulative_weights_exponential_size,
                    0,
                    num_groups);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->edge_data->backward_cumulative_weights_exponential,
                    graph->edge_data->backward_cumulative_weights_exponential_size,
                    0,
                    num_groups);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get selected group's boundaries
    const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
    if (group_range.from == group_range.to) {
        return Edge{-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_range.from +
        generate_random_number_bounded_by_host(static_cast<int>(group_range.to - group_range.from));

    return Edge {
        graph->edge_data->sources[random_idx],
        graph->edge_data->targets[random_idx],
        graph->edge_data->timestamps[random_idx]
    };
}

HOST Edge temporal_graph::get_node_edge_at_host(
    TemporalGraphStore* graph,
    const int node_id,
    const RandomPickerType picker_type,
    const int64_t timestamp,
    const bool forward) {

    if (!edge_data::is_node_active_host(graph->edge_data, node_id)) {
        return Edge{-1, -1, -1};
    }

    // Get appropriate node indices based on direction and graph type
    const size_t* timestamp_group_offsets = forward
        ? graph->node_edge_index->outbound_timestamp_group_offsets
        : (graph->is_directed ? graph->node_edge_index->inbound_timestamp_group_offsets : graph->node_edge_index->outbound_timestamp_group_offsets);

    size_t* timestamp_group_indices = forward
        ? graph->node_edge_index->outbound_timestamp_group_indices
        : (graph->is_directed ? graph->node_edge_index->inbound_timestamp_group_indices : graph->node_edge_index->outbound_timestamp_group_indices);

    size_t* edge_indices = forward
        ? graph->node_edge_index->outbound_indices
        : (graph->is_directed ? graph->node_edge_index->inbound_indices : graph->node_edge_index->outbound_indices);

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[node_id];
    const size_t group_end_offset = timestamp_group_offsets[node_id + 1];
    if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

    long group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            const auto it = std::upper_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [graph, edge_indices](int64_t ts, size_t pos) {
                    return ts < graph->edge_data->timestamps[edge_indices[pos]];
                });

            // Count available groups after timestamp
            const size_t available = std::distance(
                it,
                timestamp_group_indices + static_cast<int>(group_end_offset));
            if (available == 0) return Edge{-1, -1, -1};

            const size_t start_pos = it - timestamp_group_indices;
            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_host(
                    picker_type, 0, static_cast<int>(available), false);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = static_cast<long>(start_pos) + index;
            }
            else {
                group_pos = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                    start_pos,
                    group_end_offset);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        } else {
            // Find first group >= timestamp
            auto it = std::lower_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [graph, edge_indices](size_t pos, int64_t ts) {
                    return graph->edge_data->timestamps[edge_indices[pos]] < ts;
                });

            const size_t available = std::distance(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                it);
            if (available == 0) return Edge{-1, -1, -1};

            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_host(
                    picker_type, 0, static_cast<int>(available), true);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = static_cast<long>((it - timestamp_group_indices) - 1 - (available - index - 1));
            }
            else {
                double* weights = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                size_t weights_size = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size;

                group_pos = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    weights,
                    weights_size,
                    group_start_offset,
                    static_cast<size_t>(it - timestamp_group_indices)
                );
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return Edge{-1, -1, -1};

        if (random_pickers::is_index_based_picker(picker_type)) {
            const auto index = random_pickers::pick_using_index_based_picker_host(
                picker_type, 0, static_cast<int>(num_groups), !forward);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_pos = forward
                ? static_cast<long>(group_start_offset + index)
                : static_cast<long>(group_end_offset - 1 - (num_groups - index - 1));
        }
        else {
            if (forward) {
                group_pos = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                    group_start_offset,
                    group_end_offset);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
            else {
                double* weights = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                size_t weights_size = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size;

                group_pos = random_pickers::pick_using_weight_based_picker_host(
                    picker_type,
                    weights,
                    weights_size,
                    group_start_offset,
                    group_end_offset);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get edge range for selected group
    const size_t edge_start = timestamp_group_indices[group_pos];
    size_t edge_end;

    if (group_pos + 1 < group_end_offset) {
        edge_end = timestamp_group_indices[group_pos + 1];
    } else {
        if (forward) {
            edge_end = graph->node_edge_index->outbound_offsets[node_id + 1];
        } else {
            edge_end = graph->is_directed
                ? graph->node_edge_index->inbound_offsets[node_id + 1]
                : graph->node_edge_index->outbound_offsets[node_id + 1];
        }
    }

    // Validate range before random selection
    size_t edge_indices_size = forward
        ? graph->node_edge_index->outbound_indices_size
        : (graph->is_directed ? graph->node_edge_index->inbound_indices_size : graph->node_edge_index->outbound_indices_size);

    if (edge_start >= edge_end || edge_start >= edge_indices_size || edge_end > edge_indices_size) {
        return Edge{-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by_host(static_cast<int>(edge_end - edge_start))];

    return Edge {
        graph->edge_data->sources[edge_idx],
        graph->edge_data->targets[edge_idx],
        graph->edge_data->timestamps[edge_idx]
    };
}

#ifdef HAS_CUDA
DEVICE Edge temporal_graph::get_edge_at_device(
        const TemporalGraphStore* graph,
        const RandomPickerType picker_type,
        const int64_t timestamp,
        const bool forward,
        curandState* rand_state) {

    if (edge_data::empty(graph->edge_data)) return Edge{-1, -1, -1};

    const size_t num_groups = edge_data::get_timestamp_group_count(graph->edge_data);
    if (num_groups == 0) return Edge{-1, -1, -1};

    long group_idx;
    if (timestamp != -1) {
        if (forward) {
            const size_t first_group = edge_data::find_group_after_timestamp_device(graph->edge_data, timestamp);
            const size_t available_groups = num_groups - first_group;
            if (available_groups == 0) return Edge{-1, -1, -1};

            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_device(
                    picker_type, 0, static_cast<int>(available_groups), false, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = static_cast<long>(first_group + index);
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->edge_data->forward_cumulative_weights_exponential,
                    graph->edge_data->forward_cumulative_weights_exponential_size,
                    first_group,
                    num_groups,
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        } else {
            const size_t last_group = edge_data::find_group_before_timestamp_device(graph->edge_data, timestamp);
            if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

            const size_t available_groups = last_group + 1;
            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_device(
                    picker_type, 0, static_cast<int>(available_groups), true, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->edge_data->backward_cumulative_weights_exponential,
                    graph->edge_data->backward_cumulative_weights_exponential_size,
                    0,
                    last_group + 1,
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        if (random_pickers::is_index_based_picker(picker_type)) {
            const auto index = random_pickers::pick_using_index_based_picker_device(
                picker_type, 0, static_cast<int>(num_groups), !forward, rand_state);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_idx = index;
        } else {
            if (forward) {
                group_idx = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->edge_data->forward_cumulative_weights_exponential,
                    graph->edge_data->forward_cumulative_weights_exponential_size,
                    0,
                    num_groups,
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
            else {
                group_idx = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->edge_data->backward_cumulative_weights_exponential,
                    graph->edge_data->backward_cumulative_weights_exponential_size,
                    0,
                    num_groups,
                    rand_state);
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get selected group's boundaries
    const SizeRange group_range = edge_data::get_timestamp_group_range(graph->edge_data, group_idx);
    if (group_range.from == group_range.to) {
        return Edge{-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_range.from +
        generate_random_number_bounded_by_device(static_cast<int>(group_range.to - group_range.from), rand_state);

    return Edge {
        graph->edge_data->sources[random_idx],
        graph->edge_data->targets[random_idx],
        graph->edge_data->timestamps[random_idx]
    };
}

DEVICE Edge temporal_graph::get_node_edge_at_device(
        TemporalGraphStore* graph,
        const int node_id,
        const RandomPickerType picker_type,
        const int64_t timestamp,
        const bool forward,
        curandState* rand_state) {

    if (!edge_data::is_node_active_device(graph->edge_data, node_id)) {
        return Edge{-1, -1, -1};
    }

    // Get appropriate node indices based on direction and graph type
    const size_t* timestamp_group_offsets = forward
        ? graph->node_edge_index->outbound_timestamp_group_offsets
        : (graph->is_directed ? graph->node_edge_index->inbound_timestamp_group_offsets : graph->node_edge_index->outbound_timestamp_group_offsets);

    size_t* timestamp_group_indices = forward
        ? graph->node_edge_index->outbound_timestamp_group_indices
        : (graph->is_directed ? graph->node_edge_index->inbound_timestamp_group_indices : graph->node_edge_index->outbound_timestamp_group_indices);

    size_t* edge_indices = forward
        ? graph->node_edge_index->outbound_indices
        : (graph->is_directed ? graph->node_edge_index->inbound_indices : graph->node_edge_index->outbound_indices);

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[node_id];
    const size_t group_end_offset = timestamp_group_offsets[node_id + 1];
    if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

    long group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            const auto it = cuda::std::upper_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [graph, edge_indices](int64_t ts, size_t pos) {
                    return ts < graph->edge_data->timestamps[edge_indices[pos]];
                });

            // Count available groups after timestamp
            const size_t available = cuda::std::distance(
                it,
                timestamp_group_indices + static_cast<int>(group_end_offset));
            if (available == 0) return Edge{-1, -1, -1};

            const size_t start_pos = it - timestamp_group_indices;
            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_device(
                    picker_type, 0, static_cast<int>(available), false, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = static_cast<long>(start_pos) + index;
            }
            else {
                group_pos = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                    start_pos,
                    group_end_offset,
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        } else {
            // Find first group >= timestamp
            auto it = cuda::std::lower_bound(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                timestamp_group_indices + static_cast<int>(group_end_offset),
                timestamp,
                [graph, edge_indices](size_t pos, int64_t ts) {
                    return graph->edge_data->timestamps[edge_indices[pos]] < ts;
                });

            const size_t available = cuda::std::distance(
                timestamp_group_indices + static_cast<int>(group_start_offset),
                it);
            if (available == 0) return Edge{-1, -1, -1};

            if (random_pickers::is_index_based_picker(picker_type)) {
                const auto index = random_pickers::pick_using_index_based_picker_device(
                    picker_type, 0, static_cast<int>(available), true, rand_state);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = static_cast<long>((it - timestamp_group_indices) - 1 - (available - index - 1));
            }
            else {
                double* weights = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                size_t weights_size = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size;

                group_pos = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    weights,
                    weights_size,
                    group_start_offset,
                    static_cast<size_t>(it - timestamp_group_indices),
                    rand_state
                );
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return Edge{-1, -1, -1};

        if (random_pickers::is_index_based_picker(picker_type)) {
            const auto index = random_pickers::pick_using_index_based_picker_device(
                picker_type, 0, static_cast<int>(num_groups), !forward, rand_state);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_pos = forward
                ? static_cast<long>(group_start_offset + index)
                : static_cast<long>(group_end_offset - 1 - (num_groups - index - 1));
        }
        else {
            if (forward) {
                group_pos = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential,
                    graph->node_edge_index->outbound_forward_cumulative_weights_exponential_size,
                    group_start_offset,
                    group_end_offset,
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
            else {
                double* weights = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential;

                size_t weights_size = graph->is_directed
                    ? graph->node_edge_index->inbound_backward_cumulative_weights_exponential_size
                    : graph->node_edge_index->outbound_backward_cumulative_weights_exponential_size;

                group_pos = random_pickers::pick_using_weight_based_picker_device(
                    picker_type,
                    weights,
                    weights_size,
                    group_start_offset,
                    group_end_offset,
                    rand_state);
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get edge range for selected group
    const size_t edge_start = timestamp_group_indices[group_pos];
    size_t edge_end;

    if (group_pos + 1 < group_end_offset) {
        edge_end = timestamp_group_indices[group_pos + 1];
    } else {
        if (forward) {
            edge_end = graph->node_edge_index->outbound_offsets[node_id + 1];
        } else {
            edge_end = graph->is_directed
                ? graph->node_edge_index->inbound_offsets[node_id + 1]
                : graph->node_edge_index->outbound_offsets[node_id + 1];
        }
    }

    // Validate range before random selection
    size_t edge_indices_size = forward
        ? graph->node_edge_index->outbound_indices_size
        : (graph->is_directed ? graph->node_edge_index->inbound_indices_size : graph->node_edge_index->outbound_indices_size);

    if (edge_start >= edge_end || edge_start >= edge_indices_size || edge_end > edge_indices_size) {
        return Edge{-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by_device(static_cast<int>(edge_end - edge_start), rand_state)];

    return Edge {
        graph->edge_data->sources[edge_idx],
        graph->edge_data->targets[edge_idx],
        graph->edge_data->timestamps[edge_idx]
    };
}

HOST TemporalGraphStore* temporal_graph::to_device_ptr(const TemporalGraphStore* graph) {
    // Create a new TemporalGraph object on the device
    TemporalGraphStore* device_graph;
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

HOST void temporal_graph::free_device_pointers(TemporalGraphStore* d_graph) {
    if (!d_graph) return;

    // Copy the struct from device to host to access pointers
    TemporalGraphStore h_graph;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&h_graph, d_graph, sizeof(TemporalGraphStore), cudaMemcpyDeviceToHost));

    // Free only the nested device pointers (not their underlying data)
    if (h_graph.edge_data) clear_memory(&h_graph.edge_data, true);
    if (h_graph.node_edge_index) clear_memory(&h_graph.node_edge_index, true);

    clear_memory(&d_graph, true);
}

#endif
