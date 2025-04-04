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
#include "node_mapping.cuh"

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
    return node_mapping::active_size(graph->node_mapping);
}

HOST int64_t temporal_graph::get_latest_timestamp(const TemporalGraphStore* graph) {
    return graph->latest_timestamp;
}

HOST DataBlock<int> temporal_graph::get_node_ids(const TemporalGraphStore* graph) {
    return node_mapping::get_active_node_ids(graph->node_mapping);
}

HOST DataBlock<Edge> temporal_graph::get_edges(const TemporalGraphStore* graph) {
    return edge_data::get_edges(graph->edge_data);
}

HOST void temporal_graph::add_multiple_edges_std(TemporalGraphStore* graph, const Edge* new_edges, const size_t num_new_edges) {
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

    // Update node mappings
    node_mapping::update_std(graph->node_mapping, graph->edge_data, start_idx, start_idx + num_new_edges);

    // Sort and merge new edges
    sort_and_merge_edges_std(graph, start_idx);

    // Update timestamp groups
    edge_data::update_timestamp_groups_std(graph->edge_data);

    // Handle time window
    if (graph->max_time_capacity > 0) {
        delete_old_edges_std(graph);
    }

    // Rebuild edge indices
    node_edge_index::rebuild(graph->node_edge_index, graph->edge_data, graph->node_mapping, graph->is_directed);

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
    if (start_idx >= edge_data::size(graph->edge_data)) return;

    const size_t total_size = edge_data::size(graph->edge_data);
    const size_t new_edges_count = total_size - start_idx;

    // Sort new edges first
    auto* indices = new size_t[new_edges_count];
    for (size_t i = 0; i < new_edges_count; i++) {
        indices[i] = start_idx + i;
    }

    // Sort indices based on timestamps
    std::sort(indices, indices + new_edges_count,
        [graph](const size_t i, const size_t j) {
            return graph->edge_data->timestamps[i] < graph->edge_data->timestamps[j];
    });

    // Create temporary arrays for sorted edges
    auto sorted_sources = new int[new_edges_count];
    auto sorted_targets = new int[new_edges_count];
    auto* sorted_timestamps = new int64_t[new_edges_count];

    // Apply permutation
    for (size_t i = 0; i < new_edges_count; i++) {
        const size_t idx = indices[i];
        sorted_sources[i] = graph->edge_data->sources[idx];
        sorted_targets[i] = graph->edge_data->targets[idx];
        sorted_timestamps[i] = graph->edge_data->timestamps[idx];
    }

    // Copy back sorted edges
    for (size_t i = 0; i < new_edges_count; i++) {
        graph->edge_data->sources[start_idx + i] = sorted_sources[i];
        graph->edge_data->targets[start_idx + i] = sorted_targets[i];
        graph->edge_data->timestamps[start_idx + i] = sorted_timestamps[i];
    }

    // If we have existing edges, merge them with the new sorted edges
    if (start_idx > 0) {
        // Create buffers for the merge result
        auto merged_sources = new int[total_size];
        auto merged_targets = new int[total_size];
        auto* merged_timestamps = new int64_t[total_size];

        size_t i = 0;      // Index for existing edges
        size_t j = start_idx;  // Index for new edges
        size_t k = 0;      // Index for merged result

        // Merge while keeping arrays aligned
        while (i < start_idx && j < total_size) {
            if (graph->edge_data->timestamps[i] <= graph->edge_data->timestamps[j]) {
                merged_sources[k] = graph->edge_data->sources[i];
                merged_targets[k] = graph->edge_data->targets[i];
                merged_timestamps[k] = graph->edge_data->timestamps[i];
                i++;
            } else {
                merged_sources[k] = graph->edge_data->sources[j];
                merged_targets[k] = graph->edge_data->targets[j];
                merged_timestamps[k] = graph->edge_data->timestamps[j];
                j++;
            }
            k++;
        }

        // Copy remaining entries
        while (i < start_idx) {
            merged_sources[k] = graph->edge_data->sources[i];
            merged_targets[k] = graph->edge_data->targets[i];
            merged_timestamps[k] = graph->edge_data->timestamps[i];
            i++;
            k++;
        }

        while (j < total_size) {
            merged_sources[k] = graph->edge_data->sources[j];
            merged_targets[k] = graph->edge_data->targets[j];
            merged_timestamps[k] = graph->edge_data->timestamps[j];
            j++;
            k++;
        }

        // Copy merged data back to edge_data
        for (size_t idx = 0; idx < total_size; idx++) {
            graph->edge_data->sources[idx] = merged_sources[idx];
            graph->edge_data->targets[idx] = merged_targets[idx];
            graph->edge_data->timestamps[idx] = merged_timestamps[idx];
        }

        // Clean up
        delete[] merged_sources;
        delete[] merged_targets;
        delete[] merged_timestamps;
    }

    // Clean up
    delete[] indices;
    delete[] sorted_sources;
    delete[] sorted_targets;
    delete[] sorted_timestamps;
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

    // Track which nodes still have edges
    bool* has_edges = new bool[graph->node_mapping->capacity];
    fill_memory(has_edges, graph->node_mapping->capacity, false, false);

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

        // Mark nodes that still have edges
        for (size_t i = 0; i < remaining; i++) {
            has_edges[node_mapping::to_dense(graph->node_mapping, graph->edge_data->sources[i])] = true;
            has_edges[node_mapping::to_dense(graph->node_mapping, graph->edge_data->targets[i])] = true;
        }
    }

    edge_data::set_size(graph->edge_data, remaining);

    // Mark nodes with no edges as deleted
    for (size_t i = 0; i < graph->node_mapping->capacity; i++) {
        if (!has_edges[i]) {
            node_mapping::mark_node_deleted(graph->node_mapping, graph->node_mapping->node_index[i]);
        }
    }

    delete[] has_edges;

    // Update all data structures after edge deletion
    edge_data::update_timestamp_groups_std(graph->edge_data);
    node_mapping::update_std(graph->node_mapping, graph->edge_data, 0, graph->edge_data->timestamps_size);
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
    // Used for backward walks
    const int dense_idx = node_mapping::to_dense(graph->node_mapping, node_id);
    if (dense_idx < 0) return 0;

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

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
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
    // Used for forward walks
    int dense_idx = node_mapping::to_dense(graph->node_mapping, node_id);
    if (dense_idx < 0) return 0;

    size_t* timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t* timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t* edge_indices = graph->node_edge_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
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
HOST void temporal_graph::add_multiple_edges_cuda(TemporalGraphStore* graph, const Edge* new_edges, const size_t num_new_edges) {
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

    // Update node mappings
    node_mapping::update_cuda(graph->node_mapping, graph->edge_data, start_idx, start_idx + num_new_edges);

    // Sort and merge new edges
    sort_and_merge_edges_cuda(graph, start_idx);

    // Update timestamp groups
    edge_data::update_timestamp_groups_cuda(graph->edge_data);

    // Handle time window
    if (graph->max_time_capacity > 0) {
        delete_old_edges_cuda(graph);
    }

    // Rebuild edge indices
    node_edge_index::rebuild(graph->node_edge_index, graph->edge_data, graph->node_mapping, graph->is_directed);

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
    if (start_idx >= edge_data::size(graph->edge_data)) return;

    const size_t total_size = graph->edge_data->timestamps_size;
    const size_t new_edges_count = total_size - start_idx;

    // Create index array
    size_t* indices = nullptr;
    allocate_memory(&indices, new_edges_count, graph->use_gpu);

    // Initialize indices with sequence starting at start_idx
    thrust::sequence(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(indices),
        thrust::device_pointer_cast(indices + new_edges_count),
        start_idx
    );
    CUDA_KERNEL_CHECK("After thrust sequence in sort_and_merge_edges_cuda");

    auto timestamps_ptr = graph->edge_data->timestamps;

    // Sort indices based on timestamps
    thrust::sort(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(indices),
        thrust::device_pointer_cast(indices + new_edges_count),
        [timestamps_ptr] DEVICE (const size_t i, const size_t j) {
            return timestamps_ptr[i] < timestamps_ptr[j];
        }
    );
    CUDA_KERNEL_CHECK("After thrust sort in sort_and_merge_edges_cuda");

    // Create temporary arrays for sorted data
    int* sorted_sources = nullptr;
    int* sorted_targets = nullptr;
    int64_t* sorted_timestamps = nullptr;
    allocate_memory(&sorted_sources, new_edges_count, graph->use_gpu);
    allocate_memory(&sorted_targets, new_edges_count, graph->use_gpu);
    allocate_memory(&sorted_timestamps, new_edges_count, graph->use_gpu);

    // Apply permutation using gather
    thrust::gather(
        thrust::device,
        thrust::device_pointer_cast(indices),
        thrust::device_pointer_cast(indices + new_edges_count),
        thrust::device_pointer_cast(graph->edge_data->sources),
        thrust::device_pointer_cast(sorted_sources));
    CUDA_KERNEL_CHECK("After thrust gather sources in sort_and_merge_edges_cuda");

    thrust::gather(
        thrust::device,
        thrust::device_pointer_cast(indices),
        thrust::device_pointer_cast(indices + new_edges_count),
        thrust::device_pointer_cast(graph->edge_data->targets),
        thrust::device_pointer_cast(sorted_targets)
    );
    CUDA_KERNEL_CHECK("After thrust gather targets in sort_and_merge_edges_cuda");

    thrust::gather(
        thrust::device,
        thrust::device_pointer_cast(indices),
        thrust::device_pointer_cast(indices + new_edges_count),
        thrust::device_pointer_cast(graph->edge_data->timestamps),
        thrust::device_pointer_cast(sorted_timestamps)
    );
    CUDA_KERNEL_CHECK("After thrust gather timestamps in sort_and_merge_edges_cuda");

    // Copy sorted data back
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(
        graph->edge_data->sources + start_idx,
        sorted_sources,
        new_edges_count * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(
        graph->edge_data->targets + start_idx,
        sorted_targets,
        new_edges_count * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(
        graph->edge_data->timestamps + start_idx,
        sorted_timestamps,
        new_edges_count * sizeof(int64_t),
        cudaMemcpyDeviceToDevice
    ));

    // Handle merging if we have existing edges
    if (start_idx > 0) {
        // Create merged arrays
        int* merged_sources = nullptr;
        int* merged_targets = nullptr;
        int64_t* merged_timestamps = nullptr;
        allocate_memory(&merged_sources, total_size, true);
        allocate_memory(&merged_targets, total_size, true);
        allocate_memory(&merged_timestamps, total_size, true);

        // Create zip iterators for merge operation
        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(graph->edge_data->sources),
            thrust::device_pointer_cast(graph->edge_data->targets),
            thrust::device_pointer_cast(graph->edge_data->timestamps)
        ));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(graph->edge_data->sources + start_idx),
            thrust::device_pointer_cast(graph->edge_data->targets + start_idx),
            thrust::device_pointer_cast(graph->edge_data->timestamps + start_idx)
        ));
        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(graph->edge_data->sources + start_idx),
            thrust::device_pointer_cast(graph->edge_data->targets + start_idx),
            thrust::device_pointer_cast(graph->edge_data->timestamps + start_idx)
        ));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(graph->edge_data->sources + total_size),
            thrust::device_pointer_cast(graph->edge_data->targets + total_size),
            thrust::device_pointer_cast(graph->edge_data->timestamps + total_size)
        ));
        auto result = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(merged_sources),
            thrust::device_pointer_cast(merged_targets),
            thrust::device_pointer_cast(merged_timestamps)
        ));

        // Merge based on timestamps
        thrust::merge(
            DEVICE_EXECUTION_POLICY,
            first1, last1,
            first2, last2,
            result,
            [] DEVICE (const thrust::tuple<int, int, int64_t>& a,
                                   const thrust::tuple<int, int, int64_t>& b) {
                return thrust::get<2>(a) <= thrust::get<2>(b);
            }
        );
        CUDA_KERNEL_CHECK("After thrust merge in sort_and_merge_edges_cuda");

        // Copy merged results back
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            graph->edge_data->sources,
            merged_sources,
            total_size * sizeof(int),
            cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            graph->edge_data->targets,
            merged_targets,
            total_size * sizeof(int),
            cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            graph->edge_data->timestamps,
            merged_timestamps,
            total_size * sizeof(int64_t),
            cudaMemcpyDeviceToDevice
        ));

        // Free merged arrays
        clear_memory(&merged_sources, true);
        clear_memory(&merged_targets, true);
        clear_memory(&merged_timestamps, true);
    }

    // Clean up temporary arrays
    clear_memory(&indices, true);
    clear_memory(&sorted_sources, true);
    clear_memory(&sorted_targets, true);
    clear_memory(&sorted_timestamps, true);
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

    // Create bool array for tracking nodes with edges
    bool* has_edges = nullptr;
    allocate_memory(&has_edges, graph->node_mapping->capacity, true);
    fill_memory(has_edges, graph->node_mapping->capacity, false, true);

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

        // Mark nodes with edges in parallel
        int* sources_ptr = graph->edge_data->sources;
        int* targets_ptr = graph->edge_data->targets;
        int* node_index = graph->node_mapping->node_index;
        int capacity = graph->node_mapping->capacity;

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(remaining),
            [sources_ptr, targets_ptr, has_edges, node_index, capacity] DEVICE (const size_t i) {
                has_edges[node_mapping::to_dense_from_ptr_device(node_index, sources_ptr[i], capacity)] = true;
                has_edges[node_mapping::to_dense_from_ptr_device(node_index, targets_ptr[i], capacity)] = true;
            }
        );
        CUDA_KERNEL_CHECK("After thrust for_each mark nodes in delete_old_edges_cuda");
    }

    // Update sizes
    edge_data::set_size(graph->edge_data, remaining);

    bool* is_deleted_ptr = graph->node_mapping->is_deleted;
    int* node_index_ptr = graph->node_mapping->node_index;
    const auto node_index_capacity = graph->node_mapping->capacity;

    // Mark deleted nodes in parallel
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(node_index_capacity),
        [has_edges, is_deleted_ptr, node_index_ptr, node_index_capacity] DEVICE (const size_t i) {
            if (!has_edges[i]) {
                node_mapping::mark_node_deleted_from_ptr(is_deleted_ptr, node_index_ptr, node_index_ptr[i], node_index_capacity);
            }
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each mark deleted in delete_old_edges_cuda");

    // Free temporary memory
    clear_memory(&has_edges, true);

    // Update data structures
    edge_data::update_timestamp_groups_cuda(graph->edge_data);
    node_mapping::update_cuda(graph->node_mapping, graph->edge_data, 0, graph->edge_data->timestamps_size);
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
    // Used for backward walks
    const int dense_idx = node_mapping::to_dense(graph->node_mapping, node_id);
    if (dense_idx < 0) return 0;

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
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start, timestamp_group_offsets + dense_idx, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end, timestamp_group_offsets + dense_idx + 1, sizeof(size_t), cudaMemcpyDeviceToHost));
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
    // Used for forward walks
    const int dense_idx = node_mapping::to_dense(graph->node_mapping, node_id);
    if (dense_idx < 0) return 0;

    const size_t* timestamp_group_offsets = graph->node_edge_index->outbound_timestamp_group_offsets;
    size_t* timestamp_group_indices = graph->node_edge_index->outbound_timestamp_group_indices;
    size_t* edge_indices = graph->node_edge_index->outbound_indices;

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start, timestamp_group_offsets + dense_idx, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end, timestamp_group_offsets + (dense_idx + 1), sizeof(size_t), cudaMemcpyDeviceToHost));
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

    const int dense_idx = node_mapping::to_dense(graph->node_mapping, node_id);
    if (dense_idx < 0) return Edge{-1, -1, -1};

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
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
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
            edge_end = graph->node_edge_index->outbound_offsets[dense_idx + 1];
        } else {
            edge_end = graph->is_directed
                ? graph->node_edge_index->inbound_offsets[dense_idx + 1]
                : graph->node_edge_index->outbound_offsets[dense_idx + 1];
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

    const int dense_idx = node_mapping::to_dense_device(graph->node_mapping, node_id);
    if (dense_idx < 0) return Edge{-1, -1, -1};

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
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
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
            edge_end = graph->node_edge_index->outbound_offsets[dense_idx + 1];
        } else {
            edge_end = graph->is_directed
                ? graph->node_edge_index->inbound_offsets[dense_idx + 1]
                : graph->node_edge_index->outbound_offsets[dense_idx + 1];
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

    if (graph->node_mapping) {
        temp_graph.node_mapping = node_mapping::to_device_ptr(graph->node_mapping);
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
    if (h_graph.node_mapping) clear_memory(&h_graph.node_mapping, true);

    clear_memory(&d_graph, true);
}

#endif
