#include "temporal_graph.cuh"

#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/merge.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "../common/cuda_sort.cuh"
#endif

#include "../common/nvtx_utils.h"
#include "../common/comparators.cuh"
#include "../common/parallel_algorithms.cuh"
#include "../common/cuda_config.cuh"

#include <vector>
#include <cstring>
#include <algorithm>
#include <omp.h>

HOST void temporal_graph::update_temporal_weights(TemporalGraphData& data) {
#ifdef HAS_CUDA
    if (data.use_gpu) {
        edge_data::update_temporal_weights_cuda(data, data.timescale_bound);
        node_edge_index::update_temporal_weights_cuda(data, data.timescale_bound);
    } else
#endif
    {
        edge_data::update_temporal_weights_std(data, data.timescale_bound);
        node_edge_index::update_temporal_weights_std(data, data.timescale_bound);
    }
}

HOST DEVICE size_t temporal_graph::get_total_edges(const TemporalGraphData& data) {
    return edge_data::size(data);
}

HOST size_t temporal_graph::get_node_count(const TemporalGraphData& data) {
    return edge_data::active_node_count(data);
}

HOST int64_t temporal_graph::get_latest_timestamp(const TemporalGraphData& data) {
    return data.latest_timestamp;
}

HOST std::vector<int> temporal_graph::get_node_ids(const TemporalGraphData& data) {
    return edge_data::get_active_node_ids(data);
}

HOST std::vector<Edge> temporal_graph::get_edges(const TemporalGraphData& data) {
    return edge_data::get_edges(data);
}

/**
 * Std implementations
 */

HOST void temporal_graph::sort_and_merge_edges_std(
    TemporalGraphData& data, const size_t start_idx) {
    const size_t total_size = edge_data::size(data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* sources = data.sources.data();
    int* targets = data.targets.data();
    int64_t* timestamps = data.timestamps.data();

    // === Step 1: Build index arrays for old and new edges ===
    std::vector<int> old_indices(start_idx);
    std::vector<int> new_indices(new_edges_count);
    std::vector<int> merged_indices(total_size);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(start_idx); ++i)
        old_indices[i] = i;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(new_edges_count); ++i)
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
    std::vector<int> merged_sources(total_size);
    std::vector<int> merged_targets(total_size);
    std::vector<int64_t> merged_timestamps(total_size);

    float* edge_features = data.edge_features.data();
    const size_t feature_dim = data.feature_dim;

    std::vector<float> merged_edge_features;
    if (feature_dim > 0) {
        merged_edge_features.resize(total_size * feature_dim);
    }

    // === Step 5: Gather edge data using merged indices ===
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_size); ++i) {
        const int idx = merged_indices[i];
        merged_sources[i] = sources[idx];
        merged_targets[i] = targets[idx];
        merged_timestamps[i] = timestamps[idx];

        if (feature_dim > 0) {
            std::memcpy(
                merged_edge_features.data() + (static_cast<size_t>(i) * feature_dim),
                edge_features + (static_cast<size_t>(idx) * feature_dim),
                feature_dim * sizeof(float));
        }
    }

    // === Step 6: Copy merged data back to graph ===
    std::memcpy(sources, merged_sources.data(), total_size * sizeof(int));
    std::memcpy(targets, merged_targets.data(), total_size * sizeof(int));
    std::memcpy(timestamps, merged_timestamps.data(), total_size * sizeof(int64_t));

    if (feature_dim > 0) {
        std::memcpy(edge_features, merged_edge_features.data(),
                    total_size * feature_dim * sizeof(float));
    }
}

HOST void temporal_graph::delete_old_edges_std(TemporalGraphData& data) {
    if (data.max_time_capacity <= 0 || edge_data::empty(data)) return;

    const int64_t cutoff_time = data.latest_timestamp - data.max_time_capacity;
    const int64_t* ts_begin = data.timestamps.data();
    const int64_t* ts_end = ts_begin + data.timestamps.size();

    const auto it = std::upper_bound(ts_begin, ts_end, cutoff_time);
    if (it == ts_begin) return;

    const size_t delete_count = static_cast<size_t>(it - ts_begin);

    data.sources.drop_front(delete_count);
    data.targets.drop_front(delete_count);
    data.timestamps.drop_front(delete_count);

    if (data.feature_dim > 0) {
        const size_t feature_values_to_delete = delete_count * data.feature_dim;
        data.edge_features.drop_front(feature_values_to_delete);
    }
}

HOST void temporal_graph::add_multiple_edges_std(
    TemporalGraphData& data,
    const int* sources,
    const int* targets,
    const int64_t* timestamps,
    const size_t num_new_edges,
    const float* edge_features,
    const size_t feature_dim) {

    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(data);

    // Find maximum timestamp in the new edges
    int64_t max_timestamp = data.latest_timestamp;

    #pragma omp parallel for reduction(max:max_timestamp)
    for (size_t i = 0; i < num_new_edges; i++) {
        max_timestamp = std::max(max_timestamp, timestamps[i]);
    }

    data.latest_timestamp = max_timestamp;

    // Add edges to edge data
    edge_data::add_edges(data, sources, targets, timestamps, num_new_edges,
                         edge_features, feature_dim);

    // Sort and merge new edges
    sort_and_merge_edges_std(data, start_idx);

    // Handle time window
    if (data.max_time_capacity > 0) {
        delete_old_edges_std(data);
    }

    // Populate active node ids
    edge_data::populate_active_nodes_std(data);

    // Update timestamp groups
    edge_data::update_timestamp_groups_std(data);

    if (data.enable_temporal_node2vec) {
        edge_data::build_node_adjacency_csr_std(data);
    }

    // Rebuild edge indices
    node_edge_index::rebuild(data);

    // Update temporal weights if enabled
    if (data.enable_weight_computation) {
        update_temporal_weights(data);
    }
}

HOST size_t temporal_graph::count_timestamps_less_than_std(
    const TemporalGraphData& data, const int64_t timestamp) {
    if (edge_data::empty(data)) return 0;

    const int64_t* begin = data.unique_timestamps.data();
    const int64_t* end = begin + data.unique_timestamps.size();

    const auto it = std::lower_bound(begin, end, timestamp);
    return it - begin;
}

HOST size_t temporal_graph::count_timestamps_greater_than_std(
    const TemporalGraphData& data, const int64_t timestamp) {
    if (edge_data::empty(data)) return 0;

    const int64_t* begin = data.unique_timestamps.data();
    const int64_t* end = begin + data.unique_timestamps.size();

    const auto it = std::upper_bound(begin, end, timestamp);
    return end - it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_std(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
    const size_t* timestamp_group_offsets;
    const size_t* timestamp_group_indices;
    const size_t* edge_indices;

    if (!edge_data::is_node_active(data, node_id)) {
        return 0;
    }

    if (data.is_directed) {
        timestamp_group_offsets = data.count_ts_group_per_node_inbound.data();
        timestamp_group_indices = data.node_ts_group_inbound_offsets.data();
        edge_indices = data.node_ts_sorted_inbound_indices.data();
    } else {
        timestamp_group_offsets = data.count_ts_group_per_node_outbound.data();
        timestamp_group_indices = data.node_ts_group_outbound_offsets.data();
        edge_indices = data.node_ts_sorted_outbound_indices.data();
    }

    const size_t group_start = timestamp_group_offsets[node_id];
    const size_t group_end = timestamp_group_offsets[node_id + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps = data.timestamps.data();

    const auto it = std::lower_bound(
        timestamp_group_indices + static_cast<int>(group_start),
        timestamp_group_indices + static_cast<int>(group_end),
        timestamp,
        [timestamps, edge_indices](const size_t group_pos, const int64_t ts) {
            return timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_std(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active(data, node_id)) {
        return 0;
    }

    // Used for forward walks
    const size_t* timestamp_group_offsets =
        data.count_ts_group_per_node_outbound.data();
    const size_t* timestamp_group_indices =
        data.node_ts_group_outbound_offsets.data();
    const size_t* edge_indices =
        data.node_ts_sorted_outbound_indices.data();

    const size_t group_start = timestamp_group_offsets[node_id];
    const size_t group_end = timestamp_group_offsets[node_id + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps = data.timestamps.data();

    const auto it = std::upper_bound(
        timestamp_group_indices + static_cast<int>(group_start),
        timestamp_group_indices + static_cast<int>(group_end),
        timestamp,
        [timestamps, edge_indices](const int64_t ts, const size_t group_pos) {
            return ts < timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, timestamp_group_indices + static_cast<int>(group_end));
}

/**
 * CUDA implementations
 */

#ifdef HAS_CUDA

HOST void temporal_graph::sort_and_merge_edges_cuda(
    TemporalGraphData& data, const size_t start_idx) {
    NvtxRange r("ingestion_sort_merge");

    const size_t total_size = edge_data::size(data);
    const size_t new_edges_count = total_size - start_idx;
    if (new_edges_count == 0) return;

    int* d_sources = data.sources.data();
    int* d_targets = data.targets.data();
    int64_t* d_timestamps = data.timestamps.data();

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

    // === Step 3: Merge ===
    thrust::device_vector<int> merged_indices(static_cast<int>(total_size));

    thrust::merge(
        old_indices.begin(), old_indices.end(),
        new_indices.begin(), new_indices.end(),
        merged_indices.begin(),
        TimestampComparator(d_timestamps)
    );

    // === Step 4: Allocate temporary merged arrays (RAII via Buffer) ===
    Buffer<int> d_merged_sources(true);
    d_merged_sources.resize(total_size);
    Buffer<int> d_merged_targets(true);
    d_merged_targets.resize(total_size);
    Buffer<int64_t> d_merged_timestamps(true);
    d_merged_timestamps.resize(total_size);

    // === Step 5: Gather merged edge data ===
    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_sources),
        thrust::device_pointer_cast(d_merged_sources.data())
    );

    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_targets),
        thrust::device_pointer_cast(d_merged_targets.data())
    );

    thrust::gather(
        merged_indices.begin(), merged_indices.end(),
        thrust::device_pointer_cast(d_timestamps),
        thrust::device_pointer_cast(d_merged_timestamps.data())
    );

    CUDA_KERNEL_CHECK("After thrust gather");

    // === Step 6: Copy back to graph ===
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_sources, d_merged_sources.data(),
        total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_targets, d_merged_targets.data(),
        total_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamps, d_merged_timestamps.data(),
        total_size * sizeof(int64_t), cudaMemcpyDeviceToDevice));

    if (data.feature_dim > 0) {
        std::vector<int> host_merged_indices(total_size);
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            host_merged_indices.data(),
            thrust::raw_pointer_cast(merged_indices.data()),
            total_size * sizeof(int),
            cudaMemcpyDeviceToHost));

        const size_t feature_dim = data.feature_dim;
        float* edge_features = data.edge_features.data();
        std::vector<float> merged_edge_features(total_size * feature_dim);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_size); ++i) {
            const int idx = host_merged_indices[i];
            std::memcpy(
                merged_edge_features.data() + (static_cast<size_t>(i) * feature_dim),
                edge_features + (static_cast<size_t>(idx) * feature_dim),
                feature_dim * sizeof(float));
        }

        std::memcpy(
            edge_features,
            merged_edge_features.data(),
            total_size * feature_dim * sizeof(float));
    }
    // d_merged_* RAII-free on scope exit.
}

HOST void temporal_graph::delete_old_edges_cuda(TemporalGraphData& data) {
    if (data.max_time_capacity <= 0 || edge_data::empty(data)) return;

    const int64_t cutoff_time = data.latest_timestamp - data.max_time_capacity;

    int64_t* timestamps_ptr = data.timestamps.data();
    const size_t timestamps_size = data.timestamps.size();

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamps_ptr),
        thrust::device_pointer_cast(timestamps_ptr + timestamps_size),
        cutoff_time
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in delete_old_edges_cuda");

    if (it == thrust::device_pointer_cast(timestamps_ptr)) return;

    const size_t delete_count = static_cast<size_t>(
        it - thrust::device_pointer_cast(timestamps_ptr));

    data.sources.drop_front(delete_count);
    data.targets.drop_front(delete_count);
    data.timestamps.drop_front(delete_count);

    if (data.feature_dim > 0) {
        const size_t feature_values_to_delete = delete_count * data.feature_dim;
        data.edge_features.drop_front(feature_values_to_delete);
    }
}

HOST void temporal_graph::add_multiple_edges_cuda(
    TemporalGraphData& data,
    const int* sources,
    const int* targets,
    const int64_t* timestamps,
    const size_t num_new_edges,
    const float* edge_features,
    const size_t feature_dim) {
    if (num_new_edges == 0) return;

    // Get start index for new edges
    const size_t start_idx = edge_data::size(data);

    // Find maximum timestamp on host (timestamps is a host pointer here).
    int64_t max_timestamp = data.latest_timestamp;
    #pragma omp parallel for reduction(max:max_timestamp)
    for (size_t i = 0; i < num_new_edges; i++) {
        max_timestamp = std::max(max_timestamp, timestamps[i]);
    }
    data.latest_timestamp = max_timestamp;

    // Add edges. The new edge_data::add_edges takes host pointers and uses
    // Buffer::append_from_host internally to do the H->D copy.
    edge_data::add_edges(data, sources, targets, timestamps, num_new_edges,
                         edge_features, feature_dim);

    // Sort and merge new edges
    sort_and_merge_edges_cuda(data, start_idx);

    // Handle time window
    if (data.max_time_capacity > 0) {
        delete_old_edges_cuda(data);
    }

    // Populate active node ids
    edge_data::populate_active_nodes_cuda(data);

    // Update timestamp groups
    edge_data::update_timestamp_groups_cuda(data);

    if (data.enable_temporal_node2vec) {
        edge_data::build_node_adjacency_csr_cuda(data);
    }

    // Rebuild edge indices
    node_edge_index::rebuild(data);

    // Update temporal weights if enabled
    if (data.enable_weight_computation) {
        update_temporal_weights(data);
    }
}

HOST size_t temporal_graph::count_timestamps_less_than_cuda(
    const TemporalGraphData& data, const int64_t timestamp) {
    if (edge_data::empty(data)) return 0;

    const int64_t* unique_timestamps = data.unique_timestamps.data();
    const size_t n = data.unique_timestamps.size();

    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(unique_timestamps),
        thrust::device_pointer_cast(unique_timestamps + n),
        timestamp
    );
    CUDA_KERNEL_CHECK("After thrust lower_bound in count_timestamps_less_than_cuda");

    return it - thrust::device_pointer_cast(unique_timestamps);
}

HOST size_t temporal_graph::count_timestamps_greater_than_cuda(
    const TemporalGraphData& data, const int64_t timestamp) {
    if (edge_data::empty(data)) return 0;

    const int64_t* unique_timestamps = data.unique_timestamps.data();
    const size_t n = data.unique_timestamps.size();

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(unique_timestamps),
        thrust::device_pointer_cast(unique_timestamps + n),
        timestamp
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_timestamps_greater_than_cuda");

    return thrust::device_pointer_cast(unique_timestamps + n) - it;
}

HOST size_t temporal_graph::count_node_timestamps_less_than_cuda(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active(data, node_id)) {
        return 0;
    }

    const size_t* timestamp_group_offsets;
    const size_t* timestamp_group_indices;
    const size_t* edge_indices;

    if (data.is_directed) {
        timestamp_group_offsets = data.count_ts_group_per_node_inbound.data();
        timestamp_group_indices = data.node_ts_group_inbound_offsets.data();
        edge_indices = data.node_ts_sorted_inbound_indices.data();
    } else {
        timestamp_group_offsets = data.count_ts_group_per_node_outbound.data();
        timestamp_group_indices = data.node_ts_group_outbound_offsets.data();
        edge_indices = data.node_ts_sorted_outbound_indices.data();
    }

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start,
        timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end,
        timestamp_group_offsets + node_id + 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = data.timestamps.data();

    const auto it = thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const size_t group_pos, const int64_t ts) {
            return timestamps_ptr[edge_indices[group_pos]] < ts;
        });
    CUDA_KERNEL_CHECK("After thrust lower_bound in count_node_timestamps_less_than_cuda");

    return thrust::distance(
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start), it);
}

HOST size_t temporal_graph::count_node_timestamps_greater_than_cuda(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
    if (!edge_data::is_node_active(data, node_id)) {
        return 0;
    }

    const size_t* timestamp_group_offsets =
        data.count_ts_group_per_node_outbound.data();
    const size_t* timestamp_group_indices =
        data.node_ts_group_outbound_offsets.data();
    const size_t* edge_indices =
        data.node_ts_sorted_outbound_indices.data();

    // Copy offsets from device to host
    size_t group_start, group_end;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_start,
        timestamp_group_offsets + node_id, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&group_end,
        timestamp_group_offsets + (node_id + 1), sizeof(size_t), cudaMemcpyDeviceToHost));
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = data.timestamps.data();

    const auto it = thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_start),
        thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices] HOST DEVICE (const int64_t ts, const size_t group_pos) {
            return ts < timestamps_ptr[edge_indices[group_pos]];
        });
    CUDA_KERNEL_CHECK("After thrust upper_bound in count_node_timestamps_greater_than_cuda");

    return thrust::distance(
        it, thrust::device_pointer_cast(timestamp_group_indices) + static_cast<int>(group_end));
}

#endif

HOST size_t temporal_graph::get_memory_used(const TemporalGraphData& data) {
    return edge_data::get_memory_used(data) + node_edge_index::get_memory_used(data);
}
