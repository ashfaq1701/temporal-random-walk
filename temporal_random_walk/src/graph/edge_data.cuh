#ifndef EDGE_DATA_CUH
#define EDGE_DATA_CUH

#include <cstddef>
#include <vector>
#include <algorithm>

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#endif

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"
#include "../common/cuda_config.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../data/buffer.cuh"

/*
 * STAGING FILE for task 5a.
 *
 * This file is NOT in the CMake build. The swap (delete old
 * edge_data.{cu,cuh}, rename edge_data_new.{cu,cuh} into their
 * places) happens in task 5g alongside the other parallel files.
 */

namespace edge_data {

    /**
     * Common
     */
    HOST DEVICE size_t size(const TemporalGraphData& data);
    HOST void set_size(TemporalGraphData& data, size_t size);
    HOST bool empty(const TemporalGraphData& data);

    HOST void add_edges(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges);

    HOST void add_edges(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features,
        size_t feature_dim);

    // Convenience: append a single edge. Thin wrapper around add_edges.
    HOST inline void push_back(TemporalGraphData& data,
                               const int src, const int tgt, const int64_t ts) {
        const int srcs[1]     = {src};
        const int tgts[1]     = {tgt};
        const int64_t times[1] = {ts};
        add_edges(data, srcs, tgts, times, 1);
    }

    HOST std::vector<Edge> get_edges(const TemporalGraphData& data);

    HOST std::vector<int> get_active_node_ids(const TemporalGraphData& data);

    HOST size_t active_node_count(const TemporalGraphData& data);

    HOST DEVICE inline int get_max_node_id(const TemporalGraphData& data) {
        return data.max_node_id;
    }

    HOST DEVICE inline SizeRange get_timestamp_group_range(
        const TemporalGraphData& data, const size_t group_idx) {
        if (group_idx >= data.unique_timestamps.size()) {
            return SizeRange{0, 0};
        }
        return SizeRange{
            data.timestamp_group_offsets.data()[group_idx],
            data.timestamp_group_offsets.data()[group_idx + 1]};
    }

    HOST DEVICE inline size_t get_timestamp_group_count(const TemporalGraphData& data) {
        return data.unique_timestamps.size();
    }

    HOST inline size_t find_group_after_timestamp(
        const TemporalGraphData& data, const int64_t timestamp) {
        if (data.unique_timestamps.size() == 0) return 0;

        const int64_t* begin = data.unique_timestamps.data();
        const int64_t* end = begin + data.unique_timestamps.size();

        const auto it = std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    HOST inline size_t find_group_before_timestamp(
        const TemporalGraphData& data, const int64_t timestamp) {
        if (data.unique_timestamps.size() == 0) return 0;

        const int64_t* begin = data.unique_timestamps.data();
        const int64_t* end = begin + data.unique_timestamps.size();

        const auto it = std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    HOST bool is_node_active(const TemporalGraphData& data, int node_id);

    /**
     * Active nodes + CSR (std & cuda)
     */
    HOST void populate_active_nodes_std(TemporalGraphData& data);

    HOST void build_node_adjacency_csr_std(TemporalGraphData& data);

    HOST void update_timestamp_groups_std(TemporalGraphData& data);

    HOST void update_temporal_weights_std(
        TemporalGraphData& data, double timescale_bound);

#ifdef HAS_CUDA
    HOST void populate_active_nodes_cuda(TemporalGraphData& data);

    HOST void build_node_adjacency_csr_cuda(TemporalGraphData& data);

    HOST void update_timestamp_groups_cuda(TemporalGraphData& data);

    HOST void update_temporal_weights_cuda(
        TemporalGraphData& data, double timescale_bound);

    /**
     * Device queries — take a TemporalGraphView (POD by value) instead of
     * a TemporalGraphData pointer. Inline so kernels can call them directly.
     */
    DEVICE inline size_t find_group_after_timestamp_device(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;

        const int64_t* begin = view.unique_timestamps;
        const int64_t* end = begin + view.num_groups;

        const auto it = cuda::std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    DEVICE inline size_t find_group_before_timestamp_device(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;

        const int64_t* begin = view.unique_timestamps;
        const int64_t* end = begin + view.num_groups;

        const auto it = cuda::std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    DEVICE inline bool is_node_active_device(
        const TemporalGraphView& view, const int node_id) {
        if (node_id < 0 || node_id >= view.active_node_ids_size) {
            return false;
        }
        return view.active_node_ids[node_id] == 1;
    }
#endif

    HOST size_t get_memory_used(const TemporalGraphData& data);

    // ============================================================
    // Test / debug helpers (not hot-path). One snapshot() call does
    // Buffer<T>::to_host_vector() over every edge-layer buffer so
    // test bodies stay readable. Prefer one snapshot per assertion
    // block rather than one per assertion (each call incurs D->H
    // copies when data.use_gpu).
    // ============================================================

    struct EdgeDataSnapshot {
        std::vector<int>     sources;
        std::vector<int>     targets;
        std::vector<int64_t> timestamps;
        std::vector<size_t>  timestamp_group_offsets;
        std::vector<int64_t> unique_timestamps;
        std::vector<double>  forward_cumulative_weights_exponential;
        std::vector<double>  backward_cumulative_weights_exponential;
        std::vector<int>     active_node_ids;
        std::vector<size_t>  node_adj_offsets;
        std::vector<int>     node_adj_neighbors;
    };

    HOST inline EdgeDataSnapshot snapshot(const TemporalGraphData& data) {
        return EdgeDataSnapshot{
            data.sources.to_host_vector(),
            data.targets.to_host_vector(),
            data.timestamps.to_host_vector(),
            data.timestamp_group_offsets.to_host_vector(),
            data.unique_timestamps.to_host_vector(),
            data.forward_cumulative_weights_exponential.to_host_vector(),
            data.backward_cumulative_weights_exponential.to_host_vector(),
            data.active_node_ids.to_host_vector(),
            data.node_adj_offsets.to_host_vector(),
            data.node_adj_neighbors.to_host_vector(),
        };
    }

}

#endif // EDGE_DATA_CUH
