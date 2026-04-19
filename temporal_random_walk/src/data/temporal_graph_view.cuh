#ifndef DATA_TEMPORAL_GRAPH_VIEW_CUH
#define DATA_TEMPORAL_GRAPH_VIEW_CUH

#include <cstddef>
#include <cstdint>

#include "../common/macros.cuh"
#include "temporal_graph_data.cuh"

/**
 * TemporalGraphView — a plain-old-data view of a TemporalGraphData
 * suitable for passing to CUDA kernels by value. All pointers refer to
 * memory owned elsewhere (by the originating TemporalGraphData). The
 * view itself owns nothing.
 *
 * Used in later tasks (4+) to kill the to_device_ptr / free_device_pointers
 * mirror pattern: instead of replicating a tree of heap-allocated structs
 * onto the device, the kernel takes this POD by value.
 *
 * Do NOT add methods that allocate, free, or modify pointers. This struct
 * is intentionally a dumb bag of raw pointers.
 */
struct TemporalGraphView {
    bool    is_directed;
    int64_t latest_timestamp;
    double  inv_p;
    double  inv_q;

    const int*     sources;
    const int*     targets;
    const int64_t* timestamps;
    size_t         num_edges;

    const size_t*  timestamp_group_offsets;
    const int64_t* unique_timestamps;
    size_t         num_groups;

    const int*     active_node_ids;
    size_t         active_node_ids_size;
    int            max_node_id;

    const double*  forward_cumulative_weights_exponential;
    const double*  backward_cumulative_weights_exponential;

    const size_t*  node_adj_offsets;
    const int*     node_adj_neighbors;

    const size_t*  node_group_outbound_offsets;
    const size_t*  node_group_inbound_offsets;

    const size_t*  node_ts_sorted_outbound_indices;
    const size_t*  node_ts_sorted_inbound_indices;

    const size_t*  count_ts_group_per_node_outbound;
    const size_t*  count_ts_group_per_node_inbound;

    const size_t*  node_ts_group_outbound_offsets;
    const size_t*  node_ts_group_inbound_offsets;

    const double*  outbound_forward_cumulative_weights_exponential;
    const double*  outbound_backward_cumulative_weights_exponential;
    const double*  inbound_backward_cumulative_weights_exponential;
};

/**
 * Build a TemporalGraphView from a TemporalGraphData. Zero allocation.
 * The returned view aliases into data; it becomes invalid when data is
 * mutated or destroyed.
 *
 * This is a free function (not a method on TemporalGraphData) because it
 * is layer-crossing: TemporalGraphData lives at the data layer and is
 * unaware of the view concept.
 */
HOST inline TemporalGraphView make_temporal_graph_view(
    const TemporalGraphData& data) {
    TemporalGraphView v{};
    v.is_directed       = data.is_directed;
    v.latest_timestamp  = data.latest_timestamp;
    v.inv_p             = data.inv_p;
    v.inv_q             = data.inv_q;

    v.sources           = data.sources.data();
    v.targets           = data.targets.data();
    v.timestamps        = data.timestamps.data();
    v.num_edges         = data.timestamps.size();

    v.timestamp_group_offsets = data.timestamp_group_offsets.data();
    v.unique_timestamps       = data.unique_timestamps.data();
    v.num_groups              = data.unique_timestamps.size();

    v.active_node_ids      = data.active_node_ids.data();
    v.active_node_ids_size = data.active_node_ids.size();
    v.max_node_id          = data.max_node_id;

    v.forward_cumulative_weights_exponential =
        data.forward_cumulative_weights_exponential.data();
    v.backward_cumulative_weights_exponential =
        data.backward_cumulative_weights_exponential.data();

    v.node_adj_offsets   = data.node_adj_offsets.data();
    v.node_adj_neighbors = data.node_adj_neighbors.data();

    v.node_group_outbound_offsets = data.node_group_outbound_offsets.data();
    v.node_group_inbound_offsets  = data.node_group_inbound_offsets.data();

    v.node_ts_sorted_outbound_indices = data.node_ts_sorted_outbound_indices.data();
    v.node_ts_sorted_inbound_indices  = data.node_ts_sorted_inbound_indices.data();

    v.count_ts_group_per_node_outbound = data.count_ts_group_per_node_outbound.data();
    v.count_ts_group_per_node_inbound  = data.count_ts_group_per_node_inbound.data();

    v.node_ts_group_outbound_offsets = data.node_ts_group_outbound_offsets.data();
    v.node_ts_group_inbound_offsets  = data.node_ts_group_inbound_offsets.data();

    v.outbound_forward_cumulative_weights_exponential =
        data.outbound_forward_cumulative_weights_exponential.data();
    v.outbound_backward_cumulative_weights_exponential =
        data.outbound_backward_cumulative_weights_exponential.data();
    v.inbound_backward_cumulative_weights_exponential =
        data.inbound_backward_cumulative_weights_exponential.data();

    return v;
}

#endif // DATA_TEMPORAL_GRAPH_VIEW_CUH
