#ifndef NODE_EDGE_INDEX_H
#define NODE_EDGE_INDEX_H

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "../data/temporal_graph_data.cuh"
#include "../graph/node_edge_index.cuh"
#include "TemporalRandomWalk.cuh"

class NodeEdgeIndex {
    std::unique_ptr<TemporalRandomWalk> self_owned_;

public:
    TemporalGraphData* node_edge_index;

    explicit NodeEdgeIndex(bool use_gpu);

    explicit NodeEdgeIndex(TemporalGraphData* shared) : node_edge_index(shared) {}

    ~NodeEdgeIndex();

    NodeEdgeIndex(const NodeEdgeIndex&) = delete;
    NodeEdgeIndex& operator=(const NodeEdgeIndex&) = delete;
    NodeEdgeIndex(NodeEdgeIndex&&) noexcept = default;
    NodeEdgeIndex& operator=(NodeEdgeIndex&&) noexcept = default;

    // --- Read-back accessors (return host vectors) ---
    std::vector<size_t> node_group_outbound_offsets() const         { return node_edge_index->node_group_outbound_offsets.to_host_vector(); }
    std::vector<size_t> node_group_inbound_offsets() const          { return node_edge_index->node_group_inbound_offsets.to_host_vector(); }
    std::vector<size_t> node_ts_sorted_outbound_indices() const     { return node_edge_index->node_ts_sorted_outbound_indices.to_host_vector(); }
    std::vector<size_t> node_ts_sorted_inbound_indices() const      { return node_edge_index->node_ts_sorted_inbound_indices.to_host_vector(); }
    std::vector<size_t> count_ts_group_per_node_outbound() const    { return node_edge_index->count_ts_group_per_node_outbound.to_host_vector(); }
    std::vector<size_t> count_ts_group_per_node_inbound() const     { return node_edge_index->count_ts_group_per_node_inbound.to_host_vector(); }
    std::vector<size_t> node_ts_group_outbound_offsets() const      { return node_edge_index->node_ts_group_outbound_offsets.to_host_vector(); }
    std::vector<size_t> node_ts_group_inbound_offsets() const       { return node_edge_index->node_ts_group_inbound_offsets.to_host_vector(); }
    std::vector<double> outbound_forward_cumulative_weights_exponential() const {
        return node_edge_index->outbound_forward_cumulative_weights_exponential.to_host_vector();
    }
    std::vector<double> outbound_backward_cumulative_weights_exponential() const {
        return node_edge_index->outbound_backward_cumulative_weights_exponential.to_host_vector();
    }
    std::vector<double> inbound_backward_cumulative_weights_exponential() const {
        return node_edge_index->inbound_backward_cumulative_weights_exponential.to_host_vector();
    }

    void clear() const { node_edge_index::clear(*node_edge_index); }

    void rebuild(TemporalGraphData* shared_data, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(TemporalGraphData* shared_data, double timescale_bound) const;

    [[nodiscard]] size_t get_memory_used() const {
        return node_edge_index::get_memory_used(*node_edge_index);
    }
};

#endif // NODE_EDGE_INDEX_H
