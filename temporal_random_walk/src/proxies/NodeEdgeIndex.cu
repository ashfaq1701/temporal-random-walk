#include "NodeEdgeIndex.cuh"

#include "TemporalRandomWalk.cuh"

NodeEdgeIndex::NodeEdgeIndex(const bool use_gpu)
    : self_owned_(std::make_unique<TemporalRandomWalk>(
          /*is_directed=*/true, use_gpu)),
      node_edge_index(&self_owned_->impl()->data()) {}

NodeEdgeIndex::~NodeEdgeIndex() = default;

void NodeEdgeIndex::rebuild(TemporalGraphData* shared_data, const bool is_directed) const {
    shared_data->is_directed = is_directed;
    node_edge_index::rebuild(*shared_data);
}

std::pair<size_t, size_t> NodeEdgeIndex::get_edge_range(
    const int dense_node_id, const bool forward, const bool is_directed) const {
    const bool saved_is_directed = node_edge_index->is_directed;
    node_edge_index->is_directed = is_directed;
    const SizeRange r = node_edge_index::get_edge_range(
        *node_edge_index, dense_node_id, forward);
    node_edge_index->is_directed = saved_is_directed;
    return {r.from, r.to};
}

std::pair<size_t, size_t> NodeEdgeIndex::get_timestamp_group_range(
    const int dense_node_id, const size_t group_idx,
    const bool forward, const bool is_directed) const {
    const bool saved_is_directed = node_edge_index->is_directed;
    node_edge_index->is_directed = is_directed;
    const SizeRange r = node_edge_index::get_timestamp_group_range(
        *node_edge_index, dense_node_id, group_idx, forward);
    node_edge_index->is_directed = saved_is_directed;
    return {r.from, r.to};
}

size_t NodeEdgeIndex::get_timestamp_group_count(
    const int dense_node_id, const bool forward, const bool is_directed) const {
    const bool saved_is_directed = node_edge_index->is_directed;
    node_edge_index->is_directed = is_directed;
    const size_t c = node_edge_index::get_timestamp_group_count(
        *node_edge_index, dense_node_id, forward);
    node_edge_index->is_directed = saved_is_directed;
    return c;
}

void NodeEdgeIndex::update_temporal_weights(
    TemporalGraphData* shared_data, const double timescale_bound) const {
#ifdef HAS_CUDA
    if (shared_data->use_gpu) {
        node_edge_index::update_temporal_weights_cuda(*shared_data, timescale_bound);
        return;
    }
#endif
    node_edge_index::update_temporal_weights_std(*shared_data, timescale_bound);
}
