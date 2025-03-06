#include "INodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::clear() {
    // Clear edge CSR structures
    this->outbound_offsets.clear();
    this->outbound_indices.clear();
    this->outbound_timestamp_group_offsets.clear();
    this->outbound_timestamp_group_indices.clear();

    // Clear inbound structures
    this->inbound_offsets.clear();
    this->inbound_indices.clear();
    this->inbound_timestamp_group_offsets.clear();
    this->inbound_timestamp_group_indices.clear();
}

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::allocate_node_edge_offsets(size_t num_nodes, bool is_directed)
{
    // Initialize base CSR structures
    this->outbound_offsets.assign(num_nodes + 1, 0);
    this->outbound_timestamp_group_offsets.assign(num_nodes + 1, 0);

    if (is_directed) {
        this->inbound_offsets.assign(num_nodes + 1, 0);
        this->inbound_timestamp_group_offsets.assign(num_nodes + 1, 0);
    }
}

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::allocate_node_edge_indices(bool is_directed)
{
    this->outbound_indices.resize(this->outbound_offsets.back());
    if (is_directed) {
        this->inbound_indices.resize(this->inbound_offsets.back());
    }
}

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::allocate_node_timestamp_indices(bool is_directed)
{
    this->outbound_timestamp_group_indices.resize(this->outbound_timestamp_group_offsets.back());
    if (is_directed) {
        this->inbound_timestamp_group_indices.resize(this->inbound_timestamp_group_offsets.back());
    }
}

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::rebuild(
   const IEdgeData<GPUUsage>* edges,
   const INodeMapping<GPUUsage>* mapping,
   const bool is_directed) {

    const size_t num_nodes = mapping->size();
    const size_t num_edges = edges->size();

    IntVector dense_sources(num_edges);
    IntVector dense_targets(num_edges);
    populate_dense_ids(edges, mapping, dense_sources, dense_targets);

    this->allocate_node_edge_offsets(num_nodes, is_directed);
    compute_node_edge_offsets(edges, dense_sources, dense_targets, is_directed);

    this->allocate_node_edge_indices(is_directed);

    size_t outbound_edge_indices_len = is_directed ? num_edges : num_edges * 2;
    EdgeWithEndpointTypeVector outbound_edge_indices_buffer(outbound_edge_indices_len);

    compute_node_edge_indices(edges, dense_sources, dense_targets, outbound_edge_indices_buffer, is_directed);

    compute_node_timestamp_offsets(edges, num_nodes, is_directed);

    this->allocate_node_timestamp_indices(is_directed);

    compute_node_timestamp_indices(edges, num_nodes, is_directed);
}

template<GPUUsageMode GPUUsage>
HOST void INodeEdgeIndex<GPUUsage>::update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound) {
    const size_t num_nodes = this->outbound_offsets.size() - 1;

    this->outbound_forward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    this->outbound_backward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    if (!this->inbound_offsets.empty()) {
        this->inbound_backward_cumulative_weights_exponential.resize(this->inbound_timestamp_group_indices.size());
    }

    compute_temporal_weights(edges, timescale_bound, num_nodes);
}

template<GPUUsageMode GPUUsage>
HOST SizeRange INodeEdgeIndex<GPUUsage>::get_edge_range(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

   if (is_directed) {
       const auto& offsets = forward ? this->outbound_offsets : this->inbound_offsets;
       if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
           return SizeRange{0, 0};
       }
       return SizeRange{offsets[dense_node_id], offsets[dense_node_id + 1]};
   } else {
       if (dense_node_id < 0 || dense_node_id >= this->outbound_offsets.size() - 1) {
           return SizeRange{0, 0};
       }
       return SizeRange{this->outbound_offsets[dense_node_id], this->outbound_offsets[dense_node_id + 1]};
   }
}

template<GPUUsageMode GPUUsage>
HOST SizeRange INodeEdgeIndex<GPUUsage>::get_timestamp_group_range(
   int dense_node_id,
   size_t group_idx,
   bool forward,
   bool is_directed) const {

   const auto& group_offsets = (is_directed && !forward) ?
       this->inbound_timestamp_group_offsets : this->outbound_timestamp_group_offsets;
   const auto& group_indices = (is_directed && !forward) ?
       this->inbound_timestamp_group_indices : this->outbound_timestamp_group_indices;
   const auto& edge_offsets = (is_directed && !forward) ?
       this->inbound_offsets : this->outbound_offsets;

   if (dense_node_id < 0 || dense_node_id >= group_offsets.size() - 1) {
       return SizeRange{0, 0};
   }

   size_t num_groups = group_offsets[dense_node_id + 1] - group_offsets[dense_node_id];
   if (group_idx >= num_groups) {
       return SizeRange{0, 0};
   }

   size_t group_start_idx = group_offsets[dense_node_id] + group_idx;
   size_t group_start = group_indices[group_start_idx];

   // Group end is either next group's start or node's edge range end
   size_t group_end;
   if (group_idx == num_groups - 1) {
       group_end = edge_offsets[dense_node_id + 1];
   } else {
       group_end = group_indices[group_start_idx + 1];
   }

   return SizeRange{group_start, group_end};
}

template<GPUUsageMode GPUUsage>
HOST size_t INodeEdgeIndex<GPUUsage>::get_timestamp_group_count(
   int dense_node_id,
   bool forward,
   bool directed) const {

   const auto& offsets = get_timestamp_offset_vector(forward, directed);

   if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
       return 0;
   }

   return offsets[dense_node_id + 1] - offsets[dense_node_id];
}

template<GPUUsageMode GPUUsage>
[[nodiscard]] HOST typename INodeEdgeIndex<GPUUsage>::SizeVector INodeEdgeIndex<GPUUsage>::get_timestamp_offset_vector(
    const bool forward,
    const bool directed) const {
    return (directed && !forward) ? this->inbound_timestamp_group_offsets : this->outbound_timestamp_group_offsets;
}

template class INodeEdgeIndex<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class INodeEdgeIndex<GPUUsageMode::ON_GPU>;
#endif
