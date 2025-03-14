#ifndef NODE_EDGE_INDEX_PROXY_H
#define NODE_EDGE_INDEX_PROXY_H

#include "../stores/node_edge_index.cuh"
#include "../stores/edge_data.cuh"
#include "../stores/node_mapping.cuh"

__global__ void get_edge_range_kernel(SizeRange* result, const NodeEdgeIndex* node_edge_index, int dense_node_id, bool forward, bool is_directed);

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const NodeEdgeIndex* node_edge_index, int dense_node_id, size_t group_idx, bool forward, bool is_directed);

__global__ void get_timestamp_group_count_kernel(size_t* result, const NodeEdgeIndex* node_edge_index, int dense_node_id, bool forward, bool is_directed);

class NodeEdgeIndexProxy {
    NodeEdgeIndex* node_edge_index;
    bool owns_node_edge_index;

public:
    explicit NodeEdgeIndexProxy(bool use_gpu = false);

    explicit NodeEdgeIndexProxy(NodeEdgeIndex* existing_node_edge_index);

    ~NodeEdgeIndexProxy();

    void clear() const;

    void rebuild(EdgeData* edge_data, NodeMapping* node_mapping, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;

    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(const EdgeData* edge_data, double timescale_bound) const;

    [[nodiscard]] NodeEdgeIndex* get_node_edge_index() const;
};

#endif // NODE_EDGE_INDEX_PROXY_H
