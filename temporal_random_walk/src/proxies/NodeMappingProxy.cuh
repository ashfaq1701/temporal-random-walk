#ifndef NODE_MAPPING_PROXY_H
#define NODE_MAPPING_PROXY_H

#include "../stores/node_mapping.cuh"

__global__ void to_dense_kernel(int* result, const NodeMapping* node_mapping, int sparse_id);

__global__ void to_sparse_kernel(int* result, const NodeMapping* node_mapping, int dense_id);

__global__ void size_kernel(size_t* result, const NodeMapping* node_mapping);

__global__ void has_node_kernel(bool* result, const NodeMapping* node_mapping, int sparse_id);

class NodeMappingProxy {
    NodeMapping* node_mapping;
    bool owns_node_mapping;

public:
    explicit NodeMappingProxy(bool use_gpu = false);

    explicit NodeMappingProxy(NodeMapping* existing_node_mapping);

    ~NodeMappingProxy();

    [[nodiscard]] int to_dense(int sparse_id) const;

    [[nodiscard]] int to_sparse(int dense_id) const;

    [[nodiscard]] size_t size() const;

    [[nodiscard]] size_t active_size() const;

    [[nodiscard]] std::vector<int> get_active_node_ids() const;

    void clear() const;

    void reserve(size_t size) const;

    void mark_node_deleted(int sparse_id) const;

    [[nodiscard]] bool has_node(int sparse_id) const;

    [[nodiscard]] std::vector<int> get_all_sparse_ids() const;

    void update(const EdgeData* edge_data, size_t start_idx, size_t end_idx) const;

    [[nodiscard]] NodeMapping* get_node_mapping() const;
};

#endif // NODE_MAPPING_PROXY_H
