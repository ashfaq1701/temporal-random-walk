#ifndef NODEMAPPING_H
#define NODEMAPPING_H

#include <functional>
#include "../cpu/NodeMappingCPU.cuh"
#include "../cuda/NodeMappingCUDA.cuh"

#include "../../data/enums.h"

#ifdef HAS_CUDA
template <GPUUsageMode GPUUsage>
__global__ void to_dense_kernel(int* result, NodeMappingCUDA<GPUUsage>* node_mapping, int sparse_id);
#endif

template<GPUUsageMode GPUUsage>
class NodeMapping {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeMappingCPU<GPUUsage>, NodeMappingCUDA<GPUUsage>>;

public:

    INodeMapping<GPUUsage>* node_mapping;

    NodeMapping();
    explicit NodeMapping(INodeMapping<GPUUsage>* node_mapping);

    // Accessors for sparse_to_dense
    auto& sparse_to_dense() { return node_mapping->sparse_to_dense; }
    const auto& sparse_to_dense() const { return node_mapping->sparse_to_dense; }

    // Accessors for dense_to_sparse
    auto& dense_to_sparse() { return node_mapping->dense_to_sparse; }
    const auto& dense_to_sparse() const { return node_mapping->dense_to_sparse; }

    // Accessors for is_deleted
    auto& is_deleted() { return node_mapping->is_deleted; }
    const auto& is_deleted() const { return node_mapping->is_deleted; }

    void update(const IEdgeData<GPUUsage>* edges, size_t start_idx, size_t end_idx);
    [[nodiscard]] int to_dense(int sparse_id) const;
    [[nodiscard]] int to_sparse(int dense_idx) const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t active_size() const;

    // Helper methods
    [[nodiscard]] virtual HOST std::vector<int> get_active_node_ids() const;
    void clear();
    void reserve(size_t size);
    void mark_node_deleted(int sparse_id);
    [[nodiscard]] bool has_node(int sparse_id);
    [[nodiscard]] std::vector<int> get_all_sparse_ids() const;
};

#endif //NODEMAPPING_H
