#ifndef NODEMAPPING_H
#define NODEMAPPING_H

#include <functional>
#include "../cpu/NodeMappingCPU.cuh"
#include "../cuda/NodeMappingCUDA.cuh"

#include "../../data/enums.h"

template<GPUUsageMode GPUUsage>
class NodeMapping {
protected:
    using NodeMappingType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeMappingCPU<GPUUsage>, NodeMappingCUDA<GPUUsage>>;

    NodeMappingType node_mapping;

public:
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
    [[nodiscard]] std::vector<int> get_all_sparse_ids_host() const;
};

#endif //NODEMAPPING_H
