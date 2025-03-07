#include "NodeMappingCPU.cuh"
#include <algorithm>

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCPU<GPUUsage>::update(const typename INodeMapping<GPUUsage>::EdgeDataType* edges, const size_t start_idx, const size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({
            max_node_id,
            static_cast<int>(edges->sources[i]),
            static_cast<int>(edges->targets[i])
        });
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= this->sparse_to_dense.size()) {
        this->sparse_to_dense.resize(max_node_id + 1, -1);
        this->is_deleted.resize(max_node_id + 1, true);
    }

    typename INodeMapping<GPUUsage>::IntVector new_nodes;
    new_nodes.reserve((end_idx - start_idx) * 2);

    for (size_t i = start_idx; i < end_idx; i++) {
        new_nodes.push_back(edges->sources[i]);
        new_nodes.push_back(edges->targets[i]);
    }

    std::sort(new_nodes.begin(), new_nodes.end());

    // Map unmapped nodes
    for (int node : new_nodes) {
        if (node < 0) continue;

        this->is_deleted[node] = false;

        if (this->sparse_to_dense[node] == -1) {
            this->sparse_to_dense[node] = static_cast<int>(this->dense_to_sparse.size());
            this->dense_to_sparse.push_back(node);
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST NodeMappingCPU<GPUUsage>* NodeMappingCPU<GPUUsage>::to_device_ptr() {
    throw std::runtime_error("Unsupported operation");
}

template class NodeMappingCPU<GPUUsageMode::ON_CPU>;
