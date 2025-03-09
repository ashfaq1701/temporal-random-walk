#include "NodeMapping.cuh"

#include <stores/cuda/NodeEdgeIndexCUDA.cuh>

#ifdef HAS_CUDA
template <GPUUsageMode GPUUsage>
__global__ void to_dense_kernel(int* result, NodeMappingCUDA<GPUUsage>* node_mapping, int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping->to_dense_device(sparse_id);
    }
}
#endif

template<GPUUsageMode GPUUsage>
NodeMapping<GPUUsage>::NodeMapping(): node_mapping(new BaseType()) {}

template<GPUUsageMode GPUUsage>
NodeMapping<GPUUsage>::NodeMapping(INodeMapping<GPUUsage>* node_mapping): node_mapping(node_mapping) {}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::update(const IEdgeData<GPUUsage>* edges, size_t start_idx, size_t end_idx)
{
    using EdgeDataType = typename INodeMapping<GPUUsage>::EdgeDataType;
    const auto* typed_edges = static_cast<const EdgeDataType*>(edges);

    node_mapping->update(typed_edges, start_idx, end_idx);
}

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_dense(int sparse_id) const
{
    #ifdef HAS_CUDA
    if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
        int host_result = -1;
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        NodeMappingCUDA<GPUUsage>* node_mapping_cuda = static_cast<NodeMappingCUDA<GPUUsage>*>(node_mapping)->to_device_ptr();

        to_dense_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            node_mapping_cuda,
            sparse_id
        );

        cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(node_mapping_cuda);

        return host_result;
    }
    else
    #endif
    {
        return node_mapping->to_dense(sparse_id);
    }
}

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_sparse(int dense_idx) const
{
    return node_mapping->to_sparse(dense_idx);
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::size() const
{
    return node_mapping->size();
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::active_size() const
{
    return node_mapping->active_size();
}

template<GPUUsageMode GPUUsage>
HOST std::vector<int> NodeMapping<GPUUsage>::get_active_node_ids() const
{
    std::vector<int> result;

    auto active_ids = node_mapping->get_active_node_ids();
    for (int i = 0; i < active_ids.size(); i++)
    {
        result.push_back(active_ids[i]);
    }

    return result;
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::clear()
{
    node_mapping->clear();
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::reserve(size_t size)
{
    node_mapping->reserve(size);
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::mark_node_deleted(int sparse_id)
{
    node_mapping->mark_node_deleted(sparse_id);
}

template<GPUUsageMode GPUUsage>
bool NodeMapping<GPUUsage>::has_node(int sparse_id)
{
    return node_mapping->has_node(sparse_id);
}

template<GPUUsageMode GPUUsage>
std::vector<int> NodeMapping<GPUUsage>::get_all_sparse_ids() const
{
    std::vector<int> result;
    auto all_ids = node_mapping->get_all_sparse_ids();
    for (int i = 0; i < all_ids.size(); i++)
    {
        result.push_back(all_ids[i]);
    }
    return result;
}

template class NodeMapping<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeMapping<GPUUsageMode::ON_GPU>;
#endif
