#ifndef NODEMAPPINGCUDA_H
#define NODEMAPPINGCUDA_H

#include "../../data/enums.h"
#include "../interfaces/INodeMapping.cuh"

#ifdef HAS_CUDA
HOST DEVICE int to_dense(const int* sparse_to_dense, int sparse_id, int size);
HOST DEVICE void mark_node_deleted(bool* is_deleted, int sparse_id, int size);
#endif

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA : public INodeMapping<GPUUsage> {
public:
    int* sparse_to_dense_ptr = nullptr;
    size_t sparse_to_dense_size = 0;
    int* dense_to_sparse_ptr = nullptr;
    size_t dense_to_sparse_size = 0;
    bool* is_deleted_ptr = nullptr;
    size_t is_deleted_size = 0;

    #ifdef HAS_CUDA

    HOST void update(const typename INodeMapping<GPUUsage>::EdgeDataType* edges, size_t start_idx, size_t end_idx) override;

    DEVICE int to_dense_device(int sparse_id) const override;

    HOST typename INodeMapping<GPUUsage>::IntVector get_active_node_ids() const override;

    HOST NodeMappingCUDA* to_device_ptr();

    #endif
};



#endif //NODEMAPPINGCUDA_H
