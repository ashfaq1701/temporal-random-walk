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
#ifdef HAS_CUDA

#endif
};



#endif //NODEMAPPINGCUDA_H
