#ifndef NODEMAPPING_CUDA_H
#define NODEMAPPING_CUDA_H

#include "../cpu/NodeMapping.cuh"
#include "../../cuda_common/config.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA final : public NodeMapping<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEMAPPING_CUDA_H
