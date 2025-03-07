#ifndef NODEMAPPING_CPU_H
#define NODEMAPPING_CPU_H

#include "../../data/enums.h"
#include "../interfaces/IEdgeData.cuh"
#include "../interfaces/INodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCPU : public INodeMapping<GPUUsage> {
public:
   ~NodeMappingCPU() override = default;

   HOST void update(const typename INodeMapping<GPUUsage>::EdgeDataType* edges, size_t start_idx, size_t end_idx) override;

   HOST NodeMappingCPU* to_device_ptr();
};

#endif //NODEMAPPING_CPU_H
