#ifndef EDGEDATA_CPU_H
#define EDGEDATA_CPU_H

#include <cstdint>

#include "../interfaces/IEdgeData.cuh"
#include "../../data/enums.h"
#include "../../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCPU : public IEdgeData<GPUUsage> {

public:
    // Group management
    HOST void update_timestamp_groups() override;  // Call after sorting

    HOST void compute_temporal_weights(double timescale_bound) override;

    // Group lookup
    [[nodiscard]] HOST size_t find_group_after_timestamp(int64_t timestamp) const override;  // For forward walks
    [[nodiscard]] HOST size_t find_group_before_timestamp(int64_t timestamp) const override; // For backward walks

    HOST EdgeDataCPU* to_device_ptr();
};

#endif //EDGEDATA_CPU_H
