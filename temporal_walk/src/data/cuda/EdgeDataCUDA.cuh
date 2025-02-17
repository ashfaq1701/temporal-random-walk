#ifndef EDGEDATA_CUDA_H
#define EDGEDATA_CUDA_H

#include "../cpu/EdgeData.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA final : public EdgeData<GPUUsage>, public PolicyProvider<GPUUsage> {
public:
#ifdef HAS_CUDA
    std::vector<std::tuple<int, int, int64_t>> get_edges() override;

    void update_timestamp_groups() override;

    void update_temporal_weights(double timescale_bound) override;

    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const override;

    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const override;

#endif
};

#endif //EDGEDATA_CUDA_H
