#ifndef EDGEDATACUDA_H
#define EDGEDATACUDA_H

#include "../interfaces/IEdgeData.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA : public IEdgeData<GPUUsage> {
public:
    int* sparse_to_dense_ptr = nullptr;
    size_t sparse_to_dense_size = 0;
    int* dense_to_sparse_ptr = nullptr;
    size_t dense_to_sparse_size = 0;
    bool* is_deleted_ptr = nullptr;
    size_t is_deleted_size = 0;

    #ifdef HAS_CUDA
    // Group management
    HOST void update_timestamp_groups() override;  // Call after sorting

    HOST void compute_temporal_weights(double timescale_bound) override;

    // Group lookup
    [[nodiscard]] HOST size_t find_group_after_timestamp(int64_t timestamp) const override;  // For forward walks
    [[nodiscard]] HOST size_t find_group_before_timestamp(int64_t timestamp) const override; // For backward walks

    HOST EdgeDataCUDA* to_device_ptr();
    #endif
};



#endif //EDGEDATACUDA_H
