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
    [[nodiscard]] DEVICE bool empty_device() const;

    // Group management
    HOST void update_timestamp_groups() override;  // Call after sorting

    HOST void compute_temporal_weights(double timescale_bound) override;

    // Group lookup
    [[nodiscard]] HOST size_t find_group_after_timestamp(int64_t timestamp) const override;  // For forward walks
    [[nodiscard]] HOST size_t find_group_before_timestamp(int64_t timestamp) const override; // For backward walks

    DEVICE SizeRange get_timestamp_group_range_device(size_t group_idx) const override;
    DEVICE size_t get_timestamp_group_count_device() const override;
    DEVICE size_t find_group_before_timestamp_device(int64_t timestamp) const override;
    DEVICE size_t find_group_after_timestamp_device(int64_t timestamp) const override;

    HOST EdgeDataCUDA* to_device_ptr();
    #endif
};



#endif //EDGEDATACUDA_H
