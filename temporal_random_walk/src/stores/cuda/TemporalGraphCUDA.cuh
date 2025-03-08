#ifndef TEMPORALGRAPHCUDA_H
#define TEMPORALGRAPHCUDA_H

#include "../../data/enums.h"
#include "../interfaces/ITemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA : public ITemporalGraph<GPUUsage> {
public:
    ~TemporalGraphCUDA() override = default;

    explicit TemporalGraphCUDA(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    #ifdef HAS_CUDA
    HOST void add_multiple_edges(const typename ITemporalGraph<GPUUsage>::EdgeVector& new_edges) override;

    HOST void sort_and_merge_edges(size_t start_idx) override;

    HOST void delete_old_edges() override;

    // Timestamp group counting
    [[nodiscard]] HOST size_t count_timestamps_less_than(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_timestamps_greater_than(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const override;

    DEVICE Edge get_edge_at_device(
        RandomPicker<GPUUsage>* picker,
        curandState* rand_state,
        int64_t timestamp = -1,
        bool forward = true) const override;

    DEVICE Edge get_node_edge_at_device(int node_id,
        RandomPicker<GPUUsage>* picker,
        curandState* rand_state,
        int64_t timestamp = -1,
        bool forward = true) const override;

    #endif

    HOST TemporalGraphCUDA* to_device_ptr();
};

#endif //TEMPORALGRAPHCUDA_H
