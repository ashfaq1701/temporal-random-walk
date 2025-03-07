#ifndef TEMPORALGRAPH_CPU_H
#define TEMPORALGRAPH_CPU_H

#include <cstdint>

#include "../../data/enums.h"
#include "../../random/RandomPicker.h"

#include "../interfaces/ITemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCPU : public ITemporalGraph<GPUUsage>
{
public:
    ~TemporalGraphCPU() override = default;

    HOST explicit TemporalGraphCPU(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    HOST void sort_and_merge_edges(size_t start_idx) override;

    // Edge addition
    HOST void add_multiple_edges(const typename ITemporalGraph<GPUUsage>::EdgeVector& new_edges) override;

    HOST void delete_old_edges() override;

    // Timestamp group counting
    [[nodiscard]] HOST size_t count_timestamps_less_than(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_timestamps_greater_than(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const override;

    // Edge selection
    [[nodiscard]] HOST Edge get_edge_at_host(RandomPicker<GPUUsage>* picker, int64_t timestamp = -1,
                                            bool forward = true) const override;

    [[nodiscard]] HOST Edge get_node_edge_at_host(int node_id,
                                                RandomPicker<GPUUsage>* picker,
                                                int64_t timestamp = -1,
                                                bool forward = true) const override;

    HOST TemporalGraphCPU* to_device_ptr();
};

#endif //TEMPORALGRAPH_CPU_H
