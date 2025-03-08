#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <cstdint>
#include <functional>
#include "../cpu/TemporalGraphCPU.cuh"
#include "../cuda/TemporalGraphCUDA.cuh"

#include "../../data/enums.h"

template <GPUUsageMode GPUUsage>
__global__ void get_edge_at_kernel(
    Edge* result,
    TemporalGraphCUDA<GPUUsage>* temporal_graph,
    RandomPicker<GPUUsage>* picker,
    curandState* rand_states,
    int64_t timestamp,
    bool forward);

template <GPUUsageMode GPUUsage>
__global__ void get_node_edge_at_kernel(
    Edge* result, TemporalGraphCUDA<GPUUsage>* temporal_graph,
    int node_id,
    RandomPicker<GPUUsage>* picker,
    curandState* rand_states,
    int64_t timestamp,
    bool forward);

template<GPUUsageMode GPUUsage>
class TemporalGraph {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), TemporalGraphCPU<GPUUsage>, TemporalGraphCUDA<GPUUsage>>;

public:
    ITemporalGraph<GPUUsage>* temporal_graph;

    explicit TemporalGraph(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    void sort_and_merge_edges(size_t start_idx);

    // Edge addition
    void add_multiple_edges(const std::vector<Edge>& new_edges);

    void update_temporal_weights();

    void delete_old_edges();

    // Timestamp group counting
    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    // Edge selection
    [[nodiscard]] Edge get_edge_at(
        RandomPicker<GPUUsage>* picker, int64_t timestamp = -1,
        bool forward = true) const;

    [[nodiscard]] Edge get_node_edge_at(int node_id,
                                            RandomPicker<GPUUsage>* picker,
                                            int64_t timestamp = -1,
                                            bool forward = true) const;

    // Utility methods
    [[nodiscard]] size_t get_total_edges() const;
    [[nodiscard]] size_t get_node_count() const;
    [[nodiscard]] int64_t get_latest_timestamp();
    [[nodiscard]] std::vector<int> get_node_ids() const;
    [[nodiscard]] std::vector<Edge> get_edges();

};

#endif //TEMPORALGRAPH_H
