#ifndef EDGEDATA_H
#define EDGEDATA_H

#include "../cpu/EdgeDataCPU.cuh"
#include "../cuda/EdgeDataCUDA.cuh"

#include "../../data/enums.h"

template <GPUUsageMode GPUUsage>
__global__ void check_empty_kernel(bool* result, EdgeDataCUDA<GPUUsage>* edge_data);

template <GPUUsageMode GPUUsage>
__global__ void get_timestamp_group_range_kernel(SizeRange* result, EdgeDataCUDA<GPUUsage>* edge_data, size_t group_idx);

template <GPUUsageMode GPUUsage>
__global__ void get_timestamp_group_count_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data);

template <GPUUsageMode GPUUsage>
__global__ void find_group_after_timestamp_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data, int64_t timestamp);

template <GPUUsageMode GPUUsage>
__global__ void find_group_before_timestamp_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data, int64_t timestamp);

template<GPUUsageMode GPUUsage>
class EdgeData {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), EdgeDataCPU<GPUUsage>, EdgeDataCUDA<GPUUsage>>;

public:
    IEdgeData<GPUUsage>* edge_data;

    virtual ~EdgeData() = default;

    EdgeData();
    explicit EdgeData(IEdgeData<GPUUsage>* edge_date);

    auto& sources() { return edge_data->sources; }
    const auto& sources() const { return edge_data->sources; }

    auto& targets() { return edge_data->targets; }
    const auto& targets() const { return edge_data->targets; }

    auto& timestamps() { return edge_data->timestamps; }
    const auto& timestamps() const { return edge_data->timestamps; }

    auto& timestamp_group_offsets() { return edge_data->timestamp_group_offsets; }
    const auto& timestamp_group_offsets() const { return edge_data->timestamp_group_offsets; }

    auto& unique_timestamps() { return edge_data->unique_timestamps; }
    const auto& unique_timestamps() const { return edge_data->unique_timestamps; }

    auto& forward_cumulative_weights_exponential() { return edge_data->forward_cumulative_weights_exponential; }
    const auto& forward_cumulative_weights_exponential() const { return edge_data->forward_cumulative_weights_exponential; }

    auto& backward_cumulative_weights_exponential() { return edge_data->backward_cumulative_weights_exponential; }
    const auto& backward_cumulative_weights_exponential() const { return edge_data->backward_cumulative_weights_exponential; }

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    virtual void resize(size_t new_size);

    void add_edges(int* src, int* tgt, int64_t* ts, size_t size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<Edge> get_edges();

    // Group management
    void update_timestamp_groups();
    void update_temporal_weights(double timescale_bound);

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx);
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const;
};

#endif //EDGEDATA_H
