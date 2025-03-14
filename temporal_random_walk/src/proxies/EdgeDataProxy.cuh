#ifndef EDGE_DATA_PROXY_H
#define EDGE_DATA_PROXY_H

#include "../stores/edge_data.cuh"

__global__ void empty_kernel(bool* result, const EdgeData* edge_data);

__global__ void size_kernel(size_t* result, const EdgeData* edge_data);

__global__ void find_group_after_timestamp_kernel(size_t* result, const EdgeData* edge_data, int64_t timestamp);

__global__ void find_group_before_timestamp_kernel(size_t* result, const EdgeData* edge_data, int64_t timestamp);

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const EdgeData* edge_data, size_t group_idx);

class EdgeDataProxy {

    EdgeData* edge_data;
    bool owns_edge_data;

    explicit EdgeDataProxy(bool use_gpu = false);

    explicit EdgeDataProxy(EdgeData* existing_edge_data);

    ~EdgeDataProxy();

    void reserve(size_t size) const;

    void clear() const;

    [[nodiscard]] size_t size() const;

    [[nodiscard]] bool empty() const;

    void add_edges(const std::vector<int>& sources, const std::vector<int>& targets, const std::vector<int64_t>& timestamps) const;

    void push_back(int source, int target, int64_t timestamp) const;

    [[nodiscard]] std::vector<Edge> get_edges() const;

    void update_timestamp_groups() const;

    void update_temporal_weights(double timescale_bound) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;

    [[nodiscard]] size_t get_timestamp_group_count() const;

    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;

    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const;
};

#endif // EDGE_DATA_PROXY_H
