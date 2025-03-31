#ifndef TEMPORAL_RANDOM_WALK_H
#define TEMPORAL_RANDOM_WALK_H

#include <vector>
#include <thread>
#include "../core/temporal_random_walk.cuh"
#include "../data/structs.cuh"
#include "../data/enums.cuh"
#include "../common/const.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_count_kernel(size_t* result, const TemporalRandomWalkStore* temporal_random_walk);

#endif

class TemporalRandomWalk {
    bool use_gpu;
    TemporalRandomWalkStore* temporal_random_walk;

public:
    explicit TemporalRandomWalk(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,
        int node_count_max_bound=DEFAULT_NODE_COUNT_MAX_BOUND,
        size_t n_threads=std::thread::hardware_concurrency());

    ~TemporalRandomWalk();

    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edges) const;

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;

    [[nodiscard]] bool get_is_directed() const;

    void clear() const;
};

#endif // TEMPORAL_RANDOM_WALK_PROXY_H
