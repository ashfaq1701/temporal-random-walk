#ifndef TEMPORAL_RANDOM_WALK_H
#define TEMPORAL_RANDOM_WALK_H

#include <vector>
#include <thread>
#include "../core/temporal_random_walk.cuh"
#include "../data/walk_set/walk_set.cuh"
#include "../data/structs.cuh"
#include "../data/enums.cuh"
#include "../common/const.cuh"
#include "NodeFeatures.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_count_kernel(size_t* result, const TemporalRandomWalkStore* temporal_random_walk);

#endif

class TemporalRandomWalk {
    bool use_gpu;
    TemporalRandomWalkStore* temporal_random_walk;
    NodeFeatures* node_features;

public:
    explicit TemporalRandomWalk(
        bool is_directed,
        bool use_gpu,

        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        bool enable_temporal_node2vec=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,

        double node2vec_p=DEFAULT_NODE2VEC_P,
        double node2vec_q=DEFAULT_NODE2VEC_Q,

        double spatiotemporal_alpha = DEFAULT_SPATIOTEMPORAL_ALPHA,
        double spatiotemporal_beta = DEFAULT_SPATIOTEMPORAL_BETA,
        double spatiotemporal_gamma = DEFAULT_SPATIOTEMPORAL_GAMMA,

        int walk_padding_value=EMPTY_NODE_VALUE,
        uint64_t global_seed=EMPTY_GLOBAL_SEED,
        bool shuffle_walk_order=DEFAULT_SHUFFLE_WALK_ORDER);

    ~TemporalRandomWalk();

    void add_multiple_edges(
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t edges_size,
        const float* edge_features = nullptr,
        size_t feature_dim = 0) const;

    void add_multiple_edges(
        const std::vector<std::tuple<int, int, int64_t>>& edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0) const;

    WalksWithEdgeFeatures get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type=KernelLaunchType::FULL_WALK) const;

    WalksWithEdgeFeatures get_random_walks_and_times_for_last_batch(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type=KernelLaunchType::FULL_WALK) const;

    WalksWithEdgeFeatures get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type=KernelLaunchType::FULL_WALK) const;

    void set_node_features(
        const int* node_ids,
        size_t num_nodes,
        const float* node_features_data,
        size_t feature_dim) const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] NodeFeaturesStore* get_node_features() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;

    [[nodiscard]] bool get_is_directed() const;

    void clear() const;

    [[nodiscard]] size_t get_memory_used() const;
};

#endif // TEMPORAL_RANDOM_WALK_H
