#ifndef TEMPORAL_RANDOM_WALK_H
#define TEMPORAL_RANDOM_WALK_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../common/const.cuh"
#include "../core/temporal_random_walk.cuh"
#include "../data/enums.cuh"
#include "../data/structs.cuh"
#include "../data/walks_with_edge_features_host.cuh"

class TemporalRandomWalk {
    std::unique_ptr<core::TemporalRandomWalk> impl_;

public:
    explicit TemporalRandomWalk(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        bool enable_temporal_node2vec = false,
        double timescale_bound = DEFAULT_TIMESCALE_BOUND,
        double node2vec_p = DEFAULT_NODE2VEC_P,
        double node2vec_q = DEFAULT_NODE2VEC_Q,
        int walk_padding_value = EMPTY_NODE_VALUE,
        uint64_t global_seed = EMPTY_GLOBAL_SEED,
        bool shuffle_walk_order = DEFAULT_SHUFFLE_WALK_ORDER);

    ~TemporalRandomWalk();

    core::TemporalRandomWalk*       impl()       { return impl_.get(); }
    const core::TemporalRandomWalk* impl() const { return impl_.get(); }

    void add_multiple_edges(
        const int* sources, const int* targets,
        const int64_t* timestamps, size_t edges_size,
        const float* edge_features = nullptr,
        size_t feature_dim = 0) const;

    void add_multiple_edges(
        const std::vector<std::tuple<int, int, int64_t>>& edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0) const;

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes(
        int max_walk_len, const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = KernelLaunchType::FULL_WALK) const;

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch(
        int max_walk_len, const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = KernelLaunchType::FULL_WALK) const;

    WalksWithEdgeFeaturesHost get_random_walks_and_times(
        int max_walk_len, const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = KernelLaunchType::FULL_WALK) const;

    void set_node_features(
        const int* node_ids, size_t num_nodes,
        const float* node_features_data, size_t feature_dim) const;

    [[nodiscard]] size_t get_node_count() const;
    [[nodiscard]] size_t get_edge_count() const;
    [[nodiscard]] std::vector<int> get_node_ids() const;
    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;
    [[nodiscard]] bool   get_is_directed() const;
    void clear() const;
    [[nodiscard]] size_t get_memory_used() const;

    // Node-feature access used by py_interface (replaces the old
    // get_node_features() that returned NodeFeaturesStore*).
    [[nodiscard]] int node_feature_dim() const;
    [[nodiscard]] int node_features_max_node_id() const;
    [[nodiscard]] std::vector<float> node_features_dense() const;
};

#endif // TEMPORAL_RANDOM_WALK_H
