#ifndef TEMPORAL_RANDOM_WALK_PROXY_H
#define TEMPORAL_RANDOM_WALK_PROXY_H

#include "../src/data/structs.cuh"
#include "../src/data/enums.cuh"
#include "../src/core/temporal_random_walk.cuh"

class TemporalRandomWalkProxy {

    bool use_gpu;
    TemporalRandomWalk* temporal_random_walk;

public:
    explicit TemporalRandomWalkProxy(
        const bool is_directed,
        const bool use_gpu=false,
        const int64_t max_time_capacity=-1,
        const bool enable_weight_computation=false,
        const double timescale_bound=-1,
        const size_t n_threads=std::thread::hardware_concurrency()): use_gpu(use_gpu) {
        temporal_random_walk = new TemporalRandomWalk(is_directed, use_gpu, max_time_capacity, enable_weight_computation, timescale_bound, n_threads);
    }

    ~TemporalRandomWalkProxy() {
        delete temporal_random_walk;
    }

    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edges) const {
        Edge* edge_array = new Edge[edges.size()];
        for (size_t idx = 0; idx < edges.size(); idx++) {
            const auto& [u, i, ts] = edges[idx];
            edge_array[i] = Edge(u, i, ts);
        }

        temporal_random_walk::add_multiple_edges(temporal_random_walk, edge_array, edges.size());

        delete[] edge_array;
    }

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        const int max_walk_len,
        RandomPickerType* walk_bias,
        const int num_walks_per_node,
        RandomPickerType* initial_edge_bias=nullptr,
        const WalkDirection walk_direction=WalkDirection::Forward_In_Time) const {

        WalkSet walk_set;
        if (use_gpu) {
            walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
                temporal_random_walk,
                max_walk_len,
                walk_bias,
                num_walks_per_node,
                initial_edge_bias,
                walk_direction);
        } else {
            walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
                temporal_random_walk,
                max_walk_len,
                walk_bias,
                num_walks_per_node,
                initial_edge_bias,
                walk_direction);
        }

        std::vector<std::vector<NodeWithTime>> result(walk_set.num_walks);
        for (size_t walk_idx = 0; walk_idx < walk_set.num_walks; walk_idx++) {
            const size_t walk_len = walk_set.get_walk_len(walk_idx);
            result[walk_idx].reserve(walk_len);

            for (size_t hop = 0; hop < walk_len; hop++) {
                NodeWithTime node_time = walk_set.get_walk_hop(walk_idx, hop);
                result[walk_idx].push_back(node_time);
            }
        }

        return result;
    }

    std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        const int max_walk_len,
        RandomPickerType* walk_bias,
        const int num_walks_per_node,
        RandomPickerType* initial_edge_bias=nullptr,
        const WalkDirection walk_direction=WalkDirection::Forward_In_Time) {

        auto walks_with_times = get_random_walks_and_times_for_all_nodes(
            max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);

        std::vector<std::vector<int>> result(walks_with_times.size());
        for (size_t i = 0; i < walks_with_times.size(); i++) {
            result[i].reserve(walks_with_times[i].size());
            for (const auto& node_time : walks_with_times[i]) {
                result[i].push_back(node_time.node);
            }
        }

        return result;
    }

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        const int max_walk_len,
        RandomPickerType* walk_bias,
        const int num_walks_total,
        RandomPickerType* initial_edge_bias=nullptr,
        const WalkDirection walk_direction=WalkDirection::Forward_In_Time) const {

        WalkSet walk_set;
        if (use_gpu) {
            walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
                temporal_random_walk,
                max_walk_len,
                walk_bias,
                num_walks_total,
                initial_edge_bias,
                walk_direction);
        } else {
            walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
                temporal_random_walk,
                max_walk_len,
                walk_bias,
                num_walks_total,
                initial_edge_bias,
                walk_direction);
        }

        std::vector<std::vector<NodeWithTime>> result(walk_set.num_walks);
        for (size_t walk_idx = 0; walk_idx < walk_set.num_walks; walk_idx++) {
            const size_t walk_len = walk_set.get_walk_len(walk_idx);
            result[walk_idx].reserve(walk_len);

            for (size_t hop = 0; hop < walk_len; hop++) {
                NodeWithTime node_time = walk_set.get_walk_hop(walk_idx, hop);
                result[walk_idx].push_back(node_time);
            }
        }

        return result;
    }

    std::vector<std::vector<int>> get_random_walks(
        const int max_walk_len,
        RandomPickerType* walk_bias,
        const int num_walks_total,
        RandomPickerType* initial_edge_bias=nullptr,
        const WalkDirection walk_direction=WalkDirection::Forward_In_Time) {

        auto walks_with_times = get_random_walks_and_times_for_all_nodes(
            max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);

        std::vector<std::vector<int>> result(walks_with_times.size());
        for (size_t i = 0; i < walks_with_times.size(); i++) {
            result[i].reserve(walks_with_times[i].size());
            for (const auto& node_time : walks_with_times[i]) {
                result[i].push_back(node_time.node);
            }
        }

        return result;
    }

    [[nodiscard]] size_t get_node_count() const {
        return temporal_random_walk::get_node_count(temporal_random_walk);
    }

    [[nodiscard]] size_t get_edge_count() const {
        return temporal_random_walk::get_edge_count(temporal_random_walk);
    }

    [[nodiscard]] std::vector<int> get_node_ids() const {
        const DataBlock<int> node_ids = temporal_random_walk::get_node_ids(temporal_random_walk);
        std::vector<int> result(node_ids.data, node_ids.data + node_ids.size);
        return result;
    }

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const {
        const DataBlock<Edge> edges = temporal_random_walk::get_edges(temporal_random_walk);
        std::vector<std::tuple<int, int, int64_t>> result;
        result.reserve(edges.size);

        for (size_t i = 0; i < edges.size; i++) {
            result.emplace_back(edges.data[i].u, edges.data[i].i, edges.data[i].ts);
        }

        return result;
    }

    [[nodiscard]] bool get_is_directed() const {
        return temporal_random_walk::get_is_directed(temporal_random_walk);
    }

    void clear() const {
        temporal_random_walk::clear(temporal_random_walk);
    }
};

#endif
