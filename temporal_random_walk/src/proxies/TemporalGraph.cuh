#ifndef TEMPORAL_GRAPH_H
#define TEMPORAL_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../common/const.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/enums.cuh"
#include "../graph/temporal_graph.cuh"
#include "../graph/edge_selectors.cuh"
#include "TemporalRandomWalk.cuh"

class TemporalGraph {
    std::unique_ptr<TemporalRandomWalk> self_owned_;

public:
    TemporalGraphData* graph;

    explicit TemporalGraph(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        bool enable_temporal_node2vec = false,
        double timescale_bound = -1,
        double node2vec_p = DEFAULT_NODE2VEC_P,
        double node2vec_q = DEFAULT_NODE2VEC_Q);

    explicit TemporalGraph(TemporalGraphData* shared) : graph(shared) {}

    ~TemporalGraph();

    TemporalGraph(const TemporalGraph&) = delete;
    TemporalGraph& operator=(const TemporalGraph&) = delete;
    TemporalGraph(TemporalGraph&&) noexcept = default;
    TemporalGraph& operator=(TemporalGraph&&) noexcept = default;

    void update_temporal_weights() const {
        temporal_graph::update_temporal_weights(*graph);
    }

    [[nodiscard]] size_t get_total_edges() const {
        return temporal_graph::get_total_edges(*graph);
    }

    [[nodiscard]] size_t get_node_count() const {
        return temporal_graph::get_node_count(*graph);
    }

    [[nodiscard]] int64_t get_latest_timestamp() const {
        return temporal_graph::get_latest_timestamp(*graph);
    }

    [[nodiscard]] std::vector<int> get_node_ids() const {
        return temporal_graph::get_node_ids(*graph);
    }

    [[nodiscard]] std::vector<Edge> get_edges() const {
        return temporal_graph::get_edges(*graph);
    }

    void add_multiple_edges(
        const std::vector<int>& sources,
        const std::vector<int>& targets,
        const std::vector<int64_t>& timestamps,
        const float* edge_features = nullptr,
        size_t feature_dim = 0) const;

    void add_multiple_edges(const std::vector<Edge>& edges) const;

    void sort_and_merge_edges(size_t start_idx) const;

    void delete_old_edges() const;

    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    [[nodiscard]] double compute_node2vec_beta(int prev_node, int w) const;

    [[nodiscard]] Edge get_edge_at_with_provided_nums(
        RandomPickerType picker_type, const double* rand_nums,
        int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] Edge get_edge_at(
        RandomPickerType picker_type, int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] Edge get_node_edge_at(
        int node_id, RandomPickerType picker_type,
        int64_t timestamp, int prev_node, bool forward = true) const;

    [[nodiscard]] size_t get_memory_used() const {
        return temporal_graph::get_memory_used(*graph);
    }
};

#endif // TEMPORAL_GRAPH_H
