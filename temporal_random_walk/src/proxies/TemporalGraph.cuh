#ifndef TEMPORAL_GRAPH_H
#define TEMPORAL_GRAPH_H

#include <vector>
#include "../stores/temporal_graph.cuh"
#include "../data/structs.cuh"
#include "../data/enums.cuh"
#include "../common/const.cuh"

#ifdef HAS_CUDA

__global__ void get_total_edges_kernel(size_t* result, const TemporalGraphStore* graph);
__global__ void get_edge_at_kernel(Edge* result, const TemporalGraphStore* graph, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);
__global__ void get_node_edge_at_kernel(Edge* result, TemporalGraphStore* graph, int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);

#endif

class TemporalGraph {
public:
    TemporalGraphStore* graph;
    bool owns_graph;

    explicit TemporalGraph(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        double timescale_bound = -1);

    explicit TemporalGraph(TemporalGraphStore* existing_graph);

    ~TemporalGraph();

    TemporalGraph& operator=(const TemporalGraph& other);

    void update_temporal_weights() const;

    [[nodiscard]] size_t get_total_edges() const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] int64_t get_latest_timestamp() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<Edge> get_edges() const;

    void add_multiple_edges(const std::vector<Edge>& new_edges, int max_node_id) const;

    void sort_and_merge_edges(size_t start_idx) const;

    void delete_old_edges() const;

    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const;

    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const;

    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;

    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    [[nodiscard]] Edge get_edge_at(RandomPickerType picker_type, int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] Edge get_node_edge_at(int node_id, RandomPickerType picker_type, int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] TemporalGraphStore* get_graph() const;
};

#endif // TEMPORAL_GRAPH_H