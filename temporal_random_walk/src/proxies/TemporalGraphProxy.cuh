#ifndef TEMPORAL_GRAPH_PROXY_H
#define TEMPORAL_GRAPH_PROXY_H

#include "../stores/temporal_graph.cuh"
#include "../data/structs.cuh"
#include "../data/enums.cuh"

__global__ void get_total_edges_kernel(size_t* result, const TemporalGraph* graph);
__global__ void get_edge_at_kernel(Edge* result, const TemporalGraph* graph, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);
__global__ void get_node_edge_at_kernel(Edge* result, TemporalGraph* graph, int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);

class TemporalGraphProxy {
public:
    TemporalGraph* graph;
    bool owns_graph;

    explicit TemporalGraphProxy(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        double timescale_bound = -1);

    explicit TemporalGraphProxy(TemporalGraph* existing_graph);

    ~TemporalGraphProxy();

    TemporalGraphProxy& operator=(const TemporalGraphProxy& other);

    void update_temporal_weights() const;

    [[nodiscard]] size_t get_total_edges() const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] int64_t get_latest_timestamp() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<Edge> get_edges() const;

    void add_multiple_edges(const std::vector<Edge>& new_edges) const;

    void sort_and_merge_edges(size_t start_idx) const;

    void delete_old_edges() const;

    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const;

    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const;

    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;

    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    [[nodiscard]] Edge get_edge_at(RandomPickerType picker_type, int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] Edge get_node_edge_at(int node_id, RandomPickerType picker_type, int64_t timestamp = -1, bool forward = true) const;

    [[nodiscard]] TemporalGraph* get_graph() const;
};

#endif // TEMPORAL_GRAPH_PROXY_H