#ifndef EDGE_DATA_H
#define EDGE_DATA_H

#include <cstddef>
#include <memory>
#include <vector>

#include "../data/temporal_graph_data.cuh"
#include "../data/structs.cuh"
#include "../graph/edge_data.cuh"
#include "TemporalRandomWalk.cuh"

class EdgeData {
    std::unique_ptr<TemporalRandomWalk> self_owned_;

public:
    TemporalGraphData* edge_data;

    explicit EdgeData(bool use_gpu,
                      bool enable_weight_computation = false,
                      bool enable_temporal_node2vec = false);

    explicit EdgeData(TemporalGraphData* shared) : edge_data(shared) {}

    ~EdgeData();

    EdgeData(const EdgeData&) = delete;
    EdgeData& operator=(const EdgeData&) = delete;
    EdgeData(EdgeData&&) noexcept = default;
    EdgeData& operator=(EdgeData&&) noexcept = default;

    // --- Read-back accessors (return host vectors) ---
    std::vector<int>     sources() const                 { return edge_data->sources.to_host_vector(); }
    std::vector<int>     targets() const                 { return edge_data->targets.to_host_vector(); }
    std::vector<int64_t> timestamps() const              { return edge_data->timestamps.to_host_vector(); }
    std::vector<size_t>  timestamp_group_offsets() const { return edge_data->timestamp_group_offsets.to_host_vector(); }
    std::vector<int64_t> unique_timestamps() const       { return edge_data->unique_timestamps.to_host_vector(); }
    std::vector<double>  forward_cumulative_weights_exponential() const {
        return edge_data->forward_cumulative_weights_exponential.to_host_vector();
    }
    std::vector<double>  backward_cumulative_weights_exponential() const {
        return edge_data->backward_cumulative_weights_exponential.to_host_vector();
    }
    std::vector<int>     active_node_ids() const         { return edge_data->active_node_ids.to_host_vector(); }
    std::vector<size_t>  node_adj_offsets() const        { return edge_data->node_adj_offsets.to_host_vector(); }
    std::vector<int>     node_adj_neighbors() const      { return edge_data->node_adj_neighbors.to_host_vector(); }

    // --- Queries (all host-side; Buffer::size() is a host field) ---
    [[nodiscard]] size_t size()  const { return edge_data::size(*edge_data); }
    [[nodiscard]] bool   empty() const { return edge_data::empty(*edge_data); }

    // --- Mutations ---
    void set_size(size_t size) const;

    void add_edges(const std::vector<int>& sources,
                   const std::vector<int>& targets,
                   const std::vector<int64_t>& timestamps) const;

    void push_back(int source, int target, int64_t timestamp) const;

    [[nodiscard]] std::vector<Edge> get_edges() const {
        return edge_data::get_edges(*edge_data);
    }

    void update_timestamp_groups() const;

    void update_temporal_weights(double timescale_bound) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;

    [[nodiscard]] size_t get_timestamp_group_count() const {
        return edge_data->unique_timestamps.size();
    }

    [[nodiscard]] int max_node_id() const { return edge_data->max_node_id; }

    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;

    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const;

    [[nodiscard]] size_t get_memory_used() const {
        return edge_data::get_memory_used(*edge_data);
    }
};

#endif // EDGE_DATA_H
