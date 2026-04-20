#ifndef NODE_FEATURES_H
#define NODE_FEATURES_H

#include <cstddef>
#include <memory>
#include <vector>

#include "../data/temporal_graph_data.cuh"
#include "../graph/node_features.cuh"
#include "TemporalRandomWalk.cuh"

class NodeFeatures {
    std::unique_ptr<TemporalRandomWalk> self_owned_;

public:
    TemporalGraphData* data;

    NodeFeatures();

    explicit NodeFeatures(TemporalGraphData* shared) : data(shared) {}

    ~NodeFeatures();

    NodeFeatures(const NodeFeatures&) = delete;
    NodeFeatures& operator=(const NodeFeatures&) = delete;
    NodeFeatures(NodeFeatures&&) noexcept = default;
    NodeFeatures& operator=(NodeFeatures&&) noexcept = default;

    void set_node_features(
        int max_node_id,
        const int* node_ids,
        size_t num_nodes,
        const float* node_features_data,
        size_t feature_dim) const;

    [[nodiscard]] size_t node_feature_dim() const { return data->node_feature_dim; }
    [[nodiscard]] int    max_node_id()      const { return data->max_node_id; }

    // Read-back accessor (returns a host-side copy of all node-feature rows).
    [[nodiscard]] std::vector<float> node_features_vec() const {
        return data->node_features.to_host_vector();
    }

    [[nodiscard]] const float* node_features_raw() const {
        return data->node_features.data();
    }
};

#endif // NODE_FEATURES_H
