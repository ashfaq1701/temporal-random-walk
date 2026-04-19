#ifndef NODE_FEATURES_H
#define NODE_FEATURES_H

#include "../graph/node_features.cuh"

class NodeFeatures {
public:
    NodeFeaturesStore* node_features;
    explicit NodeFeatures();

    ~NodeFeatures();

    void set_node_features(
        int max_node_id,
        const int* node_ids,
        size_t num_nodes,
        const float* node_features,
        size_t feature_dim) const;

    [[nodiscard]] size_t node_feature_dim() const;

    [[nodiscard]] int max_node_id() const;

    [[nodiscard]] NodeFeaturesStore* get_node_features() const;
};

#endif // NODE_FEATURES_H
