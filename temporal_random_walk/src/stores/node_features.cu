#include "node_features.cuh"

#include <cstring>
#include <stdexcept>
#include <omp.h>

void node_features::ensure_size(NodeFeaturesStore* node_features_store, const int max_node_id) {
    if (max_node_id <= node_features_store->max_node_id) {
        return;
    }

    if (node_features_store->node_feature_dim == 0) {
        node_features_store->max_node_id = max_node_id;
        return;
    }

    const size_t old_values = node_features_store->node_features_size;
    const size_t new_values = static_cast<size_t>(max_node_id + 1) * node_features_store->node_feature_dim;

    resize_memory(
        &node_features_store->node_features,
        old_values,
        new_values,
        false);

    if (new_values > old_values) {
        std::memset(node_features_store->node_features + old_values, 0, (new_values - old_values) * sizeof(float));
    }

    node_features_store->node_features_size = new_values;
    node_features_store->max_node_id = max_node_id;
}

void node_features::set_node_features(
    NodeFeaturesStore* store,
    const int max_node_id,
    const int* node_ids,
    const size_t num_nodes,
    const float* node_features,
    const size_t feature_dim) {

    if (num_nodes == 0) {
        store->node_feature_dim = feature_dim;
        return;
    }

    if (feature_dim == 0) {
        throw std::runtime_error("feature_dim must be greater than 0 when setting node features");
    }

    if (store->node_feature_dim != 0 && store->node_feature_dim != feature_dim) {
        throw std::runtime_error("feature_dim mismatch with existing NodeFeaturesStore");
    }

    store->node_feature_dim = feature_dim;
    ensure_size(store, max_node_id);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_nodes; ++i) {
        const int node_id = node_ids[i];
        float* dst = store->node_features + (static_cast<size_t>(node_id) * feature_dim);
        const float* src = node_features + (i * feature_dim);
        std::memcpy(dst, src, feature_dim * sizeof(float));
    }
}
