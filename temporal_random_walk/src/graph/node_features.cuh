#ifndef NODE_FEATURES_STORE_H
#define NODE_FEATURES_STORE_H

#include <cstddef>
#include "../common/macros.cuh"
#include "../common/memory.cuh"

struct NodeFeaturesStore {
    bool use_gpu = false;
    bool owns_data = true;

    int max_node_id = -1;
    size_t node_feature_dim = 0;

    float* node_features = nullptr;
    size_t node_features_size = 0;

    NodeFeaturesStore() = default;

    ~NodeFeaturesStore() {
        if (owns_data) {
            clear_memory(&node_features, false);
        } else {
            node_features = nullptr;
        }
    }
};

namespace node_features {

    HOST inline size_t size(const NodeFeaturesStore* node_features_store) {
        return node_features_store->node_features_size;
    }

    HOST inline bool empty(const NodeFeaturesStore* node_features_store) {
        return node_features_store->node_features_size == 0;
    }

    HOST void ensure_size(NodeFeaturesStore* node_features_store, int max_node_id);

    HOST void set_node_features(
        NodeFeaturesStore* store,
        int max_node_id,
        const int* node_ids,
        size_t num_nodes,
        const float* node_features,
        size_t feature_dim);

}

#endif // NODE_FEATURES_STORE_H
