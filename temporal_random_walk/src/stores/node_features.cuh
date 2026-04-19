#ifndef NODE_FEATURES_STORE_H
#define NODE_FEATURES_STORE_H

#include <cstddef>
#include "../common/cuda_buffer.cuh"
#include "../common/macros.cuh"

struct NodeFeaturesStore {
    bool use_gpu = false;

    int max_node_id = -1;
    size_t node_feature_dim = 0;

    CudaBuffer<float> node_features;

    NodeFeaturesStore() = default;
};

namespace node_features {

    HOST inline size_t size(const NodeFeaturesStore* node_features_store) {
        return node_features_store->node_features.size();
    }

    HOST inline bool empty(const NodeFeaturesStore* node_features_store) {
        return node_features_store->node_features.size() == 0;
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
