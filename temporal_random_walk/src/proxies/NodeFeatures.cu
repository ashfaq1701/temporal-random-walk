#include "NodeFeatures.cuh"

NodeFeatures::NodeFeatures() {
    node_features = new NodeFeaturesStore();
}

NodeFeatures::~NodeFeatures() {
    delete node_features;
}

void NodeFeatures::set_node_features(
    const int max_node_id,
    const int* node_ids,
    const size_t num_nodes,
    const float* node_features,
    const size_t feature_dim) const {
    node_features::set_node_features(
        this->node_features,
        max_node_id,
        node_ids,
        num_nodes,
        node_features,
        feature_dim);
}

size_t NodeFeatures::node_feature_dim() const {
    return node_features->node_feature_dim;
}

int NodeFeatures::max_node_id() const {
    return node_features->max_node_id;
}


NodeFeaturesStore* NodeFeatures::get_node_features() const {
    return node_features;
}
