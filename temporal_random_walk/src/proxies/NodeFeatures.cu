#include "NodeFeatures.cuh"

NodeFeatures::NodeFeatures() {
    node_features = new NodeFeaturesStore();
    owns_node_features = true;
}

NodeFeatures::NodeFeatures(NodeFeaturesStore* existing_node_features) {
    node_features = existing_node_features;
    owns_node_features = false;
}

NodeFeatures::~NodeFeatures() {
    if (owns_node_features && node_features) {
        delete node_features;
    }
}

NodeFeatures& NodeFeatures::operator=(const NodeFeatures& other) {
    if (this != &other) {
        if (owns_node_features && node_features) {
            delete node_features;
        }

        node_features = other.node_features;
        owns_node_features = false;
    }
    return *this;
}

void NodeFeatures::ensure_size(const int max_node_id) const {
    node_features::ensure_size(node_features, max_node_id);
}

void NodeFeatures::set_node_features(
    const int* node_ids,
    const size_t num_nodes,
    const float* node_features,
    const size_t feature_dim) const {
    node_features::set_node_features(this->node_features, node_ids, num_nodes, node_features, feature_dim);
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
