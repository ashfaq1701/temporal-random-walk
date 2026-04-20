#include "NodeFeatures.cuh"

#include "TemporalRandomWalk.cuh"

NodeFeatures::NodeFeatures()
    : self_owned_(std::make_unique<TemporalRandomWalk>(
          /*is_directed=*/true, /*use_gpu=*/false)),
      data(&self_owned_->impl()->data()) {}

NodeFeatures::~NodeFeatures() = default;

void NodeFeatures::set_node_features(
    const int max_node_id,
    const int* node_ids,
    const size_t num_nodes,
    const float* node_features_data,
    const size_t feature_dim) const {
    node_features::set_node_features(
        *data,
        max_node_id,
        node_ids,
        num_nodes,
        node_features_data,
        feature_dim);
}
