#include "TemporalRandomWalk.cuh"

TemporalRandomWalk::TemporalRandomWalk(
    const bool is_directed, const bool use_gpu,
    const int64_t max_time_capacity,
    const bool enable_weight_computation,
    const bool enable_temporal_node2vec,
    const double timescale_bound,
    const double node2vec_p, const double node2vec_q,
    const int walk_padding_value,
    const uint64_t global_seed,
    const bool shuffle_walk_order)
    : impl_(std::make_unique<core::TemporalRandomWalk>(
        is_directed, use_gpu, max_time_capacity,
        enable_weight_computation, enable_temporal_node2vec,
        timescale_bound, node2vec_p, node2vec_q,
        walk_padding_value, global_seed, shuffle_walk_order)) {}

TemporalRandomWalk::~TemporalRandomWalk() = default;

void TemporalRandomWalk::add_multiple_edges(
    const int* sources, const int* targets, const int64_t* timestamps,
    const size_t edges_size, const float* edge_features,
    const size_t feature_dim, const size_t block_dim) const {
    impl_->add_multiple_edges(
        sources, targets, timestamps, edges_size,
        edge_features, feature_dim, block_dim);
}

void TemporalRandomWalk::add_multiple_edges(
    const std::vector<std::tuple<int, int, int64_t>>& edges,
    const float* edge_features, const size_t feature_dim,
    const size_t block_dim) const {
    impl_->add_multiple_edges(edges, edge_features, feature_dim, block_dim);
}

WalksWithEdgeFeaturesHost
TemporalRandomWalk::get_random_walks_and_times_for_all_nodes(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) const {
    return impl_->get_random_walks_and_times_for_all_nodes(
        max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction, kernel_launch_type,
        block_dim, w_threshold_warp);
}

WalksWithEdgeFeaturesHost
TemporalRandomWalk::get_random_walks_and_times_for_last_batch(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) const {
    return impl_->get_random_walks_and_times_for_last_batch(
        max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction, kernel_launch_type,
        block_dim, w_threshold_warp);
}

WalksWithEdgeFeaturesHost
TemporalRandomWalk::get_random_walks_and_times(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) const {
    return impl_->get_random_walks_and_times(
        max_walk_len, walk_bias, num_walks_total,
        initial_edge_bias, walk_direction, kernel_launch_type,
        block_dim, w_threshold_warp);
}

void TemporalRandomWalk::set_node_features(
    const int* node_ids, const size_t num_nodes,
    const float* node_features_data, const size_t feature_dim) const {
    impl_->set_node_features(node_ids, num_nodes, node_features_data, feature_dim);
}

size_t TemporalRandomWalk::get_node_count() const { return impl_->get_node_count(); }
size_t TemporalRandomWalk::get_edge_count() const { return impl_->get_edge_count(); }
std::vector<int> TemporalRandomWalk::get_node_ids() const { return impl_->get_node_ids(); }

std::vector<std::tuple<int, int, int64_t>> TemporalRandomWalk::get_edges() const {
    const auto edges = impl_->get_edges();
    std::vector<std::tuple<int, int, int64_t>> out;
    out.reserve(edges.size());
    for (const auto& e : edges) out.emplace_back(e.u, e.i, e.ts);
    return out;
}

bool TemporalRandomWalk::get_is_directed() const { return impl_->get_is_directed(); }
void TemporalRandomWalk::clear() const { impl_->clear(); }
size_t TemporalRandomWalk::get_memory_used() const { return impl_->get_memory_used(); }

int TemporalRandomWalk::node_feature_dim() const {
    return static_cast<int>(impl_->data().node_feature_dim);
}

int TemporalRandomWalk::node_features_max_node_id() const {
    return impl_->data().max_node_id;
}

std::vector<float> TemporalRandomWalk::node_features_dense() const {
    return impl_->data().node_features.to_host_vector();
}
