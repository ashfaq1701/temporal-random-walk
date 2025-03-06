#include "TemporalRandomWalk.cuh"

#include <iostream>

#include "TemporalRandomWalkCPU.cuh"
#include "TemporalRandomWalkCUDA.cuh"

template <GPUUsageMode GPUUsage>
TemporalRandomWalk<GPUUsage>::TemporalRandomWalk(bool is_directed, int64_t max_time_capacity, bool enable_weight_computation, double timescale_bound, size_t n_threads)
{
    if (GPUUsage == GPUUsageMode::ON_CPU) {
        temporal_random_walk = new TemporalRandomWalkCPU<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound, n_threads);
    }
    #ifdef HAS_CUDA
    else {
        temporal_random_walk = new TemporalRandomWalkCUDA<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
    }
    #else
    else {
        // Fallback to CPU implementation when CUDA is not available
        std::cerr << "Warning: CUDA implementation requested but not available, using CPU implementation instead" << std::endl;
        temporal_random_walk = new TemporalRandomWalkCPU<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound, n_threads);
    }
    #endif
}

template <GPUUsageMode GPUUsage>
std::vector<std::vector<NodeWithTime>> TemporalRandomWalk<GPUUsage>::get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction)
{
    WalkSet<GPUUsage> walk_set = temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        walk_bias,
        num_walks_per_node,
        initial_edge_bias,
        walk_direction
    );

    std::vector<std::vector<NodeWithTime>> walks(walk_set.num_walks);

    for (size_t i = 0; i < walk_set.num_walks; ++i) {
        const size_t walk_len = walk_set.get_walk_len(i);

        walks[i].resize(walk_len);

        for (size_t j = 0; j < walk_len; ++j) {
            walks[i][j] = walk_set.get_walk_hop(i, j);
        }
    }

    std::vector<std::vector<NodeWithTime>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<NodeWithTime>& v) { return !v.empty(); });

    return non_empty_walks;
}

template <GPUUsageMode GPUUsage>
std::vector<std::vector<int>> TemporalRandomWalk<GPUUsage>::get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction)
{
    WalkSet<GPUUsage> walk_set = temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        walk_bias,
        num_walks_per_node,
        initial_edge_bias,
        walk_direction
    );

    std::vector<std::vector<int>> walks(walk_set.num_walks);

    for (size_t i = 0; i < walk_set.num_walks; ++i) {
        const size_t walk_len = walk_set.walk_lens[i];
        walks[i].resize(walk_len);

        for (size_t j = 0; j < walk_len; ++j) {
            const NodeWithTime hop = walk_set.get_walk_hop(i, j);
            walks[i][j] = hop.node;
        }
    }

    std::vector<std::vector<int>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<int>& v) { return !v.empty(); });

    return non_empty_walks;
}

template <GPUUsageMode GPUUsage>
std::vector<std::vector<NodeWithTime>> TemporalRandomWalk<GPUUsage>::get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction)
{
    WalkSet<GPUUsage> walk_set = temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        walk_bias,
        num_walks_total,
        initial_edge_bias,
        walk_direction
    );

    std::vector<std::vector<NodeWithTime>> walks(walk_set.num_walks);

    for (size_t i = 0; i < walk_set.num_walks; ++i) {
        const size_t walk_len = walk_set.walk_lens[i];
        walks[i].resize(walk_len);

        for (size_t j = 0; j < walk_len; ++j) {
            walks[i][j] = walk_set.get_walk_hop(i, j);
        }
    }

    std::vector<std::vector<NodeWithTime>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<NodeWithTime>& v) { return !v.empty(); });

    return non_empty_walks;
}

template <GPUUsageMode GPUUsage>
std::vector<std::vector<int>> TemporalRandomWalk<GPUUsage>::get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction)
{
    WalkSet<GPUUsage> walk_set = temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        walk_bias,
        num_walks_total,
        initial_edge_bias,
        walk_direction
    );

    std::vector<std::vector<int>> walks(walk_set.num_walks);

    for (size_t i = 0; i < walk_set.num_walks; ++i) {
        const size_t walk_len = walk_set.walk_lens[i];
        walks[i].resize(walk_len);

        for (size_t j = 0; j < walk_len; ++j) {
            const NodeWithTime hop = walk_set.get_walk_hop(i, j);
            walks[i][j] = hop.node;
        }
    }

    std::vector<std::vector<int>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<int>& v) { return !v.empty(); });

    return non_empty_walks;
}

template <GPUUsageMode GPUUsage>
void TemporalRandomWalk<GPUUsage>::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edge_infos) const
{
    typename ITemporalRandomWalk<GPUUsage>::EdgeVector edges;
    edges.reserve(edge_infos.size());

    for (auto [u, i, ts] : edge_infos) {
        edges.push_back(Edge(u, i, ts));
    }

    temporal_random_walk->add_multiple_edges(edges);
}

template <GPUUsageMode GPUUsage>
size_t TemporalRandomWalk<GPUUsage>::get_node_count() const
{
    return temporal_random_walk->get_node_count();
}

template <GPUUsageMode GPUUsage>
size_t TemporalRandomWalk<GPUUsage>::get_edge_count() const
{
    return temporal_random_walk->get_edge_count();
}

template <GPUUsageMode GPUUsage>
std::vector<int> TemporalRandomWalk<GPUUsage>::get_node_ids() const
{
    auto nodes = temporal_random_walk->get_node_ids();

    std::vector<int> result(nodes.size());
    for (auto node : nodes)
    {
        result.push_back(node);
    }

    return result;
}

template <GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> TemporalRandomWalk<GPUUsage>::get_edges() const
{
    auto edges = temporal_random_walk->get_edges();
    std::vector<std::tuple<int, int, int64_t>> result;
    result.reserve(edges.size());

    #ifdef HAS_CUDA
    if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
        std::vector<Edge> host_edges(edges.size());
        thrust::copy(edges.begin(), edges.end(), host_edges.begin());

        for (const auto& edge : host_edges) {
            result.push_back({edge.u, edge.i, edge.ts});
        }
    }
    else
    #endif
    {
        for (const auto& edge : edges) {
            result.push_back({edge.u, edge.i, edge.ts});
        }
    }

    return result;
}

template <GPUUsageMode GPUUsage>
bool TemporalRandomWalk<GPUUsage>::get_is_directed() const
{
    return temporal_random_walk->get_is_directed();
}

template <GPUUsageMode GPUUsage>
void TemporalRandomWalk<GPUUsage>::clear()
{
    temporal_random_walk->clear();
}

template class TemporalRandomWalk<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalRandomWalk<GPUUsageMode::ON_GPU>;
#endif
