#include "TemporalRandomWalk.cuh"

#include <algorithm>
#include <iterator>

#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_count_kernel(size_t* result, const TemporalRandomWalkStore* temporal_random_walk) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_random_walk::get_edge_count(temporal_random_walk);
    }
}

#endif

TemporalRandomWalk::TemporalRandomWalk(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const size_t n_threads): use_gpu(use_gpu) {
    temporal_random_walk = new TemporalRandomWalkStore(
        is_directed,
        use_gpu,
        max_time_capacity,
        enable_weight_computation,
        timescale_bound,
        n_threads);
}

TemporalRandomWalk::~TemporalRandomWalk() {
    delete temporal_random_walk;
}

void TemporalRandomWalk::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edges) {
    const auto edge_array = new Edge[edges.size()];
    for (size_t idx = 0; idx < edges.size(); idx++) {
        const auto& [u, i, ts] = edges[idx];

        if (node_index.find(u) == node_index.end()) {
            node_index[u] = ++running_node_id;
            reverse_node_index[running_node_id] = u;
        }

        if (node_index.find(i) == node_index.end()) {
            node_index[i] = ++running_node_id;
            reverse_node_index[running_node_id] = i;
        }

        edge_array[idx] = Edge(node_index[u], node_index[i], ts);
    }

    temporal_random_walk::add_multiple_edges(temporal_random_walk, edge_array, edges.size(), running_node_id);

    delete[] edge_array;
}

std::vector<std::vector<NodeWithTime>> TemporalRandomWalk::get_random_walks_and_times_for_all_nodes(
        const int max_walk_len,
        const RandomPickerType* walk_bias,
        const int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        const WalkDirection walk_direction) const {

    WalkSet walk_set;

    #ifdef HAS_CUDA
    if (use_gpu) {
        walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
            temporal_random_walk,
            max_walk_len,
            walk_bias,
            num_walks_per_node,
            initial_edge_bias,
            walk_direction);
    }
    else
    #endif
    {
        walk_set = temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
            temporal_random_walk,
            max_walk_len,
            walk_bias,
            num_walks_per_node,
            initial_edge_bias,
            walk_direction);
    }

    std::vector<std::vector<NodeWithTime>> walks(walk_set.num_walks);
    for (size_t walk_idx = 0; walk_idx < walk_set.num_walks; walk_idx++) {
        const size_t walk_len = walk_set.get_walk_len(static_cast<int>(walk_idx));

        walks[walk_idx].reserve(walk_len);

        for (size_t hop = 0; hop < walk_len; hop++) {
            NodeWithTime node_time = walk_set.get_walk_hop(
                static_cast<int>(walk_idx),
                static_cast<int>(hop),
                &reverse_node_index);

            walks[walk_idx].push_back(node_time);
        }
    }

    std::vector<std::vector<NodeWithTime>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<NodeWithTime>& v) { return !v.empty(); });

    return non_empty_walks;
}

std::vector<std::vector<int>> TemporalRandomWalk::get_random_walks_for_all_nodes(
        const int max_walk_len,
        const RandomPickerType* walk_bias,
        const int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        const WalkDirection walk_direction) const {

    auto walks_with_times = get_random_walks_and_times_for_all_nodes(
        max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);

    std::vector<std::vector<int>> result(walks_with_times.size());
    for (size_t i = 0; i < walks_with_times.size(); i++) {
        result[i].reserve(walks_with_times[i].size());
        for (const auto& node_time : walks_with_times[i]) {
            result[i].push_back(node_time.node);
        }
    }

    return result;
}

std::vector<std::vector<NodeWithTime>> TemporalRandomWalk::get_random_walks_and_times(
        const int max_walk_len,
        const RandomPickerType* walk_bias,
        const int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        const WalkDirection walk_direction) const {

    WalkSet walk_set;

    #ifdef HAS_CUDA
    if (use_gpu) {
        walk_set = temporal_random_walk::get_random_walks_and_times_cuda(
            temporal_random_walk,
            max_walk_len,
            walk_bias,
            num_walks_total,
            initial_edge_bias,
            walk_direction);
    }
    else
    #endif
    {
        walk_set = temporal_random_walk::get_random_walks_and_times_std(
            temporal_random_walk,
            max_walk_len,
            walk_bias,
            num_walks_total,
            initial_edge_bias,
            walk_direction);
    }

    std::vector<std::vector<NodeWithTime>> walks(walk_set.num_walks);
    for (size_t walk_idx = 0; walk_idx < walk_set.num_walks; walk_idx++) {
        const size_t walk_len = walk_set.get_walk_len(static_cast<int>(walk_idx));

        walks[walk_idx].reserve(walk_len);

        for (size_t hop = 0; hop < walk_len; hop++) {
            NodeWithTime node_time = walk_set.get_walk_hop(
                static_cast<int>(walk_idx),
                static_cast<int>(hop),
                &reverse_node_index);

            walks[walk_idx].push_back(node_time);
        }
    }

    std::vector<std::vector<NodeWithTime>> non_empty_walks;
    std::copy_if(walks.begin(), walks.end(), std::back_inserter(non_empty_walks),
                 [](const std::vector<NodeWithTime>& v) { return !v.empty(); });

    return non_empty_walks;
}

std::vector<std::vector<int>> TemporalRandomWalk::get_random_walks(
        const int max_walk_len,
        const RandomPickerType* walk_bias,
        const int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        const WalkDirection walk_direction) const {

    auto walks_with_times = get_random_walks_and_times(
        max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);

    std::vector<std::vector<int>> result(walks_with_times.size());
    for (size_t i = 0; i < walks_with_times.size(); i++) {
        result[i].reserve(walks_with_times[i].size());
        for (const auto& node_time : walks_with_times[i]) {
            result[i].push_back(node_time.node);
        }
    }

    return result;
}

size_t TemporalRandomWalk::get_node_count() const {
    return temporal_random_walk::get_node_count(temporal_random_walk);
}

size_t TemporalRandomWalk::get_edge_count() const {
    #ifdef HAS_CUDA
    if (use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(size_t)));

        TemporalRandomWalkStore* d_temporal_random_walk = temporal_random_walk::to_device_ptr(temporal_random_walk);
        get_edge_count_kernel<<<1, 1>>>(d_result, d_temporal_random_walk);
        CUDA_KERNEL_CHECK("After get_edge_count_kernel execution");

        size_t host_result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_temporal_random_walk));

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return temporal_random_walk::get_edge_count(temporal_random_walk);
    }
}

std::vector<int> TemporalRandomWalk::get_node_ids() const {
    const DataBlock<int> node_ids = temporal_random_walk::get_node_ids(temporal_random_walk);
    std::vector<int> result;

    #ifdef HAS_CUDA
    if (node_ids.use_gpu) {
        // Allocate temporary host memory
        int* host_data = new int[node_ids.size];
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(host_data, node_ids.data,
                                     node_ids.size * sizeof(int),
                                     cudaMemcpyDeviceToHost));

        result.assign(host_data, host_data + node_ids.size);

        delete[] host_data;
    }
    else
    #endif
    {
        result.assign(node_ids.data, node_ids.data + node_ids.size);
    }

    return result;
}

std::vector<std::tuple<int, int, int64_t>> TemporalRandomWalk::get_edges() {
    const DataBlock<Edge> edges = temporal_random_walk::get_edges(temporal_random_walk);
    std::vector<std::tuple<int, int, int64_t>> result;
    result.reserve(edges.size);

    #ifdef HAS_CUDA
    if (edges.use_gpu) {
        auto host_edges = new Edge[edges.size];
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(host_edges, edges.data,
                                    edges.size * sizeof(Edge),
                                    cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < edges.size; i++) {
            result.emplace_back(
                reverse_node_index[host_edges[i].u],
                reverse_node_index[host_edges[i].i],
                host_edges[i].ts);
        }

        delete[] host_edges;
    }
    else
    #endif
    {
        for (size_t i = 0; i < edges.size; i++) {
            result.emplace_back(
                reverse_node_index[edges.data[i].u],
                reverse_node_index[edges.data[i].i],
                edges.data[i].ts);
        }
    }

    return result;
}

bool TemporalRandomWalk::get_is_directed() const {
    return temporal_random_walk::get_is_directed(temporal_random_walk);
}

void TemporalRandomWalk::clear() const {
    temporal_random_walk::clear(temporal_random_walk);
}
