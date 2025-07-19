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
        const int walk_padding_value): use_gpu(use_gpu) {
    temporal_random_walk = new TemporalRandomWalkStore(
        is_directed,
        use_gpu,
        max_time_capacity,
        enable_weight_computation,
        timescale_bound,
        walk_padding_value);
}

TemporalRandomWalk::~TemporalRandomWalk() {
    delete temporal_random_walk;
}

void TemporalRandomWalk::add_multiple_edges(
    const int* sources,
    const int* targets,
    const int64_t* timestamps,
    const size_t edges_size) const {
    temporal_random_walk::add_multiple_edges(
        temporal_random_walk,
        sources,
        targets,
        timestamps,
        edges_size);
}

void TemporalRandomWalk::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edges) const {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    sources.reserve(edges.size());
    targets.reserve(edges.size());
    timestamps.reserve(edges.size());

    for (const auto& edge : edges) {
        sources.push_back(std::get<0>(edge));
        targets.push_back(std::get<1>(edge));
        timestamps.push_back(std::get<2>(edge));
    }

    add_multiple_edges(sources.data(), targets.data(), timestamps.data(), timestamps.size());
}

WalkSet TemporalRandomWalk::get_random_walks_and_times_for_all_nodes(
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

    return walk_set;
}

WalkSet TemporalRandomWalk::get_random_walks_and_times(
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

    return walk_set;
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
        temporal_random_walk::free_device_pointers(d_temporal_random_walk);

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

std::vector<std::tuple<int, int, int64_t>> TemporalRandomWalk::get_edges() const {
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
                host_edges[i].u,
                host_edges[i].i,
                host_edges[i].ts);
        }

        delete[] host_edges;
    }
    else
    #endif
    {
        for (size_t i = 0; i < edges.size; i++) {
            result.emplace_back(
                edges.data[i].u,
                edges.data[i].i,
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

size_t TemporalRandomWalk::get_memory_used() const {
    return temporal_random_walk::get_memory_used(temporal_random_walk);
}
