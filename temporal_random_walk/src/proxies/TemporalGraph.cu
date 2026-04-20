#include "TemporalGraph.cuh"

#include <algorithm>

#include "TemporalRandomWalk.cuh"
#include "../common/error_handlers.cuh"
#include "../common/random_gen.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../graph/temporal_node2vec_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace {

template <bool Forward, RandomPickerType PickerType>
__global__ void get_edge_at_kernel_v(
    Edge* result, TemporalGraphView view, const int64_t timestamp, const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_edge_at_device<Forward, PickerType>(
            view, timestamp, rand_nums[0], rand_nums[1]);
    }
}

template <bool IsDirected, bool Forward, RandomPickerType PickerType>
__global__ void get_node_edge_at_kernel_v(
    Edge* result, TemporalGraphView view, const int node_id,
    const int64_t timestamp, const int prev_node, const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_node_edge_at_device<Forward, PickerType, IsDirected>(
            view, node_id, timestamp, prev_node, rand_nums[0], rand_nums[1]);
    }
}

__global__ void compute_node2vec_beta_kernel_v(
    double* result, TemporalGraphView view, const int prev_node, const int w) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::compute_node2vec_beta_device(view, prev_node, w);
    }
}

} // namespace
#endif

TemporalGraph::TemporalGraph(
    const bool is_directed,
    const bool use_gpu,
    const int64_t max_time_capacity,
    const bool enable_weight_computation,
    const bool enable_temporal_node2vec,
    const double timescale_bound,
    const double node2vec_p,
    const double node2vec_q)
    : self_owned_(std::make_unique<TemporalRandomWalk>(
          is_directed, use_gpu, max_time_capacity,
          enable_weight_computation, enable_temporal_node2vec,
          timescale_bound, node2vec_p, node2vec_q)),
      graph(&self_owned_->impl()->data()) {}

TemporalGraph::~TemporalGraph() = default;

void TemporalGraph::add_multiple_edges(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<int64_t>& timestamps,
    const float* edge_features,
    const size_t feature_dim) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            *graph, sources.data(), targets.data(), timestamps.data(),
            timestamps.size(), edge_features, feature_dim);
        return;
    }
#endif
    temporal_graph::add_multiple_edges_std(
        *graph, sources.data(), targets.data(), timestamps.data(),
        timestamps.size(), edge_features, feature_dim);
}

void TemporalGraph::add_multiple_edges(const std::vector<Edge>& edges) const {
    std::vector<int> sources; sources.reserve(edges.size());
    std::vector<int> targets; targets.reserve(edges.size());
    std::vector<int64_t> timestamps; timestamps.reserve(edges.size());
    for (const auto& edge : edges) {
        sources.push_back(edge.u);
        targets.push_back(edge.i);
        timestamps.push_back(edge.ts);
    }
    add_multiple_edges(sources, targets, timestamps, nullptr, 0);
}

void TemporalGraph::sort_and_merge_edges(const size_t start_idx) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) {
        temporal_graph::sort_and_merge_edges_cuda(*graph, start_idx);
        return;
    }
#endif
    temporal_graph::sort_and_merge_edges_std(*graph, start_idx);
}

void TemporalGraph::delete_old_edges() const {
#ifdef HAS_CUDA
    if (graph->use_gpu) {
        temporal_graph::delete_old_edges_cuda(*graph);
        return;
    }
#endif
    temporal_graph::delete_old_edges_std(*graph);
}

size_t TemporalGraph::count_timestamps_less_than(const int64_t timestamp) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) return temporal_graph::count_timestamps_less_than_cuda(*graph, timestamp);
#endif
    return temporal_graph::count_timestamps_less_than_std(*graph, timestamp);
}

size_t TemporalGraph::count_timestamps_greater_than(const int64_t timestamp) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) return temporal_graph::count_timestamps_greater_than_cuda(*graph, timestamp);
#endif
    return temporal_graph::count_timestamps_greater_than_std(*graph, timestamp);
}

size_t TemporalGraph::count_node_timestamps_less_than(const int node_id, const int64_t timestamp) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) return temporal_graph::count_node_timestamps_less_than_cuda(*graph, node_id, timestamp);
#endif
    return temporal_graph::count_node_timestamps_less_than_std(*graph, node_id, timestamp);
}

size_t TemporalGraph::count_node_timestamps_greater_than(const int node_id, const int64_t timestamp) const {
#ifdef HAS_CUDA
    if (graph->use_gpu) return temporal_graph::count_node_timestamps_greater_than_cuda(*graph, node_id, timestamp);
#endif
    return temporal_graph::count_node_timestamps_greater_than_std(*graph, node_id, timestamp);
}

double TemporalGraph::compute_node2vec_beta(const int prev_node, const int w) const {
    const TemporalGraphView view = make_temporal_graph_view(*graph);
#ifdef HAS_CUDA
    if (graph->use_gpu) {
        double* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(double)));
        compute_node2vec_beta_kernel_v<<<1, 1>>>(d_result, view, prev_node, w);
        CUDA_KERNEL_CHECK("compute_node2vec_beta_kernel_v");
        double result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        return result;
    }
#endif
    return temporal_graph::compute_node2vec_beta_host(view, prev_node, w);
}

Edge TemporalGraph::get_edge_at_with_provided_nums(
    const RandomPickerType picker_type, const double* rand_nums,
    const int64_t timestamp, const bool forward) const {
    const TemporalGraphView view = make_temporal_graph_view(*graph);
    Edge result{};

    #define DISPATCH_HOST(FWD, PICKER) \
        result = temporal_graph::get_edge_at_host<FWD, PICKER>( \
            view, timestamp, rand_nums[0], rand_nums[1]); break;

#ifdef HAS_CUDA
    #define DISPATCH_DEVICE(FWD, PICKER) \
        get_edge_at_kernel_v<FWD, PICKER><<<1, 1>>>( \
            d_result, view, timestamp, rand_nums); \
        CUDA_KERNEL_CHECK("After get_edge_at_kernel_v execution"); break;
#endif

    #define HANDLE_PICKER_HOST(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_HOST(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear:            DISPATCH_HOST(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST:         DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

#ifdef HAS_CUDA
    #define HANDLE_PICKER_DEVICE(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_DEVICE(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear:            DISPATCH_DEVICE(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST:         DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

    if (graph->use_gpu) {
        Edge* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));
        if (forward) { HANDLE_PICKER_DEVICE(true) } else { HANDLE_PICKER_DEVICE(false) }
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    } else
#endif
    {
        if (forward) { HANDLE_PICKER_HOST(true) } else { HANDLE_PICKER_HOST(false) }
    }

    #undef DISPATCH_HOST
    #undef HANDLE_PICKER_HOST
#ifdef HAS_CUDA
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_DEVICE
#endif

    return result;
}

Edge TemporalGraph::get_edge_at(
    const RandomPickerType picker_type, const int64_t timestamp, const bool forward) const {
    double* rand_nums = generate_n_random_numbers(2, graph->use_gpu);
    Edge result = get_edge_at_with_provided_nums(picker_type, rand_nums, timestamp, forward);
    clear_memory(&rand_nums, graph->use_gpu);
    return result;
}

Edge TemporalGraph::get_node_edge_at(
    const int node_id, const RandomPickerType picker_type,
    const int64_t timestamp, const int prev_node, const bool forward) const {
    const TemporalGraphView view = make_temporal_graph_view(*graph);
    Edge result{};
    const bool is_directed = graph->is_directed;

    #define DISPATCH_HOST(FWD, PICKER, DIR) \
        result = temporal_graph::get_node_edge_at_host<FWD, PICKER, DIR>( \
            view, node_id, timestamp, prev_node, rand_nums[0], rand_nums[1]); break;

#ifdef HAS_CUDA
    #define DISPATCH_DEVICE(FWD, PICKER, DIR) \
        get_node_edge_at_kernel_v<DIR, FWD, PICKER><<<1, 1>>>( \
            d_result, view, node_id, timestamp, prev_node, rand_nums); \
        CUDA_KERNEL_CHECK("After get_node_edge_at_kernel_v execution"); break;
#endif

    #define HANDLE_PICKER_HOST(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_HOST(FWD, RandomPickerType::Uniform,           DIR) \
            case RandomPickerType::Linear:            DISPATCH_HOST(FWD, RandomPickerType::Linear,            DIR) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex,  DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec,  DIR) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST,        DIR) \
            case RandomPickerType::TEST_LAST:         DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST,         DIR) \
            default: break; \
        }

#ifdef HAS_CUDA
    #define HANDLE_PICKER_DEVICE(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_DEVICE(FWD, RandomPickerType::Uniform,           DIR) \
            case RandomPickerType::Linear:            DISPATCH_DEVICE(FWD, RandomPickerType::Linear,            DIR) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex,  DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec,  DIR) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST,        DIR) \
            case RandomPickerType::TEST_LAST:         DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST,         DIR) \
            default: break; \
        }
#endif

    double* rand_nums = generate_n_random_numbers(2, graph->use_gpu);

#ifdef HAS_CUDA
    if (graph->use_gpu) {
        Edge* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));
        if (is_directed) {
            if (forward) { HANDLE_PICKER_DEVICE(true,  true)  } else { HANDLE_PICKER_DEVICE(false, true)  }
        } else {
            if (forward) { HANDLE_PICKER_DEVICE(true,  false) } else { HANDLE_PICKER_DEVICE(false, false) }
        }
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    } else
#endif
    {
        if (is_directed) {
            if (forward) { HANDLE_PICKER_HOST(true,  true)  } else { HANDLE_PICKER_HOST(false, true)  }
        } else {
            if (forward) { HANDLE_PICKER_HOST(true,  false) } else { HANDLE_PICKER_HOST(false, false) }
        }
    }

    clear_memory(&rand_nums, graph->use_gpu);

    #undef DISPATCH_HOST
    #undef HANDLE_PICKER_HOST
#ifdef HAS_CUDA
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_DEVICE
#endif

    return result;
}
