#include "TemporalGraph.cuh"

#include "../common/random_gen.cuh"
#include "../common/setup.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
__global__ void get_total_edges_kernel(size_t* result, const TemporalGraphStore* graph) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_total_edges(graph);
    }
}
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
    : owns_graph(true) {

    graph = new TemporalGraphStore(
        is_directed,
        use_gpu,
        max_time_capacity,
        enable_weight_computation,
        enable_temporal_node2vec,
        timescale_bound,
        node2vec_p,
        node2vec_q);
}

TemporalGraph::TemporalGraph(TemporalGraphStore* existing_graph)
    : graph(existing_graph), owns_graph(false) {}

TemporalGraph::~TemporalGraph() {
    if (owns_graph && graph) {
        delete graph;
    }
}

TemporalGraph& TemporalGraph::operator=(const TemporalGraph& other) {
    if (this != &other) {
        if (owns_graph && graph) {
            delete graph;
        }

        owns_graph = other.owns_graph;
        if (other.owns_graph) {
            graph = new TemporalGraphStore(
                other.graph->is_directed,
                other.graph->use_gpu,
                other.graph->max_time_capacity,
                other.graph->enable_weight_computation,
                other.graph->enable_temporal_node2vec,
                other.graph->timescale_bound,
                other.graph->node2vec_p,
                other.graph->node2vec_q);
        } else {
            graph = other.graph;
        }
    }
    return *this;
}

void TemporalGraph::update_temporal_weights() const {
    temporal_graph::update_temporal_weights(graph);
}

size_t TemporalGraph::get_total_edges() const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(size_t)));

        TemporalGraphStore* d_graph = temporal_graph::to_device_ptr(graph);
        get_total_edges_kernel<<<1, 1>>>(d_result, d_graph);
        CUDA_KERNEL_CHECK("After get_total_edges_kernel execution");

        size_t host_result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        temporal_graph::free_device_pointers(d_graph);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return temporal_graph::get_total_edges(graph);
    }
}

size_t TemporalGraph::get_node_count() const {
    return temporal_graph::get_node_count(graph);
}

int64_t TemporalGraph::get_latest_timestamp() const {
    return temporal_graph::get_latest_timestamp(graph);
}

std::vector<int> TemporalGraph::get_node_ids() const {
    DataBlock<int> node_ids = temporal_graph::get_node_ids(graph);
    std::vector<int> result;

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[node_ids.size];
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(host_ids, node_ids.data, node_ids.size * sizeof(int), cudaMemcpyDeviceToHost));

        result.assign(host_ids, host_ids + node_ids.size);
        delete[] host_ids;
    }
    else
    #endif
    {
        // For CPU data, can directly copy
        result.assign(node_ids.data, node_ids.data + node_ids.size);
    }

    return result;
}

std::vector<Edge> TemporalGraph::get_edges() const {
    DataBlock<Edge> edges = temporal_graph::get_edges(graph);
    std::vector<Edge> result;

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_edges = new Edge[edges.size];
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(host_edges, edges.data, edges.size * sizeof(Edge), cudaMemcpyDeviceToHost));

        result.assign(host_edges, host_edges + edges.size);
        delete[] host_edges;
    }
    else
    #endif
    {
        // For CPU data, can directly copy
        result.assign(edges.data, edges.data + edges.size);
    }

    return result;
}

void TemporalGraph::add_multiple_edges(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<int64_t>& timestamps) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Call CUDA implementation
        temporal_graph::add_multiple_edges_cuda(
            graph,
            sources.data(),
            targets.data(),
            timestamps.data(),
            timestamps.size());
    }
    else
    #endif
    {
        // Call CPU implementation directly
        temporal_graph::add_multiple_edges_std(
            graph,
            sources.data(),
            targets.data(),
            timestamps.data(),
            timestamps.size());
    }
}

void TemporalGraph::add_multiple_edges(const std::vector<Edge>& edges) const {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    sources.reserve(edges.size());
    targets.reserve(edges.size());
    timestamps.reserve(edges.size());

    for (const auto& edge : edges) {
        sources.push_back(edge.u);
        targets.push_back(edge.i);
        timestamps.push_back(edge.ts);
    }

    add_multiple_edges(sources, targets, timestamps);
}

void TemporalGraph::sort_and_merge_edges(size_t start_idx) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        temporal_graph::sort_and_merge_edges_cuda(graph, start_idx);
    }
    else
    #endif
    {
        temporal_graph::sort_and_merge_edges_std(graph, start_idx);
    }
}

void TemporalGraph::delete_old_edges() const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        temporal_graph::delete_old_edges_cuda(graph);
    }
    else
    #endif
    {
        temporal_graph::delete_old_edges_std(graph);
    }
}

size_t TemporalGraph::count_timestamps_less_than(int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        return temporal_graph::count_timestamps_less_than_cuda(graph, timestamp);
    }
    else
    #endif
    {
        return temporal_graph::count_timestamps_less_than_std(graph, timestamp);
    }
}

size_t TemporalGraph::count_timestamps_greater_than(int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        return temporal_graph::count_timestamps_greater_than_cuda(graph, timestamp);
    }
    else
    #endif
    {
        return temporal_graph::count_timestamps_greater_than_std(graph, timestamp);
    }
}

size_t TemporalGraph::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        return temporal_graph::count_node_timestamps_less_than_cuda(graph, node_id, timestamp);
    }
    else
    #endif
    {
        return temporal_graph::count_node_timestamps_less_than_std(graph, node_id, timestamp);
    }
}

size_t TemporalGraph::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        return temporal_graph::count_node_timestamps_greater_than_cuda(graph, node_id, timestamp);
    }
    else
    #endif
    {
        return temporal_graph::count_node_timestamps_greater_than_std(graph, node_id, timestamp);
    }
}

[[nodiscard]] double TemporalGraph::compute_node2vec_beta(
    const int prev_node,
    const int w) const {
    const auto* graph = this->graph;   // TemporalGraphStore*
    double result = 0.0;

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        double* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(double)));

        TemporalGraphStore* d_graph = temporal_graph::to_device_ptr(graph);

        compute_node2vec_beta_kernel<<<1,1>>>(
            d_result, d_graph, prev_node, w);
        CUDA_KERNEL_CHECK("compute_node2vec_beta_kernel");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            &result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        temporal_graph::free_device_pointers(d_graph);
    }
    else
    #endif
    {
        result = temporal_graph::compute_node2vec_beta_host(
            graph, prev_node, w);
    }

    return result;
}

[[nodiscard]] Edge TemporalGraph::get_edge_at_with_provided_nums(const RandomPickerType picker_type, const double * rand_nums, const int64_t timestamp, const bool forward) const {
    Edge result;

    #define DISPATCH_HOST(FWD, PICKER) \
        result = temporal_graph::get_edge_at_host<FWD, PICKER>( \
            graph, timestamp, rand_nums[0], rand_nums[1]); \
        break;

    #define DISPATCH_DEVICE(FWD, PICKER) \
        get_edge_at_kernel<FWD, PICKER><<<1, 1>>>( \
            d_result, d_graph, timestamp, rand_nums); \
        CUDA_KERNEL_CHECK("After get_edge_at_kernel execution"); \
        break;

    #define HANDLE_PICKER_HOST(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform: DISPATCH_HOST(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear: DISPATCH_HOST(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex: DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec: DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST: DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST: DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

    #define HANDLE_PICKER_DEVICE(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform: DISPATCH_DEVICE(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear: DISPATCH_DEVICE(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec: DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST: DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST: DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Allocate memory for the result
        Edge *d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));

        // Copy graph to device
        TemporalGraphStore *d_graph = temporal_graph::to_device_ptr(graph);

        // Dispatch to appropriate template specialization
        if (forward) {
            HANDLE_PICKER_DEVICE(true)
        } else {
            HANDLE_PICKER_DEVICE(false)
        }

        // Copy result back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));

        // Clean up
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        temporal_graph::free_device_pointers(d_graph);
    } else
    #endif
    {
        // Dispatch to appropriate host template specialization
        if (forward) {
            HANDLE_PICKER_HOST(true)
        } else {
            HANDLE_PICKER_HOST(false)
        }
    }

    #undef DISPATCH_HOST
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_HOST
    #undef HANDLE_PICKER_DEVICE

    return result;
}

[[nodiscard]] Edge TemporalGraph::get_edge_at(const RandomPickerType picker_type, const int64_t timestamp, const bool forward) const {
    double *rand_nums = generate_n_random_numbers(2, graph->use_gpu);
    auto result = get_edge_at_with_provided_nums(picker_type, rand_nums, timestamp, forward);
    clear_memory(&rand_nums, graph->use_gpu);
    return result;
}

[[nodiscard]] Edge TemporalGraph::get_node_edge_at(const int node_id, const RandomPickerType picker_type, const int64_t timestamp, const int prev_node, const bool forward) const {
    Edge result;
    const bool is_directed = graph->is_directed;

    #define DISPATCH_HOST(FWD, PICKER, DIR) \
        result = temporal_graph::get_node_edge_at_host<FWD, PICKER, DIR>( \
            graph, node_id, timestamp, prev_node, rand_nums[0], rand_nums[1]); \
        break;

    #define DISPATCH_DEVICE(FWD, PICKER, DIR) \
        get_node_edge_at_kernel<DIR, FWD, PICKER><<<1, 1>>>( \
            d_result, d_graph, node_id, timestamp, prev_node, rand_nums); \
        CUDA_KERNEL_CHECK("After get_node_edge_at_kernel execution"); \
        break;

    #define HANDLE_PICKER_HOST(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform: DISPATCH_HOST(FWD, RandomPickerType::Uniform, DIR) \
            case RandomPickerType::Linear: DISPATCH_HOST(FWD, RandomPickerType::Linear, DIR) \
            case RandomPickerType::ExponentialIndex: DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex, DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec: DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec, DIR) \
            case RandomPickerType::TEST_FIRST: DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST, DIR) \
            case RandomPickerType::TEST_LAST: DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST, DIR) \
            default: break; \
        }

    #define HANDLE_PICKER_DEVICE(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform: DISPATCH_DEVICE(FWD, RandomPickerType::Uniform, DIR) \
            case RandomPickerType::Linear: DISPATCH_DEVICE(FWD, RandomPickerType::Linear, DIR) \
            case RandomPickerType::ExponentialIndex: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex, DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec: DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec, DIR) \
            case RandomPickerType::TEST_FIRST: DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST, DIR) \
            case RandomPickerType::TEST_LAST: DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST, DIR) \
            default: break; \
        }

    double* rand_nums = generate_n_random_numbers(2, graph->use_gpu);

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        Edge* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));

        TemporalGraphStore* d_graph = temporal_graph::to_device_ptr(graph);

        if (is_directed) {
            if (forward) {
                HANDLE_PICKER_DEVICE(true, true)
            } else {
                HANDLE_PICKER_DEVICE(false, true)
            }
        } else {
            if (forward) {
                HANDLE_PICKER_DEVICE(true, false)
            } else {
                HANDLE_PICKER_DEVICE(false, false)
            }
        }

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        temporal_graph::free_device_pointers(d_graph);
    }
    else
    #endif
    {
        if (is_directed) {
            if (forward) {
                HANDLE_PICKER_HOST(true, true)
            } else {
                HANDLE_PICKER_HOST(false, true)
            }
        } else {
            if (forward) {
                HANDLE_PICKER_HOST(true, false)
            } else {
                HANDLE_PICKER_HOST(false, false)
            }
        }
    }

    clear_memory(&rand_nums, graph->use_gpu);

    #undef DISPATCH_HOST
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_HOST
    #undef HANDLE_PICKER_DEVICE

    return result;
}

TemporalGraphStore* TemporalGraph::get_graph() const {
    return graph;
}

size_t TemporalGraph::get_memory_used() const {
    return temporal_graph::get_memory_used(graph);
}
