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

__global__ void get_edge_at_kernel(Edge* result, const TemporalGraphStore* graph, const RandomPickerType picker_type, const int64_t timestamp, const bool forward, const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_edge_at_device(graph, picker_type, timestamp, forward, rand_nums[0], rand_nums[1]);
    }
}

__global__ void get_node_edge_at_kernel(Edge* result, TemporalGraphStore* graph, const int node_id, const RandomPickerType picker_type, const int64_t timestamp, const bool forward, const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_node_edge_at_device(graph, node_id, picker_type, timestamp, forward, rand_nums[0], rand_nums[1]);
    }
}

#endif

TemporalGraph::TemporalGraph(
    const bool is_directed,
    const bool use_gpu,
    const int64_t max_time_capacity,
    const bool enable_weight_computation,
    const double timescale_bound)
    : owns_graph(true) {

    graph = new TemporalGraphStore(
        is_directed,
        use_gpu,
        max_time_capacity,
        enable_weight_computation,
        timescale_bound);
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
                other.graph->timescale_bound);
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
        CUDA_CHECK_AND_CLEAR(cudaFree(d_graph));

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return temporal_graph::get_total_edges(graph);
    }
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

void TemporalGraph::add_multiple_edges(const std::vector<Edge>& new_edges) const {
    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Allocate device memory for edges
        Edge* d_edges = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_edges, new_edges.size() * sizeof(Edge)));

        // Copy edges to device
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_edges, new_edges.data(), new_edges.size() * sizeof(Edge), cudaMemcpyHostToDevice));

        // Call CUDA implementation
        temporal_graph::add_multiple_edges_cuda(graph, d_edges, new_edges.size());

        // Clean up
        CUDA_CHECK_AND_CLEAR(cudaFree(d_edges));
    }
    else
    #endif
    {
        // Call CPU implementation directly
        temporal_graph::add_multiple_edges_std(graph, new_edges.data(), new_edges.size());
    }
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

Edge TemporalGraph::get_edge_at(RandomPickerType picker_type, int64_t timestamp, bool forward) const {
    double* rand_nums = generate_n_random_numbers(1, graph->use_gpu);
    Edge result;

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Allocate memory for the result
        Edge* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));

        // Copy graph to device
        TemporalGraphStore* d_graph = temporal_graph::to_device_ptr(graph);

        // Launch kernel
        get_edge_at_kernel<<<1, 1>>>(d_result, d_graph, picker_type, timestamp, forward, rand_nums);
        CUDA_KERNEL_CHECK("After get_edge_at_kernel execution");

        // Copy result back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));

        // Clean up
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_graph));
    }
    else
    #endif
    {
        // Call CPU implementation directly
        result = temporal_graph::get_edge_at_host(graph, picker_type, timestamp, forward, rand_nums[0], rand_nums[1]);
    }

    clear_memory(&rand_nums, graph->use_gpu);
    return result;
}

Edge TemporalGraph::get_node_edge_at(int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward) const {
    double* rand_nums = generate_n_random_numbers(1, graph->use_gpu);
    Edge result;

    #ifdef HAS_CUDA
    if (graph->use_gpu) {
        // Allocate memory for the result
        Edge* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));

        // Copy graph to device
        TemporalGraphStore* d_graph = temporal_graph::to_device_ptr(graph);

        // Launch kernel
        get_node_edge_at_kernel<<<1, 1>>>(d_result, d_graph, node_id, picker_type, timestamp, forward, rand_nums);
        CUDA_KERNEL_CHECK("After get_node_edge_at_kernel execution");

        // Copy result back to host
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));

        // Clean up
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_graph));
    }
    else
    #endif
    {
        // Call CPU implementation directly
        result = temporal_graph::get_node_edge_at_host(graph, node_id, picker_type, timestamp, forward, rand_nums[0], rand_nums[1]);
    }

    clear_memory(&rand_nums, graph->use_gpu);
    return result;
}
