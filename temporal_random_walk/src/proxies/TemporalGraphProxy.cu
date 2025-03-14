#include "TemporalGraphProxy.cuh"
#include "../common/setup.cuh"

__global__ void get_total_edges_kernel(size_t* result, const TemporalGraph* graph) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_total_edges(graph);
    }
}

__global__ void get_edge_at_kernel(Edge* result, const TemporalGraph* graph, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_edge_at_device(graph, picker_type, timestamp, forward, rand_state);
    }
}

__global__ void get_node_edge_at_kernel(Edge* result, TemporalGraph* graph, int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_node_edge_at_device(graph, node_id, picker_type, timestamp, forward, rand_state);
    }
}

TemporalGraphProxy::TemporalGraphProxy(
    bool is_directed,
    bool use_gpu,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound)
    : owns_graph(true) {

    graph = new TemporalGraph(is_directed, use_gpu, max_time_capacity, enable_weight_computation, timescale_bound);
}

TemporalGraphProxy::TemporalGraphProxy(TemporalGraph* existing_graph)
    : graph(existing_graph), owns_graph(false) {}

TemporalGraphProxy::~TemporalGraphProxy() {
    if (owns_graph && graph) {
        delete graph;
    }
}

void TemporalGraphProxy::update_temporal_weights() const {
    temporal_graph::update_temporal_weights(graph);
}

size_t TemporalGraphProxy::get_total_edges() const {
    if (graph->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        TemporalGraph* d_graph = temporal_graph::to_device_ptr(graph);
        get_total_edges_kernel<<<1, 1>>>(d_result, d_graph);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_graph);

        return host_result;
    } else {
        // Direct call for CPU implementation
        return temporal_graph::get_total_edges(graph);
    }
}

size_t TemporalGraphProxy::get_node_count() const {
    return temporal_graph::get_node_count(graph);
}

int64_t TemporalGraphProxy::get_latest_timestamp() const {
    return temporal_graph::get_latest_timestamp(graph);
}

std::vector<int> TemporalGraphProxy::get_node_ids() const {
    DataBlock<int> node_ids = temporal_graph::get_node_ids(graph);
    std::vector<int> result;

    if (graph->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[node_ids.size];
        cudaMemcpy(host_ids, node_ids.data, node_ids.size * sizeof(int), cudaMemcpyDeviceToHost);

        result.assign(host_ids, host_ids + node_ids.size);
        delete[] host_ids;

        // Free device memory for DataBlock
        if (node_ids.data) {
            cudaFree(node_ids.data);
        }
    } else {
        // For CPU data, can directly copy
        result.assign(node_ids.data, node_ids.data + node_ids.size);

        // Free host memory for DataBlock
        delete[] node_ids.data;
    }

    return result;
}

std::vector<Edge> TemporalGraphProxy::get_edges() const {
    DataBlock<Edge> edges = temporal_graph::get_edges(graph);
    std::vector<Edge> result;

    if (graph->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_edges = new Edge[edges.size];
        cudaMemcpy(host_edges, edges.data, edges.size * sizeof(Edge), cudaMemcpyDeviceToHost);

        result.assign(host_edges, host_edges + edges.size);
        delete[] host_edges;

        // Free device memory for DataBlock
        if (edges.data) {
            cudaFree(edges.data);
        }
    } else {
        // For CPU data, can directly copy
        result.assign(edges.data, edges.data + edges.size);

        // Free host memory for DataBlock
        delete[] edges.data;
    }

    return result;
}

void TemporalGraphProxy::add_multiple_edges(const std::vector<Edge>& new_edges) const {
    if (graph->use_gpu) {
        // Allocate device memory for edges
        Edge* d_edges = nullptr;
        cudaMalloc(&d_edges, new_edges.size() * sizeof(Edge));

        // Copy edges to device
        cudaMemcpy(d_edges, new_edges.data(), new_edges.size() * sizeof(Edge), cudaMemcpyHostToDevice);

        // Call CUDA implementation
        temporal_graph::add_multiple_edges_cuda(graph, d_edges, new_edges.size());

        // Clean up
        cudaFree(d_edges);
    } else {
        // Call CPU implementation directly
        temporal_graph::add_multiple_edges_std(graph, new_edges.data(), new_edges.size());
    }
}

void TemporalGraphProxy::sort_and_merge_edges(size_t start_idx) const {
    if (graph->use_gpu) {
        temporal_graph::sort_and_merge_edges_cuda(graph, start_idx);
    } else {
        temporal_graph::sort_and_merge_edges_std(graph, start_idx);
    }
}

void TemporalGraphProxy::delete_old_edges() const {
    if (graph->use_gpu) {
        temporal_graph::delete_old_edges_cuda(graph);
    } else {
        temporal_graph::delete_old_edges_std(graph);
    }
}

size_t TemporalGraphProxy::count_timestamps_less_than(int64_t timestamp) const {
    if (graph->use_gpu) {
        return temporal_graph::count_timestamps_less_than_cuda(graph, timestamp);
    } else {
        return temporal_graph::count_timestamps_less_than_std(graph, timestamp);
    }
}

size_t TemporalGraphProxy::count_timestamps_greater_than(int64_t timestamp) const {
    if (graph->use_gpu) {
        return temporal_graph::count_timestamps_greater_than_cuda(graph, timestamp);
    } else {
        return temporal_graph::count_timestamps_greater_than_std(graph, timestamp);
    }
}

size_t TemporalGraphProxy::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    if (graph->use_gpu) {
        return temporal_graph::count_node_timestamps_less_than_cuda(graph, node_id, timestamp);
    } else {
        return temporal_graph::count_node_timestamps_less_than_std(graph, node_id, timestamp);
    }
}

size_t TemporalGraphProxy::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    if (graph->use_gpu) {
        return temporal_graph::count_node_timestamps_greater_than_cuda(graph, node_id, timestamp);
    } else {
        return temporal_graph::count_node_timestamps_greater_than_std(graph, node_id, timestamp);
    }
}

Edge TemporalGraphProxy::get_edge_at(RandomPickerType picker_type, int64_t timestamp, bool forward) const {
    if (graph->use_gpu) {
        // Set up random state
        curandState* d_rand_states;
        cudaMalloc(&d_rand_states, sizeof(curandState));
        setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));

        // Allocate memory for the result
        Edge* d_result;
        cudaMalloc(&d_result, sizeof(Edge));

        // Copy graph to device
        TemporalGraph* d_graph = temporal_graph::to_device_ptr(graph);

        // Launch kernel
        get_edge_at_kernel<<<1, 1>>>(d_result, d_graph, picker_type, timestamp, forward, d_rand_states);

        // Copy result back to host
        Edge host_result;
        cudaMemcpy(&host_result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_rand_states);
        cudaFree(d_result);
        cudaFree(d_graph);

        return host_result;
    } else {
        // Call CPU implementation directly
        return temporal_graph::get_edge_at_host(graph, picker_type, timestamp, forward);
    }
}

Edge TemporalGraphProxy::get_node_edge_at(int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward) const {
    if (graph->use_gpu) {
        // Set up random state
        curandState* d_rand_states;
        cudaMalloc(&d_rand_states, sizeof(curandState));
        setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));

        // Allocate memory for the result
        Edge* d_result;
        cudaMalloc(&d_result, sizeof(Edge));

        // Copy graph to device
        TemporalGraph* d_graph = temporal_graph::to_device_ptr(graph);

        // Launch kernel
        get_node_edge_at_kernel<<<1, 1>>>(d_result, d_graph, node_id, picker_type, timestamp, forward, d_rand_states);

        // Copy result back to host
        Edge host_result;
        cudaMemcpy(&host_result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_rand_states);
        cudaFree(d_result);
        cudaFree(d_graph);

        return host_result;
    } else {
        // Call CPU implementation directly
        return temporal_graph::get_node_edge_at_host(graph, node_id, picker_type, timestamp, forward);
    }
}
