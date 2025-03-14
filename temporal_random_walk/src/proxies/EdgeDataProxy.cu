#include "EdgeDataProxy.cuh"

__global__ void find_group_after_timestamp_kernel(size_t* result, const EdgeData* edge_data, int64_t timestamp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::find_group_after_timestamp_device(edge_data, timestamp);
    }
}

__global__ void find_group_before_timestamp_kernel(size_t* result, const EdgeData* edge_data, int64_t timestamp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::find_group_before_timestamp_device(edge_data, timestamp);
    }
}

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const EdgeData* edge_data, size_t group_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::get_timestamp_group_range(edge_data, group_idx);
    }
}

EdgeDataProxy::EdgeDataProxy(const bool use_gpu): owns_edge_data(true) {
    edge_data = new EdgeData(use_gpu);
}

EdgeDataProxy::EdgeDataProxy(EdgeData* existing_edge_data) : edge_data(existing_edge_data), owns_edge_data(false) {}

EdgeDataProxy::~EdgeDataProxy() {
    if (owns_edge_data && edge_data) {
        delete edge_data;
    }
}

void EdgeDataProxy::reserve(const size_t size) const {
    edge_data::reserve(edge_data, size);
}

[[nodiscard]] size_t EdgeDataProxy::size() const {
    return edge_data::size(edge_data);
}

[[nodiscard]] bool EdgeDataProxy::empty() const {
    return edge_data::empty(edge_data);
}

void EdgeDataProxy::add_edges(const std::vector<int>& sources, const std::vector<int>& targets, const std::vector<int64_t>& timestamps) const {
    if (sources.size() != targets.size() || sources.size() != timestamps.size()) {
        throw std::invalid_argument("Sources, targets and timestamps must have the same size");
    }

    if (edge_data->use_gpu) {
        // Allocate device memory for the data
        int* d_sources;
        int* d_targets;
        int64_t* d_timestamps;

        cudaMalloc(&d_sources, sources.size() * sizeof(int));
        cudaMalloc(&d_targets, targets.size() * sizeof(int));
        cudaMalloc(&d_timestamps, timestamps.size() * sizeof(int64_t));

        // Copy data to device
        cudaMemcpy(d_sources, sources.data(), sources.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timestamps, timestamps.data(), timestamps.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Add edges with device pointers
        edge_data::add_edges(edge_data, d_sources, d_targets, d_timestamps, sources.size());

        // Free device memory
        cudaFree(d_sources);
        cudaFree(d_targets);
        cudaFree(d_timestamps);
    } else {
        // CPU case - pass host pointers directly
        edge_data::add_edges(edge_data, sources.data(), targets.data(), timestamps.data(), sources.size());
    }
}

void EdgeDataProxy::push_back(const int source, const int target, const int64_t timestamp) const {
    if (edge_data->use_gpu) {
        // Allocate device memory for the data
        int* d_source;
        int* d_target;
        int64_t* d_timestamp;

        cudaMalloc(&d_source, sizeof(int));
        cudaMalloc(&d_target, sizeof(int));
        cudaMalloc(&d_timestamp, sizeof(int64_t));

        // Copy data to device
        cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, &target, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timestamp, &timestamp, sizeof(int64_t), cudaMemcpyHostToDevice);

        // Add edge with device pointers
        edge_data::add_edges(edge_data, d_source, d_target, d_timestamp, 1);

        // Free device memory
        cudaFree(d_source);
        cudaFree(d_target);
        cudaFree(d_timestamp);
    } else {
        // Create temporary arrays on host
        const int src = source;
        const int tgt = target;
        const int64_t ts = timestamp;

        // Call add_edges directly
        edge_data::add_edges(edge_data, &src, &tgt, &ts, 1);
    }
}

void EdgeDataProxy::update_timestamp_groups() const {
    if (edge_data->use_gpu) {
        edge_data::update_timestamp_groups_cuda(edge_data);
    } else {
        edge_data::update_timestamp_groups_std(edge_data);
    }
}

void EdgeDataProxy::update_temporal_weights(const double timescale_bound) const {
    if (edge_data->use_gpu) {
        edge_data::update_temporal_weights_cuda(edge_data, timescale_bound);
    } else {
        edge_data::update_temporal_weights_std(edge_data, timescale_bound);
    }
}

[[nodiscard]] std::pair<size_t, size_t> EdgeDataProxy::get_timestamp_group_range(size_t group_idx) const {
    if (edge_data->use_gpu) {
        // Allocate device memory for the result
        SizeRange host_result{0, 0};
        SizeRange* d_result;
        cudaMalloc(&d_result, sizeof(SizeRange));

        // Create a device copy of the edge_data
        EdgeData* device_edge_data = edge_data::to_device_ptr(edge_data);

        // Launch kernel to execute the device function
        get_timestamp_group_range_kernel<<<1, 1>>>(d_result, device_edge_data, group_idx);

        // Copy result back to host
        cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost);

        // Clean up device memory
        cudaFree(d_result);
        cudaFree(device_edge_data);

        return {host_result.from, host_result.to};
    } else {
        // CPU case - call the function directly
        SizeRange range = edge_data::get_timestamp_group_range(edge_data, group_idx);
        return {range.from, range.to};
    }
}
