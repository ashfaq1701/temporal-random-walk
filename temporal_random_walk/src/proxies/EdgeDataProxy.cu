#include "EdgeDataProxy.cuh"

#ifdef HAS_CUDA

__global__ void empty_kernel(bool* result, const EdgeData* edge_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::empty(edge_data);
    }
}

__global__ void size_kernel(size_t* result, const EdgeData* edge_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::size(edge_data);
    }
}


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

__global__ void get_timestamp_group_count_kernel(size_t* result, const EdgeData* edge_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data::get_timestamp_group_count(edge_data);
    }
}

#endif

EdgeDataProxy::EdgeDataProxy(const bool use_gpu): owns_edge_data(true) {
    edge_data = new EdgeData(use_gpu);
}

EdgeDataProxy::EdgeDataProxy(EdgeData* existing_edge_data) : edge_data(existing_edge_data), owns_edge_data(false) {}

EdgeDataProxy::~EdgeDataProxy() {
    if (owns_edge_data && edge_data) {
        delete edge_data;
    }
}

EdgeDataProxy& EdgeDataProxy::operator=(const EdgeDataProxy& other) {
    if (this != &other) {
        if (owns_edge_data && edge_data) {
            delete edge_data;
        }

        owns_edge_data = other.owns_edge_data;
        if (other.owns_edge_data) {
            edge_data = new EdgeData(other.edge_data->use_gpu);
        } else {
            edge_data = other.edge_data;
        }
    }
    return *this;
}

void EdgeDataProxy::clear() const {
    edge_data::clear(edge_data);
}

size_t EdgeDataProxy::size() const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        size_kernel<<<1, 1>>>(d_result, d_edge_data);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return edge_data::size(edge_data);
    }
}

void EdgeDataProxy::resize(const size_t size) const {
    edge_data::resize(edge_data, size);
}

void EdgeDataProxy::set_size(const size_t size) const {
    edge_data::set_size(edge_data, size);
}

bool EdgeDataProxy::empty() const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        empty_kernel<<<1, 1>>>(d_result, d_edge_data);

        bool host_result;
        cudaMemcpy(&host_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return edge_data::empty(edge_data);
    }
}

void EdgeDataProxy::add_edges(const std::vector<int>& sources, const std::vector<int>& targets, const std::vector<int64_t>& timestamps) const {
    if (sources.size() != targets.size() || sources.size() != timestamps.size()) {
        throw std::runtime_error("Vector sizes don't match for add_edges");
    }

    const size_t size = sources.size();
    // Direct call for CPU implementation
    edge_data::add_edges(edge_data, sources.data(), targets.data(), timestamps.data(), size);
}

void EdgeDataProxy::push_back(const int source, const int target, const int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Allocate GPU memory for single elements
        int* d_source = nullptr;
        int* d_target = nullptr;
        int64_t* d_timestamp = nullptr;

        cudaMalloc(&d_source, sizeof(int));
        cudaMalloc(&d_target, sizeof(int));
        cudaMalloc(&d_timestamp, sizeof(int64_t));

        // Copy data to GPU
        cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, &target, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timestamp, &timestamp, sizeof(int64_t), cudaMemcpyHostToDevice);

        // Call add_edges with single element
        edge_data::add_edges(edge_data, d_source, d_target, d_timestamp, 1);

        // Free GPU memory
        cudaFree(d_source);
        cudaFree(d_target);
        cudaFree(d_timestamp);
    }
    else
    #endif
    {
        // For CPU implementation, create small arrays
        const int sources[1] = { source };
        const int targets[1] = { target };
        const int64_t timestamps[1] = { timestamp };

        edge_data::add_edges(edge_data, sources, targets, timestamps, 1);
    }
}

std::vector<Edge> EdgeDataProxy::get_edges() const {
    // Call the optimized edge_data::get_edges function directly
    const DataBlock<Edge> edges_block = edge_data::get_edges(edge_data);
    std::vector<Edge> result;

    // Copy data from DataBlock to std::vector
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_edges = new Edge[edges_block.size];
        cudaMemcpy(host_edges, edges_block.data, edges_block.size * sizeof(Edge), cudaMemcpyDeviceToHost);

        result.assign(host_edges, host_edges + edges_block.size);
        delete[] host_edges;

        // Free device memory for DataBlock
        if (edges_block.data) {
            cudaFree(edges_block.data);
        }
    }
    else
    #endif
    {
        // For CPU data, can directly copy
        result.assign(edges_block.data, edges_block.data + edges_block.size);

        // Free host memory for DataBlock
        delete[] edges_block.data;
    }

    return result;
}

void EdgeDataProxy::update_timestamp_groups() const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        edge_data::update_timestamp_groups_cuda(edge_data);
    }
    else
    #endif
    {
        edge_data::update_timestamp_groups_std(edge_data);
    }
}

void EdgeDataProxy::update_temporal_weights(double timescale_bound) const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        edge_data::update_temporal_weights_cuda(edge_data, timescale_bound);
    }
    else
    #endif
    {
        edge_data::update_temporal_weights_std(edge_data, timescale_bound);
    }
}

std::pair<size_t, size_t> EdgeDataProxy::get_timestamp_group_range(size_t group_idx) const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        SizeRange* d_result;
        cudaMalloc(&d_result, sizeof(SizeRange));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        get_timestamp_group_range_kernel<<<1, 1>>>(d_result, d_edge_data, group_idx);

        SizeRange host_result;
        cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return {host_result.from, host_result.to};
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        SizeRange result = edge_data::get_timestamp_group_range(edge_data, group_idx);
        return {result.from, result.to};
    }
}

size_t EdgeDataProxy::get_timestamp_group_count() const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        get_timestamp_group_count_kernel<<<1, 1>>>(d_result, d_edge_data);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return edge_data::get_timestamp_group_count(edge_data);
    }
}

size_t EdgeDataProxy::find_group_after_timestamp(int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        find_group_after_timestamp_kernel<<<1, 1>>>(d_result, d_edge_data, timestamp);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return edge_data::find_group_after_timestamp(edge_data, timestamp);
    }
}

size_t EdgeDataProxy::find_group_before_timestamp(int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeData* d_edge_data = edge_data::to_device_ptr(edge_data);
        find_group_before_timestamp_kernel<<<1, 1>>>(d_result, d_edge_data, timestamp);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_edge_data);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return edge_data::find_group_before_timestamp(edge_data, timestamp);
    }
}
