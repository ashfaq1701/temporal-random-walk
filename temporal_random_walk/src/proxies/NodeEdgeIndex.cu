#include "NodeEdgeIndex.cuh"

#include "../stores/node_edge_index.cuh"
#include "../data/structs.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_edge_range(node_edge_index, dense_node_id, forward, is_directed);
    }
}

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, size_t group_idx, bool forward, bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_timestamp_group_range(node_edge_index, dense_node_id, group_idx, forward, is_directed);
    }
}

__global__ void get_timestamp_group_count_kernel(size_t* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_timestamp_group_count(node_edge_index, dense_node_id, forward, is_directed);
    }
}

#endif

NodeEdgeIndex::NodeEdgeIndex(bool use_gpu): owns_node_edge_index(true) {
    node_edge_index = new NodeEdgeIndexStore(use_gpu);
}

NodeEdgeIndex::NodeEdgeIndex(NodeEdgeIndexStore* existing_node_edge_index)
    : node_edge_index(existing_node_edge_index), owns_node_edge_index(false) {}

NodeEdgeIndex::~NodeEdgeIndex() {
    if (owns_node_edge_index && node_edge_index) {
        delete node_edge_index;
    }
}

NodeEdgeIndex& NodeEdgeIndex::operator=(const NodeEdgeIndex& other) {
    if (this != &other) {
        if (owns_node_edge_index && node_edge_index) {
            delete node_edge_index;
        }

        owns_node_edge_index = other.owns_node_edge_index;
        if (other.owns_node_edge_index) {
            node_edge_index = new NodeEdgeIndexStore(other.node_edge_index->use_gpu);
        } else {
            node_edge_index = other.node_edge_index;
        }
    }
    return *this;
}

void NodeEdgeIndex::clear() const {
    node_edge_index::clear(node_edge_index);
}

void NodeEdgeIndex::rebuild(EdgeDataStore* edge_data, bool is_directed) const {
    node_edge_index::rebuild(node_edge_index, edge_data, is_directed);
}

std::pair<size_t, size_t> NodeEdgeIndex::get_edge_range(int dense_node_id, bool forward, bool is_directed) const {
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        SizeRange* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(SizeRange)));

        NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
        get_edge_range_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, forward, is_directed);
        CUDA_KERNEL_CHECK("After get_edge_range_kernel execution");

        SizeRange host_result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

        return {host_result.from, host_result.to};
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        SizeRange result = node_edge_index::get_edge_range(node_edge_index, dense_node_id, forward, is_directed);
        return {result.from, result.to};
    }
}

std::pair<size_t, size_t> NodeEdgeIndex::get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const {
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        SizeRange* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(SizeRange)));

        NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
        get_timestamp_group_range_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, group_idx, forward, is_directed);
        CUDA_KERNEL_CHECK("After get_timestamp_group_range_kernel execution");

        SizeRange host_result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

        return {host_result.from, host_result.to};
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        SizeRange result = node_edge_index::get_timestamp_group_range(node_edge_index, dense_node_id, group_idx, forward, is_directed);
        return {result.from, result.to};
    }
}

size_t NodeEdgeIndex::get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const {
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(size_t)));

        NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
        get_timestamp_group_count_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, forward, is_directed);
        CUDA_KERNEL_CHECK("After get_timestamp_group_count_kernel execution");

        size_t host_result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return node_edge_index::get_timestamp_group_count(node_edge_index, dense_node_id, forward, is_directed);
    }
}

void NodeEdgeIndex::update_temporal_weights(const EdgeDataStore* edge_data, const double timescale_bound) const {
    #ifdef HAS_CUDA
    if (node_edge_index->use_gpu) {
        node_edge_index::update_temporal_weights_cuda(node_edge_index, edge_data, timescale_bound);
    }
    else
    #endif
    {
        node_edge_index::update_temporal_weights_std(node_edge_index, edge_data, timescale_bound);
    }
}

NodeEdgeIndexStore* NodeEdgeIndex::get_node_edge_index() const {
    return node_edge_index;
}