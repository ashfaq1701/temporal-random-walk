#include "NodeMappingProxy.cuh"

// Kernel implementations
__global__ void size_kernel(size_t* result, const NodeMapping* node_mapping) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::size(node_mapping);
    }
}

__global__ void to_dense_kernel(int* result, const NodeMapping* node_mapping, const int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::to_dense_device(node_mapping, sparse_id);
    }
}

__global__ void to_sparse_kernel(int* result, const NodeMapping* node_mapping, const int dense_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::to_sparse_device(node_mapping, dense_id);
    }
}

__global__ void has_node_kernel(bool* result, const NodeMapping* node_mapping, int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::has_node(node_mapping, sparse_id);
    }
}

__global__ void mark_node_deleted_kernel(const NodeMapping* node_mapping, int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        node_mapping::mark_node_deleted_from_ptr(
            node_mapping->is_deleted,
            sparse_id,
            static_cast<int>(node_mapping->is_deleted_size));
    }
}

NodeMappingProxy::NodeMappingProxy(const bool use_gpu) : owns_node_mapping(true) {
    node_mapping = new NodeMapping(use_gpu);
}

NodeMappingProxy::NodeMappingProxy(NodeMapping* existing_node_mapping)
    : node_mapping(existing_node_mapping), owns_node_mapping(false) {}

NodeMappingProxy::~NodeMappingProxy() {
    if (owns_node_mapping && node_mapping) {
        delete node_mapping;
    }
}

int NodeMappingProxy::to_dense(const int sparse_id) const {
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        NodeMapping* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        to_dense_kernel<<<1, 1>>>(d_result, d_node_mapping, sparse_id);

        int host_result;
        cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    } else {
        // Direct call for CPU implementation
        return node_mapping::to_dense(node_mapping, sparse_id);
    }
}

int NodeMappingProxy::to_sparse(int dense_id) const {
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        NodeMapping* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        to_sparse_kernel<<<1, 1>>>(d_result, d_node_mapping, dense_id);

        int host_result;
        cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    } else {
        // Direct call for CPU implementation
        return node_mapping::to_sparse(node_mapping, dense_id);
    }
}

size_t NodeMappingProxy::size() const {
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        NodeMapping* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        size_kernel<<<1, 1>>>(d_result, d_node_mapping);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    } else {
        // Direct call for CPU implementation
        return node_mapping::size(node_mapping);
    }
}

size_t NodeMappingProxy::active_size() const {
    // This function is HOST only, so no need for a kernel
    return node_mapping::active_size(node_mapping);
}

std::vector<int> NodeMappingProxy::get_active_node_ids() const {
    // Call the namespace function to get DataBlock
    DataBlock<int> ids_block = node_mapping::get_active_node_ids(node_mapping);
    std::vector<int> result;

    // Copy data from DataBlock to std::vector
    if (node_mapping->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[ids_block.size];
        cudaMemcpy(host_ids, ids_block.data, ids_block.size * sizeof(int), cudaMemcpyDeviceToHost);

        result.assign(host_ids, host_ids + ids_block.size);
        delete[] host_ids;

        // Free device memory for DataBlock
        if (ids_block.data) {
            cudaFree(ids_block.data);
        }
    } else {
        // For CPU data, can directly copy
        result.assign(ids_block.data, ids_block.data + ids_block.size);

        // Free host memory for DataBlock
        delete[] ids_block.data;
    }

    return result;
}

void NodeMappingProxy::clear() const {
    // This function is HOST only
    node_mapping::clear(node_mapping);
}

void NodeMappingProxy::reserve(size_t size) const {
    // This function is HOST only
    node_mapping::reserve(node_mapping, size);
}

void NodeMappingProxy::mark_node_deleted(int sparse_id) const {
    if (node_mapping->use_gpu) {
        // For GPU, use mark_node_deleted_from_ptr via kernel
        NodeMapping* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        mark_node_deleted_kernel<<<1, 1>>>(d_node_mapping, sparse_id);
        cudaFree(d_node_mapping);
    } else {
        // For CPU, use regular mark_node_deleted
        node_mapping::mark_node_deleted(node_mapping, sparse_id);
    }
}

bool NodeMappingProxy::has_node(int sparse_id) const {
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));

        NodeMapping* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        has_node_kernel<<<1, 1>>>(d_result, d_node_mapping, sparse_id);

        bool host_result;
        cudaMemcpy(&host_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    } else {
        // Direct call for CPU implementation
        return node_mapping::has_node(node_mapping, sparse_id);
    }
}

std::vector<int> NodeMappingProxy::get_all_sparse_ids() const {
    // This function is HOST only
    MemoryView<int> sparse_ids = node_mapping::get_all_sparse_ids(node_mapping);
    std::vector<int> result;

    if (node_mapping->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[sparse_ids.size];
        cudaMemcpy(host_ids, sparse_ids.data, sparse_ids.size * sizeof(int), cudaMemcpyDeviceToHost);

        result.assign(host_ids, host_ids + sparse_ids.size);
        delete[] host_ids;
    } else {
        // For CPU data, can directly copy
        result.assign(sparse_ids.data, sparse_ids.data + sparse_ids.size);
    }

    return result;
}

void NodeMappingProxy::update(const EdgeData* edge_data, size_t start_idx, size_t end_idx) const {
    // Choose between std and cuda implementations based on mode
    if (node_mapping->use_gpu) {
        node_mapping::update_cuda(node_mapping, edge_data, start_idx, end_idx);
    } else {
        node_mapping::update_std(node_mapping, edge_data, start_idx, end_idx);
    }
}
