#include "NodeMapping.cuh"

#ifdef HAS_CUDA

// Kernel implementations
__global__ void size_kernel(size_t* result, const NodeMappingStore* node_mapping) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::size(node_mapping);
    }
}

__global__ void to_dense_kernel(int* result, const NodeMappingStore* node_mapping, const int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::to_dense_device(node_mapping, sparse_id);
    }
}

__global__ void has_node_kernel(bool* result, const NodeMappingStore* node_mapping, int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_mapping::has_node(node_mapping, sparse_id);
    }
}

__global__ void mark_node_deleted_kernel(const NodeMappingStore* node_mapping, int sparse_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        node_mapping::mark_node_deleted_from_ptr(
            node_mapping->is_deleted,
            node_mapping->node_index,
            sparse_id,
            node_mapping->capacity);
    }
}

#endif

NodeMapping::NodeMapping(const int node_count_max_bound, const bool use_gpu) : owns_node_mapping(true) {
    node_mapping = new NodeMappingStore(node_count_max_bound, use_gpu);
}

NodeMapping::NodeMapping(NodeMappingStore* existing_node_mapping)
    : node_mapping(existing_node_mapping), owns_node_mapping(false) {}

NodeMapping::~NodeMapping() {
    if (owns_node_mapping && node_mapping) {
        delete node_mapping;
    }
}

NodeMapping& NodeMapping::operator=(const NodeMapping& other) {
    if (this != &other) {
        if (owns_node_mapping && node_mapping) {
            delete node_mapping;
        }

        owns_node_mapping = other.owns_node_mapping;
        if (other.owns_node_mapping) {
            node_mapping = new NodeMappingStore(other.node_mapping->node_count_max_bound, other.node_mapping->use_gpu);
        } else {
            node_mapping = other.node_mapping;
        }
    }
    return *this;
}

int NodeMapping::to_dense(const int sparse_id) const {
    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        NodeMappingStore* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        to_dense_kernel<<<1, 1>>>(d_result, d_node_mapping, sparse_id);

        int host_result;
        cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return node_mapping::to_dense(node_mapping, sparse_id);
    }
}

size_t NodeMapping::size() const {
    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        NodeMappingStore* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        size_kernel<<<1, 1>>>(d_result, d_node_mapping);

        size_t host_result;
        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return node_mapping::size(node_mapping);
    }
}

size_t NodeMapping::active_size() const {
    // This function is HOST only, so no need for a kernel
    return node_mapping::active_size(node_mapping);
}

std::vector<int> NodeMapping::get_active_node_ids() const {
    // Call the namespace function to get DataBlock
    DataBlock<int> ids_block = node_mapping::get_active_node_ids(node_mapping);
    std::vector<int> result;

    #ifdef HAS_CUDA
    // Copy data from DataBlock to std::vector
    if (node_mapping->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[ids_block.size];
        cudaMemcpy(host_ids, ids_block.data, ids_block.size * sizeof(int), cudaMemcpyDeviceToHost);

        result.assign(host_ids, host_ids + ids_block.size);
        delete[] host_ids;
    }
    else
    #endif
    {
        // For CPU data, can directly copy
        result.assign(ids_block.data, ids_block.data + ids_block.size);
    }

    return result;
}

void NodeMapping::clear() const {
    // This function is HOST only
    node_mapping::clear(node_mapping);
}

void NodeMapping::mark_node_deleted(int sparse_id) const {
    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // For GPU, use mark_node_deleted_from_ptr via kernel
        NodeMappingStore* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        mark_node_deleted_kernel<<<1, 1>>>(d_node_mapping, sparse_id);
        cudaFree(d_node_mapping);
    }
    else
    #endif
    {
        // For CPU, use regular mark_node_deleted
        node_mapping::mark_node_deleted(node_mapping, sparse_id);
    }
}

bool NodeMapping::has_node(int sparse_id) const {
    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // Call via CUDA kernel for GPU implementation
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));

        NodeMappingStore* d_node_mapping = node_mapping::to_device_ptr(node_mapping);
        has_node_kernel<<<1, 1>>>(d_result, d_node_mapping, sparse_id);

        bool host_result;
        cudaMemcpy(&host_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_node_mapping);

        return host_result;
    }
    else
    #endif
    {
        // Direct call for CPU implementation
        return node_mapping::has_node(node_mapping, sparse_id);
    }
}

std::vector<int> NodeMapping::get_all_sparse_ids() const {
    // This function is HOST only
    DataBlock<int> sparse_ids = node_mapping::get_all_sparse_ids(node_mapping);
    std::vector<int> result;

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // For GPU data, need to copy from device to host
        const auto host_ids = new int[sparse_ids.size];
        cudaMemcpy(host_ids, sparse_ids.data, sparse_ids.size * sizeof(int), cudaMemcpyDeviceToHost);

        result.assign(host_ids, host_ids + sparse_ids.size);
        delete[] host_ids;
    }
    else
    #endif
    {
        // For CPU data, can directly copy
        result.assign(sparse_ids.data, sparse_ids.data + sparse_ids.size);
    }

    return result;
}

void NodeMapping::update(const EdgeDataStore* edge_data, size_t start_idx, size_t end_idx) const {
    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        node_mapping::update_cuda(node_mapping, edge_data, start_idx, end_idx);
    }
    else
    #endif
    {
        node_mapping::update_std(node_mapping, edge_data, start_idx, end_idx);
    }
}
