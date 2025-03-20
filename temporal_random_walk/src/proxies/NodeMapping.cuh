#ifndef NODE_MAPPING_H
#define NODE_MAPPING_H

#include "../stores/node_mapping.cuh"

// Kernel declarations for device operations
#ifdef HAS_CUDA

__global__ void size_kernel(size_t* result, const NodeMappingStore* node_mapping);
__global__ void to_dense_kernel(int* result, const NodeMappingStore* node_mapping, int sparse_id);
__global__ void to_sparse_kernel(int* result, const NodeMappingStore* node_mapping, int dense_id);
__global__ void has_node_kernel(bool* result, const NodeMappingStore* node_mapping, int sparse_id);
__global__ void mark_node_deleted_kernel(const NodeMappingStore* node_mapping, int sparse_id);

#endif

class NodeMapping {

public:

    NodeMappingStore* node_mapping;
    bool owns_node_mapping;

    std::vector<int> sparse_to_dense() const {
        #ifdef HAS_CUDA
        if (node_mapping->use_gpu) {
            std::vector<int> result(node_mapping->sparse_to_dense_size);
            cudaMemcpy(result.data(), node_mapping->sparse_to_dense,
                      node_mapping->sparse_to_dense_size * sizeof(int),
                      cudaMemcpyDeviceToHost);
            return result;
        }
        else
        #endif
        {
            return std::vector<int>(node_mapping->sparse_to_dense,
                                   node_mapping->sparse_to_dense +
                                   node_mapping->sparse_to_dense_size);
        }
    }

    std::vector<int> dense_to_sparse() const {
        #ifdef HAS_CUDA
        if (node_mapping->use_gpu) {
            std::vector<int> result(node_mapping->dense_to_sparse_size);
            cudaMemcpy(result.data(), node_mapping->dense_to_sparse,
                      node_mapping->dense_to_sparse_size * sizeof(int),
                      cudaMemcpyDeviceToHost);
            return result;
        }
        else
        #endif
        {
            return std::vector<int>(node_mapping->dense_to_sparse,
                                   node_mapping->dense_to_sparse +
                                   node_mapping->dense_to_sparse_size);
        }
    }

    std::vector<bool> is_deleted() const {
        #ifdef HAS_CUDA
        if (node_mapping->use_gpu) {
            std::vector<char> temp_buffer(node_mapping->is_deleted_size);

            cudaMemcpy(temp_buffer.data(), node_mapping->is_deleted,
                       node_mapping->is_deleted_size * sizeof(bool),
                       cudaMemcpyDeviceToHost);

            return std::vector<bool>(temp_buffer.begin(), temp_buffer.end());
        }
        else
        #endif
        {
            return std::vector<bool>(node_mapping->is_deleted,
                        node_mapping->is_deleted + node_mapping->is_deleted_size);
        }
    }

    explicit NodeMapping(int node_count_max_bound, bool use_gpu);

    explicit NodeMapping(NodeMappingStore* existing_node_mapping);

    ~NodeMapping();

    NodeMapping& operator=(const NodeMapping& other);

    [[nodiscard]] int to_dense(int sparse_id) const;

    [[nodiscard]] int to_sparse(int dense_id) const;

    [[nodiscard]] size_t size() const;

    [[nodiscard]] size_t active_size() const;

    [[nodiscard]] std::vector<int> get_active_node_ids() const;

    void clear() const;

    void reserve(size_t size) const;

    void mark_node_deleted(int sparse_id) const;

    [[nodiscard]] bool has_node(int sparse_id) const;

    [[nodiscard]] std::vector<int> get_all_sparse_ids() const;

    void update(const EdgeDataStore* edge_data, size_t start_idx, size_t end_idx) const;
};

#endif // NODE_MAPPING_H
