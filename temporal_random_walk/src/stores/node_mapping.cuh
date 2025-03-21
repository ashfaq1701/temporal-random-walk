#ifndef NODE_MAPPING_STORE_H
#define NODE_MAPPING_STORE_H

#include <cstddef>

#include "../data/structs.cuh"
#include "edge_data.cuh"
#include "../utils/utils.cuh"

constexpr double HASH_INDEX_LOAD_FACTOR = 0.6;

struct NodeMappingStore {

    int node_count_max_bound;
    bool use_gpu;

    int* node_index = nullptr;
    bool* is_deleted = nullptr;

    int capacity;
    mutable size_t node_size;

    explicit NodeMappingStore(
        const int node_count_max_bound,
        const bool use_gpu)
        : node_count_max_bound(node_count_max_bound), use_gpu(use_gpu), node_size(0) {
        capacity = next_prime(static_cast<int>(node_count_max_bound / HASH_INDEX_LOAD_FACTOR));
        allocate_memory(&node_index, capacity, use_gpu);
        fill_memory(node_index, capacity, -1, use_gpu);

        allocate_memory(&is_deleted, capacity, use_gpu);
        fill_memory(is_deleted, capacity, true, use_gpu);
    }

    ~NodeMappingStore() {
        #ifdef HAS_CUDA
        if (use_gpu) {
            if (node_index) cudaFree(node_index);
            if (is_deleted) cudaFree(is_deleted);
        }
        else
        #endif
        {
            delete[] node_index;
            delete[] is_deleted;
        }
    }
};

namespace node_mapping {
    /**
     * Common Methods
     */

    HOST DEVICE void add_node(int* node_index, int node_id);

    HOST int to_dense(const NodeMappingStore *node_mapping, int sparse_id);

    HOST DEVICE size_t size(const NodeMappingStore *node_mapping);

    HOST size_t active_size(const NodeMappingStore *node_mapping);

    HOST DataBlock<int> get_active_node_ids(const NodeMappingStore *node_mapping);

    HOST void clear(const NodeMappingStore *node_mapping);

    HOST void mark_node_deleted(const NodeMappingStore *node_mapping, int sparse_id);

    HOST DataBlock<int> get_all_sparse_ids(const NodeMappingStore *node_mapping);

    HOST DEVICE bool has_node(const NodeMappingStore *node_mapping, int sparse_id);

    /**
     * Std Implementations
     */
    HOST void update_std(NodeMappingStore *node_mapping, const EdgeDataStore *edge_data, size_t start_idx, size_t end_idx);

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA
    HOST void update_cuda(NodeMappingStore *node_mapping, const EdgeDataStore *edge_data, size_t start_idx, size_t end_idx);

    /**
     * Device functions
     */

    DEVICE int to_dense_device(const NodeMappingStore *node_mapping, int sparse_id);

    DEVICE int to_dense_from_ptr_device(const int* node_index, int sparse_id, int capacity);

    DEVICE void mark_node_deleted_from_ptr(bool *is_deleted, const int *node_index, int sparse_id, int capacity);

    HOST NodeMappingStore* to_device_ptr(const NodeMappingStore* node_mapping);

    #endif
}

#endif // NODE_MAPPING_STORE__H
