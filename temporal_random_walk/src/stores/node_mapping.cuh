#ifndef NODE_MAPPING_STORE_H
#define NODE_MAPPING_STORE_H

#include <cstddef>

#include "../data/structs.cuh"
#include "../data/IntHashMap.cuh"
#include "edge_data.cuh"

struct NodeMappingStore {
    size_t node_count_max_bound;
    bool use_gpu;

    IntHashMap* node_index;

    bool *is_deleted = nullptr;
    size_t is_deleted_size = 0;

    explicit NodeMappingStore(const size_t node_count_max_bound, const bool use_gpu)
        : node_count_max_bound(node_count_max_bound), use_gpu(use_gpu) {
        node_index = new IntHashMap(node_count_max_bound, use_gpu);
    }
};

__global__ void copy_count_elements_kernel(size_t* dst, const IntHashMap* src);

namespace node_mapping {
    /**
     * Common Methods
     */
    HOST int to_dense(const NodeMappingStore *node_mapping, int sparse_id);

    HOST DEVICE size_t size(const NodeMappingStore *node_mapping);

    HOST size_t active_size(const NodeMappingStore *node_mapping);

    HOST DataBlock<int> get_active_node_ids(const NodeMappingStore *node_mapping);

    HOST void clear(NodeMappingStore *node_mapping);

    HOST void reserve(NodeMappingStore *node_mapping, size_t size);

    HOST void mark_node_deleted(const NodeMappingStore *node_mapping, int sparse_id);

    HOST bool has_node_host(const NodeMappingStore *node_mapping, int sparse_id);

    DEVICE bool has_node_device(const NodeMappingStore *node_mapping, int sparse_id);

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

    DEVICE int to_dense_from_ptr_device(const int *sparse_to_dense, int sparse_id, size_t size);

    DEVICE void mark_node_deleted_from_ptr(bool *is_deleted, int sparse_id, int size);

    HOST NodeMappingStore* to_device_ptr(const NodeMappingStore* node_mapping);

    #endif
}

#endif // NODE_MAPPING_STORE__H
