#ifndef NODE_MAPPING_H
#define NODE_MAPPING_H

#include <cstddef>

#include "../data/structs.cuh"
#include "edge_data.cuh"

struct NodeMapping {
    bool use_gpu;

    int *sparse_to_dense = nullptr;
    size_t sparse_to_dense_size = 0;

    int *dense_to_sparse = nullptr;
    size_t dense_to_sparse_size = 0;

    bool *is_deleted = nullptr;
    size_t is_deleted_size = 0;

    explicit NodeMapping(const bool use_gpu): use_gpu(use_gpu) {
    }

    ~NodeMapping() {
        if (use_gpu) {
            if (sparse_to_dense) cudaFree(sparse_to_dense);
            if (dense_to_sparse) cudaFree(dense_to_sparse);
            if (is_deleted) cudaFree(is_deleted);
        } else {
            delete[] sparse_to_dense;
            delete[] dense_to_sparse;
            delete[] is_deleted;
        }
    }
};

namespace node_mapping {
    /**
     * Common Methods
     */
    HOST int to_dense(NodeMapping *node_mapping, int sparse_id);

    HOST int to_sparse(NodeMapping *node_mapping, int dense_id);

    HOST size_t size(NodeMapping *node_mapping);

    HOST size_t active_size(NodeMapping *node_mapping);

    HOST DataBlock<int> get_active_node_ids(NodeMapping *node_mapping);

    HOST void clear(NodeMapping *node_mapping);

    HOST void reserve(NodeMapping *node_mapping, size_t size);

    HOST void mark_node_deleted(NodeMapping *node_mapping, int sparse_id);

    HOST DataBlock<int> get_all_sparse_ids(NodeMapping *node_mapping);

    /**
     * Std Implementations
     */
    HOST void update_std(NodeMapping *node_mapping, EdgeData *edge_data, size_t start_idx, size_t end_idx);

    /**
     * CUDA implementations
     */
    HOST void update_cuda(NodeMapping *node_mapping, EdgeData *edge_data, size_t start_idx, size_t end_idx);

    /**
     * Device functions
     */
    DEVICE int to_dense_device(NodeMapping *node_mapping, int sparse_id);

    DEVICE int to_dense_from_ptr_device(int *sparse_to_dense, int sparse_id, int size);

    DEVICE void mark_node_deleted_from_ptr(bool *is_deleted, int sparse_id, int size);

    DEVICE bool has_node(NodeMapping *node_mapping, int sparse_id);
}

#endif // NODE_MAPPING_H
