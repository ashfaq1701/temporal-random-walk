#ifndef NODE_MAPPING_H
#define NODE_MAPPING_H

#include <cstddef>

struct NodeMapping {
    bool use_gpu;

    int* sparse_to_dense = nullptr;
    size_t sparse_to_dense_size = 0;

    int* dense_to_sparse = nullptr;
    size_t dense_to_sparse_size = 0;

    bool* is_deleted = nullptr;
    size_t is_deleted_size = 0;

    explicit NodeMapping(const bool use_gpu): use_gpu(use_gpu) {}
};

namespace node_mapping {

}

#endif // NODE_MAPPING_H
