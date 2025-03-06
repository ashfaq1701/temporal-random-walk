#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA

HOST DEVICE int to_dense(const int* sparse_to_dense, const int sparse_id, const int size) {
    return (sparse_id < size) ? sparse_to_dense[sparse_id] : -1;
}

HOST DEVICE void mark_node_deleted(bool* is_deleted, const int sparse_id, const int size) {
    if (sparse_id < size) {
        is_deleted[sparse_id] = true;
    }
}

template class NodeMappingCUDA<GPUUsageMode::ON_GPU>;
#endif
