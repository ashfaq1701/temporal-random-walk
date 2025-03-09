#include "INodeMapping.cuh"

template<GPUUsageMode GPUUsage>
HOST void INodeMapping<GPUUsage>::clear() {
    this->sparse_to_dense.clear();
    this->dense_to_sparse.clear();
    this->is_deleted.clear();
}

template<GPUUsageMode GPUUsage>
HOST void INodeMapping<GPUUsage>::reserve(const size_t size) {
    this->sparse_to_dense.reserve(size);
    this->dense_to_sparse.reserve(size);
    this->is_deleted.reserve(size);
}

template<GPUUsageMode GPUUsage>
HOST void INodeMapping<GPUUsage>::mark_node_deleted(const int sparse_id) {
    if (sparse_id < this->is_deleted.size()) {
        this->is_deleted[sparse_id] = true;
    }
}

template<GPUUsageMode GPUUsage>
HOST int INodeMapping<GPUUsage>::to_dense(const int sparse_id) const {
    return sparse_id < this->sparse_to_dense.size() ? this->sparse_to_dense[sparse_id] : -1;
}

template<GPUUsageMode GPUUsage>
HOST int INodeMapping<GPUUsage>::to_sparse(const int dense_idx) const {
    return dense_idx < this->dense_to_sparse.size() ? this->dense_to_sparse[dense_idx] : -1;
}

template<GPUUsageMode GPUUsage>
HOST size_t INodeMapping<GPUUsage>::size() const {
    return this->dense_to_sparse.size();
}

template<GPUUsageMode GPUUsage>
HOST size_t INodeMapping<GPUUsage>::active_size() const {
    return std::count(this->is_deleted.begin(), this->is_deleted.end(), false);
}

template<GPUUsageMode GPUUsage>
HOST typename INodeMapping<GPUUsage>::IntVector INodeMapping<GPUUsage>::get_active_node_ids() const {
    IntVector active_ids;
    active_ids.reserve(this->dense_to_sparse.size());
    for (int sparse_id : this->dense_to_sparse) {
        if (!this->is_deleted[sparse_id]) {
            active_ids.push_back(sparse_id);
        }
    }

    return active_ids;
}

template<GPUUsageMode GPUUsage>
HOST bool INodeMapping<GPUUsage>::has_node(int sparse_id) const {
    return sparse_id < this->sparse_to_dense.size() && this->sparse_to_dense[sparse_id] != -1;
}

template<GPUUsageMode GPUUsage>
HOST typename INodeMapping<GPUUsage>::IntVector INodeMapping<GPUUsage>::get_all_sparse_ids() const {
    return this->dense_to_sparse;
}

template class INodeMapping<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class INodeMapping<GPUUsageMode::ON_GPU>;
#endif

