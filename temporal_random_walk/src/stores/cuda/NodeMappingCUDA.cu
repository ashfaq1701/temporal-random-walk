#include "NodeMappingCUDA.cuh"

#include "../../cuda_common/cuda_config.cuh"

#ifdef HAS_CUDA

#include <thrust/extrema.h>
#include <thrust/count.h>

HOST DEVICE int to_dense(const int* sparse_to_dense, const int sparse_id, const int size) {
    return (sparse_id < size) ? sparse_to_dense[sparse_id] : -1;
}

HOST DEVICE void mark_node_deleted(bool* is_deleted, const int sparse_id, const int size) {
    if (sparse_id < size) {
        is_deleted[sparse_id] = true;
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCUDA<GPUUsage>::update(const typename INodeMapping<GPUUsage>::EdgeDataType* edges, size_t start_idx, size_t end_idx) {
    auto max_source = thrust::max_element(
        DEVICE_EXECUTION_POLICY,
        edges->sources.begin() + start_idx,
        edges->sources.begin() + end_idx
    );

    auto max_target = thrust::max_element(
        DEVICE_EXECUTION_POLICY,
        edges->targets.begin() + start_idx,
        edges->targets.begin() + end_idx
    );

    auto max_source_element = *max_source;
    auto max_target_element = *max_target;

    int max_node_id = std::max(
        max_source != edges->sources.end() ? *max_source : 0,
        max_target != edges->targets.end() ? *max_target : 0
    );

    if (max_node_id < 0) {
        return;
    }

    // Extend vectors if needed
    if (max_node_id >= this->sparse_to_dense.size()) {
        this->sparse_to_dense.resize(max_node_id + 1, -1);
        this->is_deleted.resize(max_node_id + 1, true);
    }

    typename SelectVectorType<int, GPUUsage>::type new_node_flags(max_node_id + 1, 0);
    int* new_node_flags_ptr = thrust::raw_pointer_cast(new_node_flags.data());
    int* sparse_to_dense_ptr = thrust::raw_pointer_cast(this->sparse_to_dense.data());
    bool* is_deleted_ptr = thrust::raw_pointer_cast(this->is_deleted.data());
    const int* sources_ptr = thrust::raw_pointer_cast(edges->sources.data());
    const int* targets_ptr = thrust::raw_pointer_cast(edges->targets.data());


    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(start_idx),
        thrust::make_counting_iterator<size_t>(end_idx),
        [new_node_flags_ptr, sparse_to_dense_ptr, is_deleted_ptr, sources_ptr, targets_ptr]
        __host__ __device__ (const size_t idx) {
            const int source = sources_ptr[idx];
            const int target = targets_ptr[idx];

            is_deleted_ptr[source] = false;
            is_deleted_ptr[target] = false;

            if (sparse_to_dense_ptr[source] == -1) {
                new_node_flags_ptr[source] = 1;
            }
            if (sparse_to_dense_ptr[target] == -1) {
                new_node_flags_ptr[target] = 1;
            }
        }
    );

    // Calculate positions for new nodes
    typename SelectVectorType<int, GPUUsage>::type new_node_positions(new_node_flags.size());
    thrust::exclusive_scan(
        DEVICE_EXECUTION_POLICY,
        new_node_flags.begin(),
        new_node_flags.end(),
        new_node_positions.begin()
    );

    // Get total count and resize dense_to_sparse
    size_t old_size = this->dense_to_sparse.size();
    const size_t new_nodes = thrust::reduce(new_node_flags.begin(), new_node_flags.end());
    this->dense_to_sparse.resize(old_size + new_nodes);

    // Get raw pointers for final phase
    int* dense_to_sparse_ptr = thrust::raw_pointer_cast(this->dense_to_sparse.data());
    const int* new_node_positions_ptr = thrust::raw_pointer_cast(new_node_positions.data());

    // Assign dense indices in parallel
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(new_node_flags.size()),
        [sparse_to_dense_ptr, dense_to_sparse_ptr, new_node_flags_ptr, new_node_positions_ptr, old_size]
        __host__ __device__ (const size_t idx) {
            if(new_node_flags_ptr[idx]) {
                const int new_dense_idx = static_cast<int>(old_size) + new_node_positions_ptr[idx];
                sparse_to_dense_ptr[idx] = new_dense_idx;
                dense_to_sparse_ptr[new_dense_idx] = static_cast<int>(idx);
            }
        }
    );
}

template<GPUUsageMode GPUUsage>
DEVICE int NodeMappingCUDA<GPUUsage>::to_dense_device(int sparse_id) const {
    return sparse_id < this->sparse_to_dense_size ? this->sparse_to_dense_ptr[sparse_id] : -1;
}

template<GPUUsageMode GPUUsage>
HOST NodeMappingCUDA<GPUUsage>* NodeMappingCUDA<GPUUsage>::to_device_ptr() {
    // Allocate device memory for the NodeMappingCUDA object
    NodeMappingCUDA<GPUUsage>* device_node_mapping;
    cudaMalloc(&device_node_mapping, sizeof(NodeMappingCUDA<GPUUsage>));

    // Set the pointers and sizes for device vectors directly
    if (!this->sparse_to_dense.empty()) {
        this->sparse_to_dense_ptr = thrust::raw_pointer_cast(this->sparse_to_dense.data());
        this->sparse_to_dense_size = this->sparse_to_dense.size();
    }

    if (!this->dense_to_sparse.empty()) {
        this->dense_to_sparse_ptr = thrust::raw_pointer_cast(this->dense_to_sparse.data());
        this->dense_to_sparse_size = this->dense_to_sparse.size();
    }

    if (!this->is_deleted.empty()) {
        this->is_deleted_ptr = thrust::raw_pointer_cast(this->is_deleted.data());
        this->is_deleted_size = this->is_deleted.size();
    }

    // Copy the object with pointers to the device
    cudaMemcpy(device_node_mapping, this, sizeof(NodeMappingCUDA<GPUUsage>), cudaMemcpyHostToDevice);

    return device_node_mapping;
}

template class NodeMappingCUDA<GPUUsageMode::ON_GPU>;
#endif
