#ifndef NODE_EDGE_INDEX_H
#define NODE_EDGE_INDEX_H

#include <vector>
#include "../stores/node_edge_index.cuh"
#include "../stores/edge_data.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA

__global__ inline void get_edge_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, const int dense_node_id, const bool forward, const bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_edge_range(node_edge_index, dense_node_id, forward, is_directed);
    }
}

__global__ inline void get_timestamp_group_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, const int dense_node_id, const size_t group_idx, const bool forward, const bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_timestamp_group_range(node_edge_index, dense_node_id, group_idx, forward, is_directed);
    }
}

__global__ inline void get_timestamp_group_count_kernel(size_t* result, const NodeEdgeIndexStore* node_edge_index, const int dense_node_id, const bool forward, const bool is_directed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = node_edge_index::get_timestamp_group_count(node_edge_index, dense_node_id, forward, is_directed);
    }
}

#endif

class NodeEdgeIndex {
public:

    NodeEdgeIndexStore* node_edge_index;
    bool owns_node_edge_index;

    std::vector<size_t> outbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->outbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_offsets,
                      node_edge_index->outbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->outbound_offsets,
                                     node_edge_index->outbound_offsets +
                                     node_edge_index->outbound_offsets_size);
        }
    }

    std::vector<size_t> inbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->inbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->inbound_offsets,
                      node_edge_index->inbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->inbound_offsets,
                                     node_edge_index->inbound_offsets +
                                     node_edge_index->inbound_offsets_size);
        }
    }

    std::vector<size_t> outbound_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->outbound_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_indices,
                      node_edge_index->outbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->outbound_indices,
                                     node_edge_index->outbound_indices +
                                     node_edge_index->outbound_indices_size);
        }
    }

    std::vector<size_t> inbound_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->inbound_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->inbound_indices,
                      node_edge_index->inbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->inbound_indices,
                                     node_edge_index->inbound_indices +
                                     node_edge_index->inbound_indices_size);
        }
    }

    std::vector<size_t> outbound_timestamp_group_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->outbound_timestamp_group_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_timestamp_group_offsets,
                      node_edge_index->outbound_timestamp_group_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->outbound_timestamp_group_offsets,
                                     node_edge_index->outbound_timestamp_group_offsets +
                                     node_edge_index->outbound_timestamp_group_offsets_size);
        }
    }

    std::vector<size_t> inbound_timestamp_group_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->inbound_timestamp_group_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->inbound_timestamp_group_offsets,
                      node_edge_index->inbound_timestamp_group_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->inbound_timestamp_group_offsets,
                                     node_edge_index->inbound_timestamp_group_offsets +
                                     node_edge_index->inbound_timestamp_group_offsets_size);
        }
    }

    std::vector<size_t> outbound_timestamp_group_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->outbound_timestamp_group_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_timestamp_group_indices,
                      node_edge_index->outbound_timestamp_group_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->outbound_timestamp_group_indices,
                                     node_edge_index->outbound_timestamp_group_indices +
                                     node_edge_index->outbound_timestamp_group_indices_size);
        }
    }

    std::vector<size_t> inbound_timestamp_group_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->inbound_timestamp_group_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->inbound_timestamp_group_indices,
                      node_edge_index->inbound_timestamp_group_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->inbound_timestamp_group_indices,
                                     node_edge_index->inbound_timestamp_group_indices +
                                     node_edge_index->inbound_timestamp_group_indices_size);
        }
    }

    std::vector<double> outbound_forward_cumulative_weights_exponential() const {
        if (!node_edge_index->outbound_forward_cumulative_weights_exponential) {
            return std::vector<double>();
        }

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<double> result(node_edge_index->outbound_forward_cumulative_weights_exponential_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_forward_cumulative_weights_exponential,
                      node_edge_index->outbound_forward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<double>(
                node_edge_index->outbound_forward_cumulative_weights_exponential,
                node_edge_index->outbound_forward_cumulative_weights_exponential +
                node_edge_index->outbound_forward_cumulative_weights_exponential_size);
        }
    }

    std::vector<double> outbound_backward_cumulative_weights_exponential() const {
        if (!node_edge_index->outbound_backward_cumulative_weights_exponential) {
            return std::vector<double>();
        }

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<double> result(node_edge_index->outbound_backward_cumulative_weights_exponential_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->outbound_backward_cumulative_weights_exponential,
                      node_edge_index->outbound_backward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<double>(
                node_edge_index->outbound_backward_cumulative_weights_exponential,
                node_edge_index->outbound_backward_cumulative_weights_exponential +
                node_edge_index->outbound_backward_cumulative_weights_exponential_size);
        }
    }

    std::vector<double> inbound_backward_cumulative_weights_exponential() const {
        if (!node_edge_index->inbound_backward_cumulative_weights_exponential) {
            return std::vector<double>();
        }

        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<double> result(node_edge_index->inbound_backward_cumulative_weights_exponential_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->inbound_backward_cumulative_weights_exponential,
                      node_edge_index->inbound_backward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<double>(
                node_edge_index->inbound_backward_cumulative_weights_exponential,
                node_edge_index->inbound_backward_cumulative_weights_exponential +
                node_edge_index->inbound_backward_cumulative_weights_exponential_size);
        }
    }

    explicit NodeEdgeIndex(const bool use_gpu): owns_node_edge_index(true) {
        node_edge_index = new NodeEdgeIndexStore(use_gpu);
    }

    explicit NodeEdgeIndex(NodeEdgeIndexStore* existing_node_edge_index)
        : node_edge_index(existing_node_edge_index), owns_node_edge_index(false) {}

    ~NodeEdgeIndex() {
        if (owns_node_edge_index && node_edge_index) {
            delete node_edge_index;
        }
    }

    NodeEdgeIndex& operator=(const NodeEdgeIndex& other) {
        if (this != &other) {
            if (owns_node_edge_index && node_edge_index) {
                delete node_edge_index;
            }

            owns_node_edge_index = other.owns_node_edge_index;
            if (other.owns_node_edge_index) {
                node_edge_index = new NodeEdgeIndexStore(other.node_edge_index->use_gpu);
            } else {
                node_edge_index = other.node_edge_index;
            }
        }
        return *this;
    }

    void clear() const {
        node_edge_index::clear(node_edge_index);
    }

    void rebuild(EdgeDataStore* edge_data, const bool is_directed) const {
        node_edge_index::rebuild(node_edge_index, edge_data, is_directed);
    }

    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(const int dense_node_id, const bool forward, const bool is_directed) const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            // Call via CUDA kernel for GPU implementation
            SizeRange* d_result;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(SizeRange)));

            NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
            get_edge_range_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, forward, is_directed);
            CUDA_KERNEL_CHECK("After get_edge_range_kernel execution");

            SizeRange host_result;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost));

            CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
            CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

            return {host_result.from, host_result.to};
        }
        else
        #endif
        {
            // Direct call for CPU implementation
            SizeRange result = node_edge_index::get_edge_range(node_edge_index, dense_node_id, forward, is_directed);
            return {result.from, result.to};
        }
    }

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(const int dense_node_id, const size_t group_idx, const bool forward, const bool is_directed) const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            // Call via CUDA kernel for GPU implementation
            SizeRange* d_result;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(SizeRange)));

            NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
            get_timestamp_group_range_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, group_idx, forward, is_directed);
            CUDA_KERNEL_CHECK("After get_timestamp_group_range_kernel execution");

            SizeRange host_result;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost));

            CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
            CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

            return {host_result.from, host_result.to};
        }
        else
        #endif
        {
            // Direct call for CPU implementation
            SizeRange result = node_edge_index::get_timestamp_group_range(node_edge_index, dense_node_id, group_idx, forward, is_directed);
            return {result.from, result.to};
        }
    }

    [[nodiscard]] size_t get_timestamp_group_count(const int dense_node_id, const bool forward, const bool is_directed) const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            // Call via CUDA kernel for GPU implementation
            size_t* d_result;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(size_t)));

            NodeEdgeIndexStore* d_node_edge_index = node_edge_index::to_device_ptr(node_edge_index);
            get_timestamp_group_count_kernel<<<1, 1>>>(d_result, d_node_edge_index, dense_node_id, forward, is_directed);
            CUDA_KERNEL_CHECK("After get_timestamp_group_count_kernel execution");

            size_t host_result;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost));

            CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
            CUDA_CHECK_AND_CLEAR(cudaFree(d_node_edge_index));

            return host_result;
        }
        else
        #endif
        {
            // Direct call for CPU implementation
            return node_edge_index::get_timestamp_group_count(node_edge_index, dense_node_id, forward, is_directed);
        }
    }

    void update_temporal_weights(const EdgeDataStore* edge_data, double timescale_bound) const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            node_edge_index::update_temporal_weights_cuda(node_edge_index, edge_data, timescale_bound);
        }
        else
        #endif
        {
            node_edge_index::update_temporal_weights_std(node_edge_index, edge_data, timescale_bound);
        }
    }

    [[nodiscard]] NodeEdgeIndexStore* get_node_edge_index() const {
        return node_edge_index;
    }
};

#endif // NODE_EDGE_INDEX_H
