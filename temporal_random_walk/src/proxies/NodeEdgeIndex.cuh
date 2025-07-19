#ifndef NODE_EDGE_INDEX_H
#define NODE_EDGE_INDEX_H

#include <vector>
#include "../stores/node_edge_index.cuh"
#include "../stores/edge_data.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, size_t group_idx, bool forward, bool is_directed);

__global__ void get_timestamp_group_count_kernel(size_t* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

#endif

class NodeEdgeIndex {
public:

    NodeEdgeIndexStore* node_edge_index;
    bool owns_node_edge_index;

    std::vector<size_t> node_group_outbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_group_outbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_group_outbound_offsets,
                      node_edge_index->node_group_outbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_group_outbound_offsets,
                                     node_edge_index->node_group_outbound_offsets +
                                     node_edge_index->node_group_outbound_offsets_size);
        }
    }

    std::vector<size_t> node_group_inbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_group_inbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_group_inbound_offsets,
                      node_edge_index->node_group_inbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_group_inbound_offsets,
                                     node_edge_index->node_group_inbound_offsets +
                                     node_edge_index->node_group_inbound_offsets_size);
        }
    }

    std::vector<size_t> node_ts_sorted_outbound_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_ts_sorted_outbound_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_ts_sorted_outbound_indices,
                      node_edge_index->node_ts_sorted_outbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_ts_sorted_outbound_indices,
                                     node_edge_index->node_ts_sorted_outbound_indices +
                                     node_edge_index->node_ts_sorted_outbound_indices_size);
        }
    }

    std::vector<size_t> node_ts_sorted_inbound_indices() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_ts_sorted_inbound_indices_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_ts_sorted_inbound_indices,
                      node_edge_index->node_ts_sorted_inbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_ts_sorted_inbound_indices,
                                     node_edge_index->node_ts_sorted_inbound_indices +
                                     node_edge_index->node_ts_sorted_inbound_indices_size);
        }
    }

    std::vector<size_t> count_ts_group_per_node_outbound() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->count_ts_group_per_node_outbound_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->count_ts_group_per_node_outbound,
                      node_edge_index->count_ts_group_per_node_outbound_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->count_ts_group_per_node_outbound,
                                     node_edge_index->count_ts_group_per_node_outbound +
                                     node_edge_index->count_ts_group_per_node_outbound_size);
        }
    }

    std::vector<size_t> count_ts_group_per_node_inbound() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->count_ts_group_per_node_inbound_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->count_ts_group_per_node_inbound,
                      node_edge_index->count_ts_group_per_node_inbound_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->count_ts_group_per_node_inbound,
                                     node_edge_index->count_ts_group_per_node_inbound +
                                     node_edge_index->count_ts_group_per_node_inbound_size);
        }
    }

    std::vector<size_t> node_ts_group_outbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_ts_group_outbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_ts_group_outbound_offsets,
                      node_edge_index->node_ts_group_outbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_ts_group_outbound_offsets,
                                     node_edge_index->node_ts_group_outbound_offsets +
                                     node_edge_index->node_ts_group_outbound_offsets_size);
        }
    }

    std::vector<size_t> node_ts_group_inbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->node_ts_group_inbound_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), node_edge_index->node_ts_group_inbound_offsets,
                      node_edge_index->node_ts_group_inbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(node_edge_index->node_ts_group_inbound_offsets,
                                     node_edge_index->node_ts_group_inbound_offsets +
                                     node_edge_index->node_ts_group_inbound_offsets_size);
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

    explicit NodeEdgeIndex(bool use_gpu);

    explicit NodeEdgeIndex(NodeEdgeIndexStore* existing_node_edge_index);

    ~NodeEdgeIndex();

    NodeEdgeIndex& operator=(const NodeEdgeIndex& other);

    void clear() const;

    void rebuild(EdgeDataStore* edge_data, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;

    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(const EdgeDataStore* edge_data, double timescale_bound) const;

    [[nodiscard]] NodeEdgeIndexStore* get_node_edge_index() const;

    [[nodiscard]] size_t get_memory_used() const;
};

#endif // NODE_EDGE_INDEX_H
