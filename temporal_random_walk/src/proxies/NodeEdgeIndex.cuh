#ifndef NODE_EDGE_INDEX_H
#define NODE_EDGE_INDEX_H

#include "../stores/node_edge_index.cuh"
#include "../stores/edge_data.cuh"
#include "../stores/node_mapping.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, size_t group_idx, bool forward, bool is_directed);

__global__ void get_timestamp_group_count_kernel(size_t* result, const NodeEdgeIndexStore* node_edge_index, int dense_node_id, bool forward, bool is_directed);

#endif

class NodeEdgeIndex {
public:

    NodeEdgeIndexStore* node_edge_index;
    bool owns_node_edge_index;

    std::vector<size_t> outbound_offsets() const {
        #ifdef HAS_CUDA
        if (node_edge_index->use_gpu) {
            std::vector<size_t> result(node_edge_index->outbound_offsets_size);
            cudaMemcpy(result.data(), node_edge_index->outbound_offsets,
                      node_edge_index->outbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->inbound_offsets,
                      node_edge_index->inbound_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->outbound_indices,
                      node_edge_index->outbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->inbound_indices,
                      node_edge_index->inbound_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->outbound_timestamp_group_offsets,
                      node_edge_index->outbound_timestamp_group_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->inbound_timestamp_group_offsets,
                      node_edge_index->inbound_timestamp_group_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->outbound_timestamp_group_indices,
                      node_edge_index->outbound_timestamp_group_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->inbound_timestamp_group_indices,
                      node_edge_index->inbound_timestamp_group_indices_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->outbound_forward_cumulative_weights_exponential,
                      node_edge_index->outbound_forward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->outbound_backward_cumulative_weights_exponential,
                      node_edge_index->outbound_backward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost);
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
            cudaMemcpy(result.data(), node_edge_index->inbound_backward_cumulative_weights_exponential,
                      node_edge_index->inbound_backward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost);
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

    void rebuild(EdgeDataStore* edge_data, NodeMappingStore* node_mapping, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;

    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(const EdgeDataStore* edge_data, double timescale_bound) const;

    [[nodiscard]] NodeEdgeIndexStore* get_node_edge_index() const;
};

#endif // NODE_EDGE_INDEX_H
