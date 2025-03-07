#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../../data/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public INodeEdgeIndex<GPUUsage> {
public:
    #ifdef HAS_CUDA

    size_t* outbound_offsets_ptr = nullptr;
    size_t outbound_offsets_size = 0;
    size_t* outbound_indices_ptr = nullptr;
    size_t outbound_indices_size = 0;

    size_t* outbound_timestamp_group_offsets_ptr = nullptr;
    size_t outbound_timestamp_group_offsets_size = 0;
    size_t* outbound_timestamp_group_indices_ptr = nullptr;
    size_t outbound_timestamp_group_indices_size = 0;

    size_t* inbound_offsets_ptr = nullptr;
    size_t inbound_offsets_size = 0;
    size_t* inbound_indices_ptr = nullptr;
    size_t inbound_indices_size = 0;
    size_t* inbound_timestamp_group_offsets_ptr = nullptr;
    size_t inbound_timestamp_group_offsets_size = 0;
    size_t* inbound_timestamp_group_indices_ptr = nullptr;
    size_t inbound_timestamp_group_indices_size = 0;

    double* outbound_forward_cumulative_weights_exponential_ptr = nullptr;
    size_t outbound_forward_cumulative_weights_exponential_size = 0;
    double* outbound_backward_cumulative_weights_exponential_ptr = nullptr;
    size_t outbound_backward_cumulative_weights_exponential_size = 0;
    double* inbound_backward_cumulative_weights_exponential_ptr = nullptr;
    size_t inbound_backward_cumulative_weights_exponential_size = 0;

    HOST void populate_dense_ids(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        const typename INodeEdgeIndex<GPUUsage>::NodeMappingType* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets) override;

    HOST void compute_node_edge_offsets(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        bool is_directed) override;

    HOST void compute_node_edge_indices(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        typename INodeEdgeIndex<GPUUsage>::EdgeWithEndpointTypeVector& outbound_edge_indices_buffer,
        bool is_directed) override;

    HOST void compute_node_timestamp_offsets(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        size_t num_nodes,
        bool is_directed) override;

    HOST void compute_node_timestamp_indices(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        size_t num_nodes,
        bool is_directed) override;

    HOST void compute_temporal_weights(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        double timescale_bound,
        size_t num_nodes) override;

    HOST NodeEdgeIndexCUDA* to_device_ptr();

    #endif
};

#endif //NODEEDGEINDEXCUDA_H
