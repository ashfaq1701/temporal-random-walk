#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../../data/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public INodeEdgeIndex<GPUUsage> {
public:
    #ifdef HAS_CUDA

    HOST void populate_dense_ids(
        const IEdgeData<GPUUsage>* edges,
        const INodeMapping<GPUUsage>* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets) override;

    HOST void compute_node_edge_offsets(
        const IEdgeData<GPUUsage>* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        bool is_directed) override;

    HOST void compute_node_edge_indices(
        const IEdgeData<GPUUsage>* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        typename INodeEdgeIndex<GPUUsage>::EdgeWithEndpointTypeVector& outbound_edge_indices_buffer,
        bool is_directed) override;

    HOST void compute_node_timestamp_offsets(
        const IEdgeData<GPUUsage>* edges,
        size_t num_nodes,
        bool is_directed) override;

    HOST void compute_node_timestamp_indices(
        const IEdgeData<GPUUsage>* edges,
        size_t num_nodes,
        bool is_directed) override;

    HOST void compute_temporal_weights(
        const IEdgeData<GPUUsage>* edges,
        double timescale_bound,
        size_t num_nodes) override;

    #endif
};

#endif //NODEEDGEINDEXCUDA_H
