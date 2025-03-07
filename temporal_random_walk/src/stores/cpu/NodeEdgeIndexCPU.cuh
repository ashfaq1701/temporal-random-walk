#ifndef NODEEDGEINDEX_CPU_H
#define NODEEDGEINDEX_CPU_H

#include <cstdint>
#include "../../data/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"
#include "../interfaces/INodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCPU : public INodeEdgeIndex<GPUUsage>
{
public:
    ~NodeEdgeIndexCPU() override = default;

    /**
     * START METHODS FOR REBUILD
     */
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
    /**
     * END METHODS FOR REBUILD
     */

    HOST void compute_temporal_weights(
        const typename INodeEdgeIndex<GPUUsage>::EdgeDataType* edges,
        double timescale_bound,
        size_t num_nodes) override;

    HOST NodeEdgeIndexCPU* to_device_ptr();
};

#endif //NODEEDGEINDEX_CPU_H
