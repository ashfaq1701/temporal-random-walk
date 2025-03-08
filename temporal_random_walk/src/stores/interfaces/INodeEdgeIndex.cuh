#ifndef I_NODEEDGEINDEX_H
#define I_NODEEDGEINDEX_H

#include "../../data/structs.cuh"
#include "../../data/enums.h"

#include "../cpu/NodeMappingCPU.cuh"
#include "../cuda/NodeMappingCUDA.cuh"

#include "../cpu/EdgeDataCPU.cuh"
#include "../cuda/EdgeDataCUDA.cuh"

template<GPUUsageMode GPUUsage>
class INodeEdgeIndex
{
public:
    virtual ~INodeEdgeIndex() = default;

    #ifdef HAS_CUDA
    using NodeMappingType = std::conditional_t<
        GPUUsage == GPUUsageMode::ON_CPU,
        NodeMappingCPU<GPUUsage>,
        NodeMappingCUDA<GPUUsage>
    >;
    using EdgeDataType = std::conditional_t<
        GPUUsage == GPUUsageMode::ON_CPU,
        EdgeDataCPU<GPUUsage>,
        EdgeDataCUDA<GPUUsage>
    >;
    #else
    using NodeMappingType = NodeMappingCPU<GPUUsage>;
    using EdgeDataType = EdgeDataCPU<GPUUsage>;
    #endif

    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using DoubleVector = typename SelectVectorType<double, GPUUsage>::type;
    using EdgeWithEndpointTypeVector = typename SelectVectorType<EdgeWithEndpointType, GPUUsage>::type;

    // Base CSR format for edges
    SizeVector outbound_offsets{}; // Size: num_nodes + 1
    SizeVector outbound_indices{}; // Size: num_edges

    // CSR format for timestamp groups
    SizeVector outbound_timestamp_group_offsets{}; // Size: num_nodes + 1
    SizeVector outbound_timestamp_group_indices{}; // Each group's start position in outbound_indices

    // Mirror structures for directed graphs
    SizeVector inbound_offsets{};
    SizeVector inbound_indices{};
    SizeVector inbound_timestamp_group_offsets{};
    SizeVector inbound_timestamp_group_indices{};

    DoubleVector outbound_forward_cumulative_weights_exponential{};   // For all forward walks
    DoubleVector outbound_backward_cumulative_weights_exponential{};  // For undirected backward walks
    DoubleVector inbound_backward_cumulative_weights_exponential{};   // For directed backward walks

    /**
    * HOST METHODS
    */
    virtual HOST void clear();

    /**
     * START METHODS FOR REBUILD
     */
    virtual HOST void populate_dense_ids(
        const EdgeDataType* edges,
        const NodeMappingType* mapping,
        IntVector& dense_sources,
        IntVector& dense_targets) {}

    virtual HOST void allocate_node_edge_offsets(size_t num_nodes, bool is_directed);

    virtual HOST void compute_node_edge_offsets(
        const EdgeDataType* edges,
        IntVector& dense_sources,
        IntVector& dense_targets,
        bool is_directed) {}

    virtual HOST void allocate_node_edge_indices(bool is_directed);

    virtual HOST void compute_node_edge_indices(
        const EdgeDataType* edges,
        IntVector& dense_sources,
        IntVector& dense_targets,
        EdgeWithEndpointTypeVector& outbound_edge_indices_buffer,
        bool is_directed) {}

    virtual HOST void compute_node_timestamp_offsets(
        const EdgeDataType* edges,
        size_t num_nodes,
        bool is_directed) {}

    virtual HOST void allocate_node_timestamp_indices(bool is_directed);

    virtual HOST void compute_node_timestamp_indices(
        const EdgeDataType* edges,
        size_t num_nodes,
        bool is_directed) {}
    /**
     * END METHODS FOR REBUILD
     */

    virtual HOST void rebuild(const EdgeDataType* edges, const NodeMappingType* mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] virtual HOST SizeRange get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] virtual HOST SizeRange get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] virtual HOST size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;

    virtual HOST void compute_temporal_weights(
        const EdgeDataType* edges,
        double timescale_bound,
        size_t num_nodes) {}

    virtual HOST void update_temporal_weights(const EdgeDataType* edges, double timescale_bound);



protected:
    virtual HOST SizeVector get_timestamp_offset_vector(bool forward, bool directed) const;
};

#endif //I_NODEEDGEINDEX_H
