#ifndef I_TEMPORALGRAPH_H
#define I_TEMPORALGRAPH_H

#include <cstdint>

#include "../../cuda_common/types.cuh"
#include "../../data/structs.cuh"
#include "../../data/enums.h"

#include "../../random/RandomPicker.h"

#include "../cpu/NodeMappingCPU.cuh"
#include "../cuda/NodeMappingCUDA.cuh"

#include "../cpu/EdgeDataCPU.cuh"
#include "../cuda/EdgeDataCUDA.cuh"

#include "../cpu/NodeEdgeIndexCPU.cuh"
#include "../cuda/NodeEdgeIndexCUDA.cuh"

template<GPUUsageMode GPUUsage>
class ITemporalGraph
{
public:
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
    using NodeEdgeIndexType = std::conditional_t<
        GPUUsage == GPUUsageMode::ON_CPU,
        NodeEdgeIndexCPU<GPUUsage>,
        NodeEdgeIndexCUDA<GPUUsage>
    >;
    #else
    using NodeMappingType = NodeMappingCPU<GPUUsage>;
    using EdgeDataType = EdgeDataCPU<GPUUsage>;
    using NodeEdgeIndexType = NodeEdgeIndexCPU<GPUUsage>;
    #endif

    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64TVector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

    using EdgeVector = typename SelectVectorType<Edge, GPUUsage>::type;

    int64_t time_window = -1; // Time duration to keep edges (-1 means keep all)
    bool enable_weight_computation = false;
    double timescale_bound = -1;
    int64_t latest_timestamp = 0; // Track latest edge timestamp

    virtual ~ITemporalGraph()
    {
        delete node_index;
        delete edges;
        delete node_mapping;
    };

    bool is_directed = false;

    NodeEdgeIndexType* node_index = nullptr; // Node to edge mappings
    EdgeDataType* edges = nullptr; // Main edge storage
    NodeMappingType* node_mapping = nullptr; // Sparse to dense node ID mapping

    explicit ITemporalGraph(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1)
        : is_directed(directed)
            , time_window(window)
            , enable_weight_computation(enable_weight_computation)
            , timescale_bound(timescale_bound) {}

    /**
    * HOST METHODS
    */
    // Edge addition
    virtual HOST void add_multiple_edges(const EdgeVector& new_edges) {}

    virtual HOST void sort_and_merge_edges(size_t start_idx) {}

    virtual HOST void update_temporal_weights();

    virtual HOST void delete_old_edges() {}

    // Timestamp group counting
    [[nodiscard]] virtual HOST size_t count_timestamps_less_than(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_timestamps_greater_than(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const { return 0; }

    // Edge selection
    [[nodiscard]] virtual HOST Edge get_edge_at_host(
        RandomPicker<GPUUsage>* picker, int64_t timestamp = -1,
        bool forward = true) const { return Edge{-1, -1, -1}; }

    [[nodiscard]] virtual HOST Edge get_node_edge_at_host(int node_id,
                                                                 RandomPicker<GPUUsage>* picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const { return Edge{-1, -1, -1}; }

    // Utility methods
    [[nodiscard]] virtual HOST size_t get_total_edges() const;
    [[nodiscard]] virtual HOST size_t get_node_count() const;
    [[nodiscard]] virtual HOST int64_t get_latest_timestamp();
    [[nodiscard]] virtual HOST IntVector get_node_ids() const;
    [[nodiscard]] virtual HOST EdgeVector get_edges();
};

#endif //I_TEMPORALGRAPH_H
