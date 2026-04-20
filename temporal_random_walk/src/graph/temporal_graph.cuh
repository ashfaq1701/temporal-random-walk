#ifndef TEMPORAL_GRAPH_CUH
#define TEMPORAL_GRAPH_CUH

#include <cstddef>
#include <vector>

#include "../common/macros.cuh"
#include "../common/const.cuh"
#include "../common/error_handlers.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../data/buffer.cuh"

// Include the OTHER parallel-file headers so temporal_graph_new.cu
// can call the new overloads of edge_data::, node_edge_index::,
// node_features::.
#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "node_features.cuh"

// STAGING FILE for task 5d. Not in CMake. Swapped in by task 5g
// alongside all the other parallel files; the _new suffixes are
// removed from includes at that time.

namespace temporal_graph {

    /**
     * Common
     */
    HOST void update_temporal_weights(TemporalGraphData& data);

    HOST DEVICE size_t get_total_edges(const TemporalGraphData& data);

    HOST size_t get_node_count(const TemporalGraphData& data);

    HOST int64_t get_latest_timestamp(const TemporalGraphData& data);

    HOST std::vector<int> get_node_ids(const TemporalGraphData& data);

    HOST std::vector<Edge> get_edges(const TemporalGraphData& data);

    /**
     * Std implementations
     */
    HOST void sort_and_merge_edges_std(
        TemporalGraphData& data,
        size_t start_idx);

    HOST void delete_old_edges_std(TemporalGraphData& data);

    HOST void add_multiple_edges_std(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0);

    HOST size_t count_timestamps_less_than_std(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_std(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_std(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_std(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    /**
     * CUDA implementations
     */
    #ifdef HAS_CUDA

    HOST void sort_and_merge_edges_cuda(
        TemporalGraphData& data,
        size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraphData& data);

    HOST void add_multiple_edges_cuda(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features = nullptr,
        size_t feature_dim = 0);

    HOST size_t count_timestamps_less_than_cuda(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(
        const TemporalGraphData& data, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(
        const TemporalGraphData& data, int node_id, int64_t timestamp);

    #endif

    HOST size_t get_memory_used(const TemporalGraphData& data);

}

#endif // TEMPORAL_GRAPH_CUH
