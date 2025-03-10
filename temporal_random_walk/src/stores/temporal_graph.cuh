#ifndef TEMPORAL_GRAPH_H
#define TEMPORAL_GRAPH_H

#include <curand_kernel.h>

#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "node_mapping.cuh"

#include "../data/enums.cuh"

struct TemporalGraph {
    bool is_directed;
    bool use_gpu;

    EdgeData* edge_data;
    NodeEdgeIndex* node_edge_index;
    NodeMapping* node_mapping;

    explicit TemporalGraph(const bool is_directed, const bool use_gpu): is_directed(is_directed), use_gpu(use_gpu) {
        edge_data = new EdgeData(use_gpu);
        node_edge_index = new NodeEdgeIndex(use_gpu);
        node_mapping = new NodeMapping(use_gpu);
    }
};

namespace temporal_graph {

    /**
     * Common functions
     */

    HOST void update_temporal_weights(TemporalGraph* graph);

    HOST size_t get_total_edges(TemporalGraph* graph);

    HOST size_t get_node_count(TemporalGraph* graph);

    HOST int64_t get_latest_timestamp(TemporalGraph* graph);

    HOST DataBlock<int> get_node_ids(TemporalGraph* graph);

    HOST DataBlock<Edge> get_edges(TemporalGraph* graph);

    /**
     * Std implementations
     */

    HOST void add_multiple_edges_std(TemporalGraph* graph, Edge* new_edges, size_t num_new_edges);

    HOST void sort_and_merge_edges_std(TemporalGraph* graph, size_t start_idx);

    HOST void delete_old_edges_std(TemporalGraph* graph);

    HOST size_t count_timestamps_less_than_std(TemporalGraph* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_std(TemporalGraph* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_std(TemporalGraph* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_std(TemporalGraph* graph, int node_id, int64_t timestamp);

    /**
     * CUDA implementations
     */

    HOST void add_multiple_edges_cuda(TemporalGraph* graph, Edge* new_edges, size_t num_new_edges);

    HOST void sort_and_merge_edges_cuda(TemporalGraph* graph, size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraph* graph);

    HOST size_t count_timestamps_less_than_cuda(TemporalGraph* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(TemporalGraph* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(TemporalGraph* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(TemporalGraph* graph, int node_id, int64_t timestamp);

    /**
     * Host functions
     */

    HOST Edge get_edge_at_host(TemporalGraph* graph, RandomPickerType picker_type, int64_t timestamp, bool forward);

    HOST Edge get_node_edge_at_host(TemporalGraph* graph, int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward);

    /**
     * Device functions
     */

    DEVICE Edge get_edge_at_device(TemporalGraph* graph, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);

    DEVICE Edge get_node_edge_at_device(TemporalGraph* graph, int node_id, RandomPickerType picker_type, int64_t timestamp, bool forward, curandState* rand_state);

}

#endif // TEMPORAL_GRAPH_H
