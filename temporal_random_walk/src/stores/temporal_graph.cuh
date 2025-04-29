#ifndef TEMPORAL_GRAPH_STORE_H
#define TEMPORAL_GRAPH_STORE_H

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

#include "edge_data.cuh"
#include "node_edge_index.cuh"

#include "../data/enums.cuh"

struct TemporalGraphStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data;

    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;

    EdgeDataStore *edge_data;
    NodeEdgeIndexStore *node_edge_index;

    int64_t latest_timestamp;

    TemporalGraphStore(): is_directed(false), use_gpu(false), owns_data(true) {
        max_time_capacity = 0;
        enable_weight_computation = false;
        timescale_bound = -1;
        edge_data = nullptr;
        node_edge_index = nullptr;
        latest_timestamp = 0;
    }

    explicit TemporalGraphStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound):
        is_directed(is_directed), use_gpu(use_gpu), owns_data(true),
        max_time_capacity(max_time_capacity), enable_weight_computation(enable_weight_computation),
        timescale_bound(timescale_bound), latest_timestamp(0) {

        edge_data = new EdgeDataStore(use_gpu);
        node_edge_index = new NodeEdgeIndexStore(use_gpu);
    }

    ~TemporalGraphStore() {
        if (owns_data) {
            delete edge_data;
            delete node_edge_index;
        }
    }
};

namespace temporal_graph {

    /**
     * Common functions
     */

    HOST void update_temporal_weights(const TemporalGraphStore* graph);

    HOST DEVICE size_t get_total_edges(const TemporalGraphStore* graph);

    HOST size_t get_node_count(const TemporalGraphStore* graph);

    HOST int64_t get_latest_timestamp(const TemporalGraphStore* graph);

    HOST DataBlock<int> get_node_ids(const TemporalGraphStore* graph);

    HOST DataBlock<Edge> get_edges(const TemporalGraphStore* graph);

    /**
     * Std implementations
     */

    HOST void add_multiple_edges_std(
        TemporalGraphStore* graph,
        const Edge* new_edges,
        size_t num_new_edges);

    HOST void sort_and_merge_edges_std(TemporalGraphStore* graph, size_t start_idx);

    HOST void delete_old_edges_std(TemporalGraphStore* graph);

    HOST size_t count_timestamps_less_than_std(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_std(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_std(TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_std(TemporalGraphStore* graph, int node_id, int64_t timestamp);

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA

    HOST void add_multiple_edges_cuda(
        TemporalGraphStore* graph,
        const Edge* new_edges,
        size_t num_new_edges);

    HOST void sort_and_merge_edges_cuda(TemporalGraphStore* graph, size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraphStore* graph);

    HOST size_t count_timestamps_less_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST TemporalGraphStore* to_device_ptr(const TemporalGraphStore* graph);

    HOST void free_device_pointers(TemporalGraphStore* d_graph);

    #endif

}

#endif // TEMPORAL_GRAPH_STORE_H
