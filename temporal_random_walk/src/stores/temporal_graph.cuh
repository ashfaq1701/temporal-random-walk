#ifndef TEMPORAL_GRAPH_STORE_H
#define TEMPORAL_GRAPH_STORE_H

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "node_mapping.cuh"

#include "../data/enums.cuh"

struct TemporalGraphStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data;

    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;
    int node_count_max_bound;

    EdgeDataStore *edge_data;
    NodeEdgeIndexStore *node_edge_index;
    NodeMappingStore *node_mapping;

    int64_t latest_timestamp;

    explicit TemporalGraphStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const int node_count_max_bound):
        is_directed(is_directed), use_gpu(use_gpu), owns_data(true),
        max_time_capacity(max_time_capacity), enable_weight_computation(enable_weight_computation),
        timescale_bound(timescale_bound), node_count_max_bound(node_count_max_bound), latest_timestamp(0) {

        edge_data = new EdgeDataStore(use_gpu);
        node_edge_index = new NodeEdgeIndexStore(use_gpu);
        node_mapping = new NodeMappingStore(node_count_max_bound, use_gpu);
    }

    ~TemporalGraphStore() {
        if (owns_data) {
            #ifdef HAS_CUDA
            if (use_gpu) {
                if (edge_data) clear_memory(&edge_data, use_gpu);
                if (node_mapping) clear_memory(&node_mapping, use_gpu);
                if (node_edge_index) clear_memory(&node_edge_index, use_gpu);
            }
            else
            #endif
            {
                delete edge_data;
                delete node_mapping;
                delete node_edge_index;
            }
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

    HOST void add_multiple_edges_std(TemporalGraphStore* graph, const Edge* new_edges, size_t num_new_edges);

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

    HOST void add_multiple_edges_cuda(TemporalGraphStore* graph, const Edge* new_edges, size_t num_new_edges);

    HOST void sort_and_merge_edges_cuda(TemporalGraphStore* graph, size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraphStore* graph);

    HOST size_t count_timestamps_less_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    #endif

    /**
     * Host functions
     */

    HOST Edge get_edge_at_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward);

    HOST Edge get_node_edge_at_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward);

    /**
     * Device functions
     */

    #ifdef HAS_CUDA

    DEVICE Edge get_edge_at_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        curandState* rand_state);

    DEVICE Edge get_node_edge_at_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        curandState* rand_state);

    HOST TemporalGraphStore* to_device_ptr(const TemporalGraphStore* graph);

    #endif

}

#endif // TEMPORAL_GRAPH_STORE_H
