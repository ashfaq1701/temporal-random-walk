#ifndef TEMPORAL_GRAPH_STORE_H
#define TEMPORAL_GRAPH_STORE_H

#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "../common/const.cuh"

struct TemporalGraphStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data;

    int64_t max_time_capacity;
    bool enable_weight_computation;
    bool enable_temporal_node2vec = false;
    double timescale_bound;
    double node2vec_p;
    double node2vec_q;
    double inv_p;
    double inv_q;

    EdgeDataStore *edge_data;
    NodeEdgeIndexStore *node_edge_index;

    int64_t latest_timestamp;

    TemporalGraphStore(): is_directed(false), use_gpu(false), owns_data(true) {
        max_time_capacity = 0;
        enable_weight_computation = false;
        enable_temporal_node2vec = false;
        timescale_bound = -1;
        node2vec_p = DEFAULT_NODE2VEC_P;
        node2vec_q = DEFAULT_NODE2VEC_Q;
        inv_p = 1.0 / node2vec_p;
        inv_q = 1.0 / node2vec_q;
        edge_data = nullptr;
        node_edge_index = nullptr;
        latest_timestamp = 0;
    }

    explicit TemporalGraphStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const double node2vec_p = DEFAULT_NODE2VEC_P,
        const double node2vec_q = DEFAULT_NODE2VEC_Q,
        const bool enable_temporal_node2vec = false):
        is_directed(is_directed), use_gpu(use_gpu), owns_data(true),
        max_time_capacity(max_time_capacity),
        enable_weight_computation(enable_weight_computation || enable_temporal_node2vec),
        enable_temporal_node2vec(enable_temporal_node2vec),
        timescale_bound(timescale_bound),
        node2vec_p(node2vec_p), node2vec_q(node2vec_q), inv_p(1.0 / node2vec_p),
        inv_q(1.0 / node2vec_q), latest_timestamp(0) {

        edge_data = new EdgeDataStore(use_gpu);
        node_edge_index = new NodeEdgeIndexStore(use_gpu);

        edge_data->enable_weight_computation = this->enable_weight_computation;
        edge_data->enable_temporal_node2vec = this->enable_temporal_node2vec;
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

    HOST void sort_and_merge_edges_std(TemporalGraphStore *graph, size_t start_idx);

    HOST void delete_old_edges_std(TemporalGraphStore* graph);

    HOST void add_multiple_edges_std(
        TemporalGraphStore *graph,
        const int *sources,
        const int *targets,
        const int64_t *timestamps,
        size_t num_new_edges);

    HOST size_t count_timestamps_less_than_std(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_std(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_std(TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_std(TemporalGraphStore* graph, int node_id, int64_t timestamp);

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA

    HOST void sort_and_merge_edges_cuda(TemporalGraphStore *graph, size_t start_idx);

    HOST void delete_old_edges_cuda(TemporalGraphStore *graph);

    HOST void add_multiple_edges_cuda(
        TemporalGraphStore *graph,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges);

    HOST size_t count_timestamps_less_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int64_t timestamp);

    HOST size_t count_node_timestamps_less_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST size_t count_node_timestamps_greater_than_cuda(const TemporalGraphStore* graph, int node_id, int64_t timestamp);

    HOST TemporalGraphStore* to_device_ptr(const TemporalGraphStore* graph);

    HOST void free_device_pointers(TemporalGraphStore* d_graph);

    #endif

    HOST size_t get_memory_used(TemporalGraphStore* graph);

}

#endif // TEMPORAL_GRAPH_STORE_H
