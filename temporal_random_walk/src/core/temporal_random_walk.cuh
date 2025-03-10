#ifndef TEMPORAL_RANDOM_WALK_H
#define TEMPORAL_RANDOM_WALK_H

#include "../stores/temporal_graph.cuh"
#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../../libs/thread-pool/ThreadPool.h"

struct TemporalRandomWalk {
    bool is_directed;
    bool use_gpu;
    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;
    ThreadPool thread_pool;

    TemporalGraph* temporal_graph;

    TemporalRandomWalk(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const size_t n_threads): thread_pool(n_threads) {

        this->is_directed = is_directed;
        this->use_gpu = use_gpu;
        this->max_time_capacity = max_time_capacity;
        this->enable_weight_computation = enable_weight_computation;
        this->timescale_bound = timescale_bound;

        this->temporal_graph = new TemporalGraph(
            is_directed,
            use_gpu,
            max_time_capacity,
            enable_weight_computation,
            timescale_bound);
    }
};

namespace temporal_random_walk {

    /**
     * Common functions
     */
    HOST void add_multiple_edges(const Edge* edge_infos, size_t num_edges);

    HOST size_t get_node_count(TemporalRandomWalk* temporal_random_walk);

    HOST size_t get_edge_count(TemporalRandomWalk* temporal_random_walk);

    HOST DataBlock<int> get_node_ids(TemporalRandomWalk* temporal_random_walk);

    HOST DataBlock<Edge> get_edges(TemporalRandomWalk* temporal_random_walk);

    HOST bool get_is_directed(TemporalRandomWalk* temporal_random_walk);

    HOST void clear(TemporalRandomWalk* temporal_random_walk);

    /**
     * Std implementations
     */

    HOST WalkSet get_random_walks_and_times_for_all_nodes_std(
        TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_std(
        TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * CUDA implementations
     */

    HOST WalkSet get_random_walks_and_times_for_all_nodes_cuda(
        TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_cuda(
        TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

}

#endif // TEMPORAL_RANDOM_WALK_H
