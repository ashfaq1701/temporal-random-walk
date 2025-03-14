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
    size_t n_threads;
    ThreadPool* thread_pool;

    cudaDeviceProp* cuda_device_prop;
    TemporalGraph* temporal_graph;

    TemporalRandomWalk(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const size_t n_threads): thread_pool(new ThreadPool(n_threads)) {

        this->is_directed = is_directed;
        this->use_gpu = use_gpu;
        this->max_time_capacity = max_time_capacity;
        this->enable_weight_computation = enable_weight_computation;
        this->timescale_bound = timescale_bound;
        this->n_threads = n_threads;

        this->temporal_graph = new TemporalGraph(
            is_directed,
            use_gpu,
            max_time_capacity,
            enable_weight_computation,
            timescale_bound);

        cuda_device_prop = new cudaDeviceProp();
        cudaGetDeviceProperties(cuda_device_prop, 0);
    }
};

namespace temporal_random_walk {

    /**
     * Common functions
     */
    HOST void add_multiple_edges(const TemporalRandomWalk* temporal_random_walk, const Edge* edge_infos, size_t num_edges);

    HOST size_t get_node_count(const TemporalRandomWalk* temporal_random_walk);

    HOST DEVICE size_t get_edge_count(const TemporalRandomWalk* temporal_random_walk);

    HOST DataBlock<int> get_node_ids(const TemporalRandomWalk* temporal_random_walk);

    HOST DataBlock<Edge> get_edges(const TemporalRandomWalk* temporal_random_walk);

    HOST bool get_is_directed(const TemporalRandomWalk* temporal_random_walk);

    HOST void clear(TemporalRandomWalk* temporal_random_walk);

    HOST DEVICE bool get_should_walk_forward(const WalkDirection walk_direction);

    /**
     * Std implementations
     */

    HOST void generate_random_walk_and_time_std(
        const TemporalRandomWalk* temporal_random_walk,
        int walk_idx,
        WalkSet* walk_set,
        const RandomPickerType* edge_picker_type,
        const RandomPickerType* start_picker_type,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id);

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

    __global__ void generate_random_walks_kernel(
        WalkSet* walk_set,
        TemporalGraph* temporal_graph,
        const int* start_node_ids,
        RandomPickerType edge_picker_type,
        RandomPickerType start_picker_type,
        curandState* rand_states,
        int max_walk_len,
        bool is_directed,
        WalkDirection walk_direction,
        int num_walks);

    HOST WalkSet get_random_walks_and_times_for_all_nodes_cuda(
        const TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_cuda(
        const TemporalRandomWalk* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST TemporalRandomWalk* to_device_ptr(const TemporalRandomWalk* graph);

}
#endif // TEMPORAL_RANDOM_WALK_H
