#ifndef TEMPORAL_RANDOM_WALK_STORE_H
#define TEMPORAL_RANDOM_WALK_STORE_H

#include "../stores/temporal_graph.cuh"
#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../../libs/thread-pool/ThreadPool.h"

struct TemporalRandomWalkStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data = true;
    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;
    int node_count_max_bound;
    size_t n_threads;
    ThreadPool* thread_pool;

    #ifdef HAS_CUDA
    cudaDeviceProp* cuda_device_prop;
    #endif
    TemporalGraphStore* temporal_graph;

    TemporalRandomWalkStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const int node_count_max_bound,
        const size_t n_threads): thread_pool(new ThreadPool(n_threads)) {

        this->is_directed = is_directed;
        this->use_gpu = use_gpu;
        this->max_time_capacity = max_time_capacity;
        this->enable_weight_computation = enable_weight_computation;
        this->timescale_bound = timescale_bound;
        this->node_count_max_bound = node_count_max_bound;
        this->n_threads = n_threads;

        this->temporal_graph = new TemporalGraphStore(
            is_directed,
            use_gpu,
            max_time_capacity,
            enable_weight_computation,
            timescale_bound,
            node_count_max_bound);

        #ifdef HAS_CUDA
        cuda_device_prop = new cudaDeviceProp();
        cudaGetDeviceProperties(cuda_device_prop, 0);
        #endif
    }

    ~TemporalRandomWalkStore() {
        if (owns_data) {
            delete temporal_graph;

            #ifdef HAS_CUDA
            if (use_gpu) {
                clear_memory(&cuda_device_prop, use_gpu);
            }
            else
            #endif
            {
                #ifdef HAS_CUDA
                delete cuda_device_prop;
                #endif
            }
        }
    }
};

namespace temporal_random_walk {

    /**
     * Common functions
     */
    HOST void add_multiple_edges(const TemporalRandomWalkStore* temporal_random_walk, const Edge* edge_infos, size_t num_edges);

    HOST size_t get_node_count(const TemporalRandomWalkStore* temporal_random_walk);

    HOST DEVICE size_t get_edge_count(const TemporalRandomWalkStore* temporal_random_walk);

    HOST DataBlock<int> get_node_ids(const TemporalRandomWalkStore* temporal_random_walk);

    HOST DataBlock<Edge> get_edges(const TemporalRandomWalkStore* temporal_random_walk);

    HOST bool get_is_directed(const TemporalRandomWalkStore* temporal_random_walk);

    HOST void clear(TemporalRandomWalkStore* temporal_random_walk);

    HOST DEVICE bool get_should_walk_forward(const WalkDirection walk_direction);

    /**
     * Std implementations
     */

    HOST void generate_random_walk_and_time_std(
        const TemporalRandomWalkStore* temporal_random_walk,
        int walk_idx,
        WalkSet* walk_set,
        const RandomPickerType* edge_picker_type,
        const RandomPickerType* start_picker_type,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id);

    HOST WalkSet get_random_walks_and_times_for_all_nodes_std(
        TemporalRandomWalkStore* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_std(
        TemporalRandomWalkStore* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA

    __global__ void generate_random_walks_kernel(
        WalkSet* walk_set,
        TemporalGraphStore* temporal_graph,
        const int* start_node_ids,
        RandomPickerType edge_picker_type,
        RandomPickerType start_picker_type,
        curandState* rand_states,
        int max_walk_len,
        bool is_directed,
        WalkDirection walk_direction,
        int num_walks);

    HOST WalkSet get_random_walks_and_times_for_all_nodes_cuda(
        const TemporalRandomWalkStore* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_cuda(
        const TemporalRandomWalkStore* temporal_random_walk,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    HOST TemporalRandomWalkStore* to_device_ptr(const TemporalRandomWalkStore* graph);

    #endif

}
#endif // TEMPORAL_RANDOM_WALK_STORE_H
