#ifndef TEMPORAL_RANDOM_WALK_STORE_H
#define TEMPORAL_RANDOM_WALK_STORE_H

#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../data/walk_set/walk_set.cuh"
#include "../stores/temporal_graph.cuh"
#include "../utils/utils.cuh"
#include "../utils/random.cuh"
#include "../common/setup.cuh"
#include "../common/random_gen.cuh"
#include "../data/enums.cuh"

struct TemporalRandomWalkStore {
    bool is_directed;
    bool use_gpu;
    bool owns_data = true;
    int64_t max_time_capacity;
    bool enable_weight_computation;
    double timescale_bound;
    int walk_padding_value;

    #ifdef HAS_CUDA
    cudaDeviceProp *cuda_device_prop;
    #endif
    TemporalGraphStore *temporal_graph;

    TemporalRandomWalkStore(
        const bool is_directed,
        const bool use_gpu,
        const int64_t max_time_capacity,
        const bool enable_weight_computation,
        const double timescale_bound,
        const int walk_padding_value=EMPTY_NODE_VALUE) {
        this->is_directed = is_directed;
        this->use_gpu = use_gpu;
        this->max_time_capacity = max_time_capacity;
        this->enable_weight_computation = enable_weight_computation;
        this->timescale_bound = timescale_bound;
        this->walk_padding_value = walk_padding_value;

        this->temporal_graph = new TemporalGraphStore(
            is_directed,
            use_gpu,
            max_time_capacity,
            enable_weight_computation,
            timescale_bound);

        #ifdef HAS_CUDA
        cuda_device_prop = new cudaDeviceProp();
        cudaGetDeviceProperties(cuda_device_prop, 0);
        #endif
    }

    TemporalRandomWalkStore()
        : is_directed(false), use_gpu(false), max_time_capacity(-1),
          enable_weight_computation(false), timescale_bound(-1),
          temporal_graph(nullptr), walk_padding_value(EMPTY_NODE_VALUE) {
        #ifdef HAS_CUDA
        cuda_device_prop = nullptr;
        #endif
    }

    ~TemporalRandomWalkStore() {
        if (owns_data) {
            delete temporal_graph;

            #ifdef HAS_CUDA
            if (use_gpu) {
                clear_memory(&cuda_device_prop, use_gpu);
            } else
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
    HOST void add_multiple_edges(
        const TemporalRandomWalkStore *temporal_random_walk,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_edges);

    HOST size_t get_node_count(const TemporalRandomWalkStore *temporal_random_walk);
    
    HOST DEVICE size_t get_edge_count(const TemporalRandomWalkStore *temporal_random_walk);

    HOST DataBlock<int> get_node_ids(const TemporalRandomWalkStore *temporal_random_walk);

    HOST DataBlock<Edge> get_edges(const TemporalRandomWalkStore *temporal_random_walk);

    HOST bool get_is_directed(const TemporalRandomWalkStore *temporal_random_walk);

    HOST void clear(TemporalRandomWalkStore *temporal_random_walk);

    /**
     * Std implementations
     */

    HOST WalkSet get_random_walks_and_times_for_all_nodes_std(
        const TemporalRandomWalkStore *temporal_random_walk,
        int max_walk_len,
        const RandomPickerType *walk_bias,
        int num_walks_per_node,
        const RandomPickerType *initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_std(
        const TemporalRandomWalkStore *temporal_random_walk,
        int max_walk_len,
        const RandomPickerType *walk_bias,
        int num_walks_total,
        const RandomPickerType *initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    /**
     * CUDA implementations
     */

    #ifdef HAS_CUDA

    HOST WalkSet get_random_walks_and_times_for_all_nodes_cuda(
        const TemporalRandomWalkStore *temporal_random_walk,
        int max_walk_len,
        const RandomPickerType *walk_bias,
        int num_walks_per_node,
        const RandomPickerType *initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalkSet get_random_walks_and_times_cuda(
        const TemporalRandomWalkStore *temporal_random_walk,
        int max_walk_len,
        const RandomPickerType *walk_bias,
        int num_walks_total,
        const RandomPickerType *initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST TemporalRandomWalkStore *to_device_ptr(const TemporalRandomWalkStore* temporal_random_walk);

    HOST void free_device_pointers(TemporalRandomWalkStore* d_temporal_random_walk);

    #endif

    HOST size_t get_memory_used(TemporalRandomWalkStore* temporal_random_walk);

}
#endif // TEMPORAL_RANDOM_WALK_STORE_H
