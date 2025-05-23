#ifndef TEMPORAL_RANDOM_WALK_KERNELS_LAUNCHER_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_LAUNCHER_CUH

#include "temporal_random_walk_kernels_full_walk.cuh"
#include "temporal_random_walk_kernels_step_based.cuh"

namespace temporal_random_walk {
    #ifdef HAS_CUDA

    inline void launch_random_walk_kernel(
        TemporalGraphStore *temporal_graph,
        const bool is_directed,
        const WalkSet *walk_set,
        const int max_walk_len,
        const int *start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const double *rand_nums,
        const dim3 &grid_dim,
        const dim3 &block_dim) {
        if (max_walk_len <= RANDOM_WALK_FULL_WALK_SWITCH_THRESHOLD) {
            launch_random_walk_kernel_step_based(
                temporal_graph,
                is_directed,
                walk_set,
                max_walk_len,
                start_node_ids,
                num_walks,
                edge_picker_type,
                start_picker_type,
                walk_direction,
                rand_nums,
                grid_dim,
                block_dim);
        } else {
            launch_random_walk_kernel_full_walk(
                temporal_graph,
                is_directed,
                walk_set,
                max_walk_len,
                start_node_ids,
                num_walks,
                edge_picker_type,
                start_picker_type,
                walk_direction,
                rand_nums,
                grid_dim,
                block_dim);
        }
    }

    #endif

}

#endif //TEMPORAL_RANDOM_WALK_KERNELS_LAUNCHER_CUH
