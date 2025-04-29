#ifndef TEMPORAL_RANDOM_WALK_KERNELS_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_CUH

#include "../data/WalkSet.cuh"
#include "../stores/temporal_graph.cuh"

namespace temporal_random_walk {

    #ifdef HAS_CUDA

    __global__ void generate_random_walks_kernel(
        const WalkSet* walk_set,
        TemporalGraphStore* temporal_graph,
        const int* start_node_ids,
        RandomPickerType edge_picker_type,
        RandomPickerType start_picker_type,
        int max_walk_len,
        bool is_directed,
        WalkDirection walk_direction,
        int num_walks,
        const double* rand_nums);

    #endif
}


#endif //TEMPORAL_RANDOM_WALK_KERNELS_CUH
