#ifndef WALK_SET_STEP_CUH
#define WALK_SET_STEP_CUH

#include "common.cuh"

// Step representation for a node and timestamp pair
struct Step {
    int node;
    int64_t timestamp;
};

#endif // WALK_SET_STEP_CUH
