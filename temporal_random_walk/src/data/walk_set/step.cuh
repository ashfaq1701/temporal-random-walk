#ifndef STEP_CUH
#define STEP_CUH

#include <cstdint>

// Step representation for a node and timestamp pair
struct Step {
    int node;
    int64_t timestamp;
};

#endif // STEP_CUH
