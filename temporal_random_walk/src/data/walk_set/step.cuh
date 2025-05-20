#ifndef STEP_CUH
#define STEP_CUH

#include <cstdint>

// Step representation for a node and timestamp pair
struct Step {
    int node;
    int64_t timestamp;

    HOST DEVICE Step() : node(-1), timestamp(-1) {}

    HOST DEVICE Step(const int n, const int64_t ts) : node(n), timestamp(ts) {}

    HOST DEVICE bool operator==(const Step& other) const {
        return node == other.node && timestamp == other.timestamp;
    }
};

#endif // STEP_CUH
