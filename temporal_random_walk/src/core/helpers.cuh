#ifndef TEMPORAL_RANDOM_WALK_HELPERS_CUH
#define TEMPORAL_RANDOM_WALK_HELPERS_CUH

#include "../common/macros.cuh"
#include "../data/enums.cuh"

namespace temporal_random_walk {
    HOST DEVICE inline bool get_should_walk_forward(const WalkDirection walk_direction) {
        switch (walk_direction) {
            case WalkDirection::Forward_In_Time:
                return true;
            case WalkDirection::Backward_In_Time:
                return false;
            default:
                return true;
        }
    }
}

#endif //TEMPORAL_RANDOM_WALK_HELPERS_CUH
