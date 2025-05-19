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

inline RandomPickerType picker_type_from_string(const std::string& picker_type_str)
{
    if (picker_type_str == "Uniform")
    {
        return RandomPickerType::Uniform;
    }
    else if (picker_type_str == "Linear")
    {
        return RandomPickerType::Linear;
    }
    else if (picker_type_str == "ExponentialIndex")
    {
        return RandomPickerType::ExponentialIndex;
    }
    else if (picker_type_str == "ExponentialWeight")
    {
        return RandomPickerType::ExponentialWeight;
    }
    else
    {
        throw std::invalid_argument("Invalid picker type: " + picker_type_str);
    }
}

inline WalkDirection walk_direction_from_string(const std::string& walk_direction_str)
{
    if (walk_direction_str == "Forward_In_Time")
    {
        return WalkDirection::Forward_In_Time;
    }
    else if (walk_direction_str == "Backward_In_Time")
    {
        return WalkDirection::Backward_In_Time;
    }
    else
    {
        throw std::invalid_argument("Invalid walk direction: " + walk_direction_str);
    }
}

#endif //TEMPORAL_RANDOM_WALK_HELPERS_CUH
