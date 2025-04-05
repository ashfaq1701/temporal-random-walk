#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = false;
#else
constexpr bool USE_GPU = false;
#endif

int main() {
    const std::vector<std::tuple<int, int, int64_t>> edges {
        {10, 25, 100},
        {10, 30, 100},
        {30, 45, 100},
        {25, 30, 200},
        {45, 60, 200},
        {25, 60, 200},
        {10, 50, 200},
        {30, 10, 300},
        {60, 25, 300},
        {10, 25, 300},
        {50, 45, 400},
        {25, 10, 400},
        {60, 10, 400},
        {60, 50, 400}
    };

    TemporalRandomWalk temporal_random_walk(true, USE_GPU, -1, false, DEFAULT_TIMESCALE_BOUND, 6);
    temporal_random_walk.add_multiple_edges(edges);

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;

    const auto walks_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(20, &linear_picker_type, 10, &exponential_picker_type, WalkDirection::Forward_In_Time);
    std::cout << "Forward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_forward);

    const auto walks_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(20, &linear_picker_type, 10, &exponential_picker_type, WalkDirection::Backward_In_Time);
    std::cout << "Backward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_backward);

    return 0;
}
