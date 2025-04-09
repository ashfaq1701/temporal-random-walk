#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

int main() {
    const std::vector<std::tuple<int, int, int64_t>> edges {
        {4, 5, 71},
        {3, 5, 82},
        {1, 3, 19},
        {4, 2, 34},
        {4, 3, 79},
        {2, 5, 19},
        {2, 3, 70},
        {5, 4, 97},
        {4, 6, 57},
        {6, 4, 27},
        {2, 6, 80},
        {6, 1, 42},
        {4, 6, 98},
        {1, 4, 17},
        {5, 4, 32}
    };

    TemporalRandomWalk temporal_random_walk(true, USE_GPU);
    temporal_random_walk.add_multiple_edges(edges);

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;

    const auto walks_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes_formatted(
        20,
        &linear_picker_type,
        10,
        &exponential_picker_type, WalkDirection::
        Forward_In_Time);

    std::cout << "Forward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_forward);

    const auto walks_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes_formatted(
        20,
        &linear_picker_type,
        10,
        &exponential_picker_type,
        WalkDirection::Backward_In_Time);

    std::cout << "Backward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_backward);

    return 0;
}
