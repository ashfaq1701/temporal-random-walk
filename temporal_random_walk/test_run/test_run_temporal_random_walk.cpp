#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

int main(const int argc, char* argv[]) {
    bool use_gpu = USE_GPU;

    if (argc > 1) {
        std::string gpu_arg = argv[1];
        // Convert to lowercase for case-insensitive comparison
        std::transform(gpu_arg.begin(), gpu_arg.end(), gpu_arg.begin(),
                       [](const unsigned char c){ return std::tolower(c); });

        // Accept various forms of true/false input
        if (gpu_arg == "1" || gpu_arg == "true" || gpu_arg == "yes" || gpu_arg == "y") {
            use_gpu = true;
        } else if (gpu_arg == "0" || gpu_arg == "false" || gpu_arg == "no" || gpu_arg == "n") {
            use_gpu = false;
        } else {
            std::cerr << "Error: Invalid value '" << gpu_arg << "' for use_gpu parameter. Expected 1/0, true/false, yes/no, or y/n." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    const std::vector<std::tuple<int, int, int64_t>> edges {
        {0, 1, 100}, {1, 2, 100}, {3, 0, 100},
        {0, 2, 101}, {1, 3, 101}, {2, 4, 101},
        {4, 0, 101}, {1, 4, 102}, {2, 3, 102},
        {3, 1, 102}, {4, 2, 102}, {0, 3, 103},
        {2, 0, 103}, {3, 4, 104}, {4, 1, 104},
        {0, 4, 104}, {1, 0, 105}, {2, 1, 105},
        {3, 2, 106}, {4, 3, 107}
    };

    auto [sources, targets, timestamps] = convert_edge_tuples_to_components(edges);

    const TemporalRandomWalk temporal_random_walk(true, use_gpu);
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;

    const auto walks_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        20,
        &linear_picker_type,
        10,
        &exponential_picker_type, WalkDirection::
        Forward_In_Time);

    std::cout << "Forward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_forward);

    const auto walks_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        20,
        &linear_picker_type,
        10,
        &exponential_picker_type,
        WalkDirection::Backward_In_Time);

    std::cout << "Backward walks:" << std::endl;
    print_temporal_random_walks_with_times(walks_backward);

    return 0;
}
