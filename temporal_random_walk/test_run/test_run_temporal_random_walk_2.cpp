#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

constexpr int NUM_WALKS_TOTAL = 1000000;


int main(int argc, char* argv[]) {
    std::string file_path = "../../data/sample_data.csv";
    char delimiter = ',';
    int num_rows = 1000000;
    bool use_gpu = USE_GPU;

    if (argc > 1) {
        file_path = argv[1];
    }

    if (argc > 2) {
        std::string gpu_arg = argv[2];
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
    std::cout << "Running on: " << (use_gpu ? "GPU" : "CPU") << std::endl;

    if (argc > 3) {
        delimiter = argv[3][0];
    }

    if (argc > 4) {
        num_rows = std::stoi(argv[4]);
    }

    const auto edge_infos = read_edges_from_csv(file_path, num_rows, delimiter);
    std::cout << "Total edges loaded: " << edge_infos.size() << std::endl;

    // Split edges into two equal parts
    size_t mid_point = edge_infos.size() / 2;
    auto first_half_begin = edge_infos.begin();
    auto first_half_end = edge_infos.begin() + static_cast<long>(mid_point);
    auto second_half_begin = edge_infos.begin() + static_cast<long>(mid_point);
    auto second_half_end = edge_infos.end();

    std::cout << "First half: " << mid_point << " edges" << std::endl;
    std::cout << "Second half: " << (edge_infos.size() - mid_point) << " edges" << std::endl;

    // Start timing for the entire process
    auto start = std::chrono::high_resolution_clock::now();
    TemporalRandomWalk temporal_random_walk(false, use_gpu, -1, true, false, 34);

    // Add first half of edges
    std::vector first_half(first_half_begin, first_half_end);
    auto [sources_first_half, targets_first_half, timestamps_first_half] =
        convert_edge_tuples_to_components(first_half);

    auto first_half_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources_first_half.data(),
        targets_first_half.data(),
        timestamps_first_half.data(),
        timestamps_first_half.size());

    auto first_half_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> first_half_duration = first_half_end_time - first_half_start;
    std::cout << "First half edge addition time: " << first_half_duration.count() << " seconds" << std::endl;
    std::cout << "Inserted edge count after first half insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    // Generate walks with first half
    auto first_half_walks_start = std::chrono::high_resolution_clock::now();

    const auto walks_backward_for_all_nodes_1 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Backward_In_Time);

    const auto walks_forward_for_all_nodes_1 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time);

    auto first_half_walks_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> first_half_walks_duration = first_half_walks_end - first_half_walks_start;
    std::cout << "First half walk generation time: " << first_half_walks_duration.count() << " seconds" << std::endl;

    std::vector second_half(second_half_begin, second_half_end);
    auto [sources_second_half, targets_second_half, timestamps_second_half] =
        convert_edge_tuples_to_components(second_half);

    // Add second half of edges to the same object
    auto second_half_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources_second_half.data(),
        targets_second_half.data(),
        timestamps_second_half.data(),
        timestamps_second_half.size());

    auto second_half_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> second_half_duration = second_half_end_time - second_half_start;
    std::cout << "Second half edge addition time: " << second_half_duration.count() << " seconds" << std::endl;
    std::cout << "Inserted edge count after second half insertion: : " << temporal_random_walk.get_edge_count() << std::endl;

    // Generate walks with all edges
    auto second_half_walks_start = std::chrono::high_resolution_clock::now();

    const auto walks_backward_for_all_nodes_2 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Backward_In_Time);

    const auto walks_forward_for_all_nodes_2 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time);

    auto second_half_walks_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> second_half_walks_duration = second_half_walks_end - second_half_walks_start;
    std::cout << "Second half walk generation time: " << second_half_walks_duration.count() << " seconds" << std::endl;

    std::cout << "\nWalks with first half edges:" << std::endl;
    std::cout << "  Forward walks: " << walks_forward_for_all_nodes_1.size() << ", average length " << get_average_walk_length(walks_forward_for_all_nodes_1) << std::endl;
    std::cout << "  Backward walks: " << walks_backward_for_all_nodes_1.size() << ", average length " << get_average_walk_length(walks_backward_for_all_nodes_1) << std::endl;

    std::cout << "\nWalks with all edges:" << std::endl;
    std::cout << "  Forward walks: " << walks_forward_for_all_nodes_2.size() << ", average length " << get_average_walk_length(walks_forward_for_all_nodes_2) << std::endl;
    std::cout << "  Backward walks: " << walks_backward_for_all_nodes_2.size() << ", average length " << get_average_walk_length(walks_backward_for_all_nodes_2) << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end - start;
    std::cout << "\nTotal execution time: " << total_duration.count() << " seconds" << std::endl;

    size_t inserted_edge_count = temporal_random_walk.get_edge_count();
    std::cout << "Inserted edge count: " << inserted_edge_count << std::endl;

    print_temporal_random_walks_with_times(walks_backward_for_all_nodes_2, 100);

    return 0;
}
