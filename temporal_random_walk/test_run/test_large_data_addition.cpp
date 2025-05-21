#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

const auto EDGE_PICKER_TYPE = "ExponentialIndex";
const auto START_PICKER_TYPE = "Uniform";
constexpr auto WALK_LEN = 100;
constexpr int NUM_WALKS_TOTAL = 1000000;
constexpr int SLIDING_WINDOW_DURATION = 540000;  // 9 minutes data

int main(const int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [base_dir] [use_gpu] [SLIDING_WINDOW_DURATION_MS] [edge_picker_type] [start_picker_type] [walk_len] [num_walks_total]" << std::endl;
        return -1;
    }

    const std::string file_base_path = argv[1];

    bool use_gpu = USE_GPU;  // Default value from a defined constant
    if (argc >= 3) {
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

    auto sliding_window_duration = SLIDING_WINDOW_DURATION;
    if (argc >= 4) {
        sliding_window_duration = std::stoi(argv[3]);
    }

    auto edge_picker = picker_type_from_string(EDGE_PICKER_TYPE);
    if (argc >= 5) {
        edge_picker = picker_type_from_string(argv[4]);
    }

    auto start_picker = picker_type_from_string(START_PICKER_TYPE);
    if (argc >= 6) {
        start_picker = picker_type_from_string(argv[5]);
    }

    auto walk_len = WALK_LEN;
    if (argc >= 7) {
        walk_len = std::stoi(argv[6]);
    }

    auto num_walks_total = NUM_WALKS_TOTAL;
    if (argc >= 8) {
        num_walks_total = std::stoi(argv[7]);
    }

    const auto file_paths = get_sorted_data_files(file_base_path);

    const TemporalRandomWalk temporal_random_walk(true, use_gpu, sliding_window_duration, true);

    std::vector<double> edge_addition_elapsed_times;
    double edge_addition_total_time = 0.0;

    std::vector<double> walk_sampling_elapsed_times;
    double walk_sampling_total_time = 0.0;

    for (size_t i = 0; i < file_paths.size(); ++i) {
        const auto& file_path = file_paths[i];
        std::cout << "\nProcessing file " << (i+1) << "/" << file_paths.size()
                  << ": " << file_path << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;

        const auto edge_infos = read_edges_from_csv(file_path);

        auto [current_sources, current_targets, current_timestamps] =
        convert_edge_tuples_to_components(edge_infos);

        std::cout << "Adding " << current_sources.size() << " edges to the temporal random walk..." << std::endl;

        auto edge_addition_start_time = std::chrono::high_resolution_clock::now();
        temporal_random_walk.add_multiple_edges(
            current_sources.data(),
            current_targets.data(),  // Fixed: was using current_sources.data() twice
            current_timestamps.data(),
            current_sources.size());
        auto edge_addition_end_time = std::chrono::high_resolution_clock::now();

        const auto active_edge_count = temporal_random_walk.get_edge_count();

        // Fixed: end_time - start_time (was reversed)
        std::chrono::duration<double> edge_addition_elapsed = edge_addition_end_time - edge_addition_start_time;
        double edge_addition_elapsed_sec = edge_addition_elapsed.count();
        edge_addition_elapsed_times.push_back(edge_addition_elapsed_sec);
        edge_addition_total_time += edge_addition_elapsed_sec;

        std::cout << "add_multiple_edges call #" << (i+1) << " completed in "
                  << std::fixed << std::setprecision(2) << edge_addition_elapsed_sec << " seconds. Active edge count " <<  active_edge_count << std::endl;

        std::cout << "Sampling " << num_walks_total << " walks with length " << walk_len << "..." << std::endl;

        auto walk_sampling_start_time = std::chrono::high_resolution_clock::now();
        auto walks = temporal_random_walk.get_random_walks_and_times(
            walk_len,  // Fixed: was using hardcoded 100 instead of walk_len
            &edge_picker,
            num_walks_total,  // Fixed: was using NUM_WALKS_TOTAL constant instead of variable
            &start_picker,
            WalkDirection::Forward_In_Time);
        auto walk_sampling_end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> walk_sampling_elapsed = walk_sampling_end_time - walk_sampling_start_time;
        double walk_sampling_elapsed_sec = walk_sampling_elapsed.count();
        walk_sampling_elapsed_times.push_back(walk_sampling_elapsed_sec);
        walk_sampling_total_time += walk_sampling_elapsed_sec;

        std::cout << "get_random_walks_and_times call #" << (i+1) << " completed in "
                  << std::fixed << std::setprecision(2) << walk_sampling_elapsed_sec << " seconds" << std::endl;
    }

    // Calculate averages
    const double edge_addition_average_time = edge_addition_total_time / static_cast<double>(edge_addition_elapsed_times.size());
    const double walk_sampling_average_time = walk_sampling_total_time / static_cast<double>(walk_sampling_elapsed_times.size());

    // Print summary for edge addition
    std::cout << "\n==== Edge Addition Summary ====" << std::endl;
    std::cout << "Total time spent in add_multiple_edges: " << std::fixed << std::setprecision(2)
              << edge_addition_total_time << " seconds" << std::endl;
    std::cout << "Average time per add_multiple_edges call: " << std::fixed << std::setprecision(2)
              << edge_addition_average_time << " seconds" << std::endl;

    // Print summary for walk sampling
    std::cout << "\n==== Walk Sampling Summary ====" << std::endl;
    std::cout << "Total time spent in get_random_walks_and_times: " << std::fixed << std::setprecision(2)
              << walk_sampling_total_time << " seconds" << std::endl;
    std::cout << "Average time per get_random_walks_and_times call: " << std::fixed << std::setprecision(2)
              << walk_sampling_average_time << " seconds" << std::endl;

    return 0;
}
