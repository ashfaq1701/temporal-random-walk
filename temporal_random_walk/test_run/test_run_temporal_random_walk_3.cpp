#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

constexpr int MAX_WALK_LEN = 80;

void benchmark_exponential_index_forward(const std::vector<int> &sources,
                                         const std::vector<int> &targets,
                                         const std::vector<int64_t> &timestamps,
                                         const int num_walks_per_node,
                                         const double timescale_bound) {
    std::cout << "\n=== ExponentialIndex Forward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, false, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    const auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType exponential_index_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_index_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Forward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

void benchmark_exponential_index_backward(const std::vector<int> &sources,
                                          const std::vector<int> &targets,
                                          const std::vector<int64_t> &timestamps,
                                          const int num_walks_per_node,
                                          const double timescale_bound) {
    std::cout << "\n=== ExponentialIndex Backward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, false, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    const auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType exponential_index_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_index_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Backward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

void benchmark_exponential_weight_forward(const std::vector<int> &sources,
                                          const std::vector<int> &targets,
                                          const std::vector<int64_t> &timestamps,
                                          const int num_walks_per_node,
                                          const double timescale_bound) {
    std::cout << "\n=== ExponentialWeight Forward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, true, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType exponential_weight_picker_type = RandomPickerType::ExponentialWeight;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Forward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

void benchmark_exponential_weight_backward(const std::vector<int> &sources,
                                           const std::vector<int> &targets,
                                           const std::vector<int64_t> &timestamps,
                                           const int num_walks_per_node,
                                           const double timescale_bound) {
    std::cout << "\n=== ExponentialWeight Backward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, true, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    const auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType exponential_weight_picker_type = RandomPickerType::ExponentialWeight;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Backward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

void benchmark_linear_forward(const std::vector<int> &sources,
                              const std::vector<int> &targets,
                              const std::vector<int64_t> &timestamps,
                              const int num_walks_per_node,
                              const double timescale_bound) {
    std::cout << "\n=== Linear Forward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, false, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    const auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Forward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

void benchmark_linear_backward(const std::vector<int> &sources,
                               const std::vector<int> &targets,
                               const std::vector<int64_t> &timestamps,
                               const int num_walks_per_node,
                               const double timescale_bound) {
    std::cout << "\n=== Linear Backward Walks ===" << std::endl;

    const TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, false, timescale_bound);

    const auto edge_addition_start = std::chrono::high_resolution_clock::now();
    temporal_random_walk.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size());
    const auto edge_addition_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> edge_addition_duration = edge_addition_end - edge_addition_start;

    std::cout << "Edge addition time: " << edge_addition_duration.count() << " seconds" << std::endl;
    std::cout << "Edge count after insertion: " << temporal_random_walk.get_edge_count() << std::endl;

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walk_start = std::chrono::high_resolution_clock::now();
    const auto walks = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, num_walks_per_node, &uniform_picker_type,
        WalkDirection::Backward_In_Time);
    const auto walk_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> walk_duration = walk_end - walk_start;

    std::cout << "Walks: " << walks.size() << ", average length " << get_average_walk_length(walks)
            << ", time: " << walk_duration.count() << " seconds" << std::endl;
}

int main(const int argc, char *argv[]) {
    char delimiter = ',';
    double timescale_bound = 34;
    int num_walks_per_node = 1;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [timescale_bound] [delimiter] [num_walks_per_node]\n";
        return 1;
    }

    const std::string file_path = argv[1];

    if (argc > 2) {
        timescale_bound = std::stod(argv[2]);
    }

    if (argc > 3) {
        delimiter = argv[3][0];
    }

    if (argc > 4) {
        num_walks_per_node = std::stoi(argv[4]);
    }

    const auto edge_infos = read_edges_from_csv(file_path, -1, delimiter);
    auto [sources, targets, timestamps] = convert_edge_tuples_to_components(edge_infos);
    std::cout << "Total edges loaded: " << edge_infos.size() << std::endl;

    const auto start = std::chrono::high_resolution_clock::now();

    // Run each benchmark separately - TRW instance is created and destroyed in each function
    benchmark_exponential_index_forward(sources, targets, timestamps, num_walks_per_node, timescale_bound);
    benchmark_exponential_index_backward(sources, targets, timestamps, num_walks_per_node, timescale_bound);
    benchmark_exponential_weight_forward(sources, targets, timestamps, num_walks_per_node, timescale_bound);
    benchmark_exponential_weight_backward(sources, targets, timestamps, num_walks_per_node, timescale_bound);
    benchmark_linear_forward(sources, targets, timestamps, num_walks_per_node, timescale_bound);
    benchmark_linear_backward(sources, targets, timestamps, num_walks_per_node, timescale_bound);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> total_duration = end - start;

    std::cout << "\n=== Overall Summary ===" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " seconds" << std::endl;

    return 0;
}
