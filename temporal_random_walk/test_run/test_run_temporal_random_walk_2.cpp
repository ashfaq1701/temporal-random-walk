#include <vector>

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

constexpr int NUM_WALKS_TOTAL = 100000;
constexpr int NODE_COUNT_MAX_BOUND = 1000000;


int main(int argc, char* argv[]) {
    std::string file_path = "../../data/sample_data.csv";
    char delimiter = ',';
    int num_rows = 1000000;

    if (argc > 1) {
        file_path = argv[1];
    }

    if (argc > 2) {
        delimiter = argv[2][0];
    }

    if (argc > 3) {
        num_rows = std::stoi(argv[3]);
    }

    const auto edge_infos = read_edges_from_csv(file_path, num_rows, delimiter);
    std::cout << edge_infos.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    TemporalRandomWalk temporal_random_walk(false, USE_GPU, -1, true, 34, NODE_COUNT_MAX_BOUND);
    temporal_random_walk.add_multiple_edges(edge_infos);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Edge addition time: " << duration.count() << " seconds" << std::endl;

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    start = std::chrono::high_resolution_clock::now();

    const auto walks_backward_for_all_nodes_1 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Backward_In_Time);

    const auto walks_backward_for_all_nodes_2 = temporal_random_walk.get_random_walks_and_times(
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

    const auto walks_forward_for_all_nodes_2 = temporal_random_walk.get_random_walks_and_times(
        80,
        &exponential_picker_type,
        NUM_WALKS_TOTAL,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time);

    std::cout << "Walks forward: " << walks_forward_for_all_nodes_2.size() << ", average length " << get_average_walk_length(walks_forward_for_all_nodes_2) << std::endl;

    std::cout << "Walks backward: " << walks_backward_for_all_nodes_2.size() << ", average length " << get_average_walk_length(walks_backward_for_all_nodes_2) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Walk generation time: " << duration.count() << " seconds" << std::endl;

    std::vector<std::vector<NodeWithTime>> first_100_walks_forward;
    first_100_walks_forward.assign(walks_forward_for_all_nodes_2.begin(), walks_forward_for_all_nodes_2.begin() + min(NUM_WALKS_TOTAL, 100));

    print_temporal_random_walks_with_times(first_100_walks_forward);

    return 0;
}
