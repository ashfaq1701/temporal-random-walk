#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>

#include "test_utils.h"
#include "../src/core/helpers.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

// ============================
// Defaults
// ============================

constexpr bool DEFAULT_IS_DIRECTED = false;
constexpr int DEFAULT_MAX_WALK_LENGTH = 80;
constexpr int DEFAULT_NUM_TOTAL_WALKS = 10'000'000;
constexpr int DEFAULT_NUM_WALKS_PER_NODE = -1;
constexpr auto DEFAULT_EDGE_PICKER = "ExponentialIndex";
constexpr auto DEFAULT_START_PICKER = "Uniform";

// ============================
// Performance Stats
// ============================

void print_walk_performance_stats(
    const size_t num_walks,
    const size_t* walk_lens,
    const double duration_seconds)
{
    if (duration_seconds <= 0.0) {
        std::cout << "Duration too small to compute throughput.\n";
        return;
    }

    size_t total_steps = 0;
    for (size_t i = 0; i < num_walks; ++i) {
        total_steps += walk_lens[i];
    }

    const double walks_per_sec =
        static_cast<double>(num_walks) / duration_seconds;

    const double steps_per_sec =
        static_cast<double>(total_steps) / duration_seconds;

    const double avg_walk_length =
        num_walks > 0
            ? static_cast<double>(total_steps) / static_cast<double>(num_walks)
            : 0.0;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Walk generation completed.\n"
              << "Total walks: " << num_walks << "\n"
              << "Total steps: " << total_steps << "\n"
              << "Average walk length: " << avg_walk_length << "\n"
              << "Time taken: " << duration_seconds << " seconds\n"
              << "Throughput:\n"
              << "  Walks/sec: " << walks_per_sec << "\n"
              << "  Steps/sec: " << steps_per_sec << "\n";
}

// ============================
// Main
// ============================

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << argv[0]
                  << " <edge_file_path>"
                  << " [is_directed (0|1, default=0)]"
                  << " [num_total_walks]"
                  << " [num_walks_per_node]"
                  << " [max_walk_length]"
                  << " [edge_picker]"
                  << " [start_picker]"
                  << " [walk_dump_file]\n";
        return 1;
    }

    const std::string file_path = argv[1];

    // Optional is_directed (default = false)
    const bool is_directed =
        (argc > 2)
            ? (std::stoi(argv[2]) != 0)
            : DEFAULT_IS_DIRECTED;

    const int num_total_walks =
        (argc > 3) ? std::stoi(argv[3]) : DEFAULT_NUM_TOTAL_WALKS;

    const int num_walks_per_node =
        (argc > 4) ? std::stoi(argv[4]) : DEFAULT_NUM_WALKS_PER_NODE;

    const int max_walk_length =
        (argc > 5) ? std::stoi(argv[5]) : DEFAULT_MAX_WALK_LENGTH;

    const std::string edge_picker =
        (argc > 6) ? argv[6] : DEFAULT_EDGE_PICKER;

    const std::string start_picker =
        (argc > 7) ? argv[7] : DEFAULT_START_PICKER;

    const std::string walk_dump_file =
        (argc > 8) ? argv[8] : "";

    std::cout << "Running on: " << (USE_GPU ? "GPU" : "CPU") << "\n";
    std::cout << "Graph type: "
              << (is_directed ? "Directed" : "Undirected")
              << "\n";

    // ============================
    // Load Edges
    // ============================

    const auto edge_infos = read_edges_from_csv(file_path);

    auto [sources, targets, timestamps] =
        convert_edge_tuples_to_components(edge_infos);

    std::cout << "Edges loaded: "
              << edge_infos.size() << "\n";

    // ============================
    // Construct Walker
    // ============================

    TemporalRandomWalk walker(
        is_directed,
        USE_GPU,
        -1,     // max_time_capacity
        true,   // enable_weight_computation
        true,   // enable_temporal_node2vec
        34      // timescale_bound
    );

    walker.add_multiple_edges(
        sources.data(),
        targets.data(),
        timestamps.data(),
        timestamps.size()
    );

    std::cout << "Graph constructed. Nodes: "
              << walker.get_node_count()
              << " Edges: "
              << walker.get_edge_count()
              << "\n";

    // ============================
    // Resolve Pickers
    // ============================

    const RandomPickerType edge_picker_enum =
        picker_type_from_string(edge_picker);

    const RandomPickerType start_picker_enum =
        picker_type_from_string(start_picker);

    // ============================
    // Generate Walks (Timed)
    // ============================

    std::cout << "\nGenerating walks...\n";

    const auto start_time =
        std::chrono::high_resolution_clock::now();

    WalkSet walk_set;

    if (num_walks_per_node == -1) {
        walk_set = walker.get_random_walks_and_times(
            max_walk_length,
            &edge_picker_enum,
            num_total_walks,
            &start_picker_enum,
            WalkDirection::Forward_In_Time
        );
    } else {
        walk_set = walker.get_random_walks_and_times_for_all_nodes(
            max_walk_length,
            &edge_picker_enum,
            num_walks_per_node,
            &start_picker_enum,
            WalkDirection::Forward_In_Time
        );
    }

    const auto end_time =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> duration =
        end_time - start_time;

    print_walk_performance_stats(
        walk_set.num_walks,
        walk_set.walk_lens,
        duration.count()
    );

    // ============================
    // Optional Dump
    // ============================

    if (!walk_dump_file.empty()) {
        dump_walks_to_file(
            walk_set,
            max_walk_length,
            walk_dump_file
        );
    }

    return 0;
}
