#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <climits>

#include "test_utils.h"
#include "../src/core/helpers.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../test/test_utils.h"

#ifdef HAS_CUDA
constexpr bool DEFAULT_USE_GPU = true;
#else
constexpr bool DEFAULT_USE_GPU = false;
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
// Match the library default (DEFAULT_KERNEL_LAUNCH_TYPE in enums.cuh)
// as a string so CLI `--help` shows a realistic default.
constexpr auto DEFAULT_KERNEL_LAUNCH_TYPE_STR = "NODE_GROUPED";
// Streaming knobs. 1/1 keeps the historical bulk path (single batch, no
// eviction). >1 on either dimension switches to a per-batch loop.
constexpr int DEFAULT_BATCH_DIVIDER  = 1;
constexpr int DEFAULT_WINDOW_DIVIDER = 1;

// ============================
// Performance Stats
// ============================

void print_walk_performance_stats(
    const size_t num_walks,
    const size_t total_steps,
    const double duration_seconds)
{
    if (duration_seconds <= 0.0) {
        std::cout << "Duration too small to compute throughput.\n";
        return;
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
                  << " [use_gpu (0|1, default=" << (DEFAULT_USE_GPU ? 1 : 0) << ")]"
                  << " [is_directed (0|1, default=0)]"
                  << " [num_total_walks]"
                  << " [num_walks_per_node]"
                  << " [max_walk_length]"
                  << " [edge_picker]"
                  << " [start_picker]"
                  << " [kernel_launch_type=FULL_WALK|NODE_GROUPED|NODE_GROUPED_GLOBAL_ONLY]"
                  << " [batch_divider=" << DEFAULT_BATCH_DIVIDER << "]"
                  << " [window_divider=" << DEFAULT_WINDOW_DIVIDER << "]"
                  << " [walk_dump_file]"
                  << "\n\n"
                  << "Bulk-ingest by default. If batch_divider > 1 or\n"
                  << "window_divider > 1, switches to a streaming loop:\n"
                  << "edges are sorted by timestamp and split into\n"
                  << "batch_divider equal-time batches; max_time_capacity is\n"
                  << "set to (ts_max - ts_min) / window_divider so older edges\n"
                  << "are evicted as new ones arrive. Per-batch ingest and\n"
                  << "walk-sampling times are summed and reported on the same\n"
                  << "\"Ingest time:\" / \"Walk time:\" lines as bulk mode, so\n"
                  << "the same Python parser handles both regimes.\n";
        return 1;
    }

    const std::string file_path = argv[1];

    const bool use_gpu =
        (argc > 2)
            ? (std::stoi(argv[2]) != 0)
            : DEFAULT_USE_GPU;

    const bool is_directed =
        (argc > 3)
            ? (std::stoi(argv[3]) != 0)
            : DEFAULT_IS_DIRECTED;

    const int num_total_walks =
        (argc > 4) ? std::stoi(argv[4]) : DEFAULT_NUM_TOTAL_WALKS;

    const int num_walks_per_node =
        (argc > 5) ? std::stoi(argv[5]) : DEFAULT_NUM_WALKS_PER_NODE;

    const int max_walk_length =
        (argc > 6) ? std::stoi(argv[6]) : DEFAULT_MAX_WALK_LENGTH;

    const std::string edge_picker =
        (argc > 7) ? argv[7] : DEFAULT_EDGE_PICKER;

    const std::string start_picker =
        (argc > 8) ? argv[8] : DEFAULT_START_PICKER;

    const std::string kernel_launch_type_str =
        (argc > 9) ? argv[9] : DEFAULT_KERNEL_LAUNCH_TYPE_STR;

    const int batch_divider =
        (argc > 10) ? std::stoi(argv[10]) : DEFAULT_BATCH_DIVIDER;

    const int window_divider =
        (argc > 11) ? std::stoi(argv[11]) : DEFAULT_WINDOW_DIVIDER;

    const std::string walk_dump_file =
        (argc > 12) ? argv[12] : "";

    if (batch_divider < 1 || window_divider < 1) {
        std::cerr << "batch_divider and window_divider must both be >= 1.\n";
        return 1;
    }

    const bool streaming = (batch_divider > 1) || (window_divider > 1);

    std::cout << "Running on: " << (use_gpu ? "GPU" : "CPU") << "\n";
    std::cout << "Graph type: "
              << (is_directed ? "Directed" : "Undirected")
              << "\n";
    std::cout << "Kernel launch type: " << kernel_launch_type_str << "\n";
    std::cout << "Edge picker: " << edge_picker
              << "  Start picker: " << start_picker << "\n";
    std::cout << "Mode: " << (streaming ? "STREAMING" : "BULK")
              << "  (batch_divider=" << batch_divider
              << ", window_divider=" << window_divider << ")\n";

    // ============================
    // Load Edges
    // ============================

    auto edge_infos = read_edges_from_csv(file_path);

    // Streaming mode needs strictly time-ordered ingestion, so sort here.
    // Stable to preserve original tie-break order on equal timestamps.
    if (streaming) {
        std::stable_sort(
            edge_infos.begin(), edge_infos.end(),
            [](const auto& a, const auto& b) {
                return std::get<2>(a) < std::get<2>(b);
            });
    }

    auto [sources, targets, timestamps] =
        convert_edge_tuples_to_components(edge_infos);

    std::cout << "Edges loaded: "
              << edge_infos.size() << "\n";

    if (timestamps.empty()) {
        std::cerr << "Empty edge file; nothing to do.\n";
        return 1;
    }

    // ============================
    // Resolve Pickers
    // ============================

    const RandomPickerType edge_picker_enum =
        picker_type_from_string(edge_picker);

    const RandomPickerType start_picker_enum =
        picker_type_from_string(start_picker);

    const KernelLaunchType kernel_launch_type =
        kernel_launch_type_from_string(kernel_launch_type_str);

    // ============================
    // Construct Walker
    // ============================

    const bool enable_temporal_node2vec =
        edge_picker_enum == RandomPickerType::TemporalNode2Vec;
    const bool enable_weight_computation =
        edge_picker_enum  == RandomPickerType::ExponentialWeight
        || start_picker_enum == RandomPickerType::ExponentialWeight;

    // Streaming window cutoff (max_time_capacity). -1 disables eviction.
    int64_t total_span = 0;
    int64_t t_min = 0;
    int64_t t_max = 0;
    if (streaming) {
        t_min = timestamps.front();
        t_max = timestamps.back();
        total_span = t_max - t_min;
    }
    const int64_t max_time_capacity =
        (window_divider > 1)
            ? std::max<int64_t>(1, total_span / window_divider)
            : -1;

    TemporalRandomWalk walker(
        is_directed,
        use_gpu,
        max_time_capacity,
        enable_weight_computation,
        enable_temporal_node2vec,
        34                           // timescale_bound
    );

    auto run_walks = [&]() {
        return (num_walks_per_node == -1)
            ? walker.get_random_walks_and_times(
                max_walk_length,
                &edge_picker_enum,
                num_total_walks,
                &start_picker_enum,
                WalkDirection::Forward_In_Time,
                kernel_launch_type)
            : walker.get_random_walks_and_times_for_all_nodes(
                max_walk_length,
                &edge_picker_enum,
                num_walks_per_node,
                &start_picker_enum,
                WalkDirection::Forward_In_Time,
                kernel_launch_type);
    };

    auto sum_walk_lens = [](const auto& walk_set) -> size_t {
        const size_t* walk_lens = walk_set.walk_lens_ptr();
        size_t total = 0;
        for (size_t i = 0; i < walk_set.num_walks(); ++i) {
            total += walk_lens[i];
        }
        return total;
    };

    double ingest_total_s = 0.0;
    double walk_total_s   = 0.0;
    size_t total_walks    = 0;
    size_t total_steps    = 0;
    // Captured for the optional walk dump (last batch in streaming, full
    // run in bulk). Kept alive past the loop scope for the dump call.
    decltype(run_walks()) last_walks_with_edge_features;

    if (!streaming) {
        std::cout << "\nIngesting edges in bulk...\n";

        const auto t1 = std::chrono::high_resolution_clock::now();
        walker.add_multiple_edges(
            sources.data(), targets.data(), timestamps.data(),
            timestamps.size());
        const auto t2 = std::chrono::high_resolution_clock::now();
        ingest_total_s = std::chrono::duration<double>(t2 - t1).count();

        std::cout << std::fixed << std::setprecision(6)
                  << "Ingest time: " << ingest_total_s << " seconds\n";
        std::cout << "Graph constructed. Nodes: "
                  << walker.get_node_count()
                  << " Edges: "
                  << walker.get_edge_count() << "\n";

        std::cout << "\nGenerating walks...\n";
        const auto t3 = std::chrono::high_resolution_clock::now();
        last_walks_with_edge_features = run_walks();
        const auto t4 = std::chrono::high_resolution_clock::now();
        walk_total_s = std::chrono::duration<double>(t4 - t3).count();

        const auto& walk_set = last_walks_with_edge_features.walk_set;
        total_walks = walk_set.num_walks();
        total_steps = sum_walk_lens(walk_set);
    } else {
        std::cout << "\nStreaming over " << batch_divider << " batches"
                  << " (window=" << max_time_capacity << " ts-units)...\n";

        // Equal-time batches over [t_min, t_max]. Last batch absorbs any
        // integer-division remainder by including everything up to t_max.
        const int64_t batch_step =
            std::max<int64_t>(1, total_span / batch_divider);

        for (int b = 0; b < batch_divider; ++b) {
            const int64_t ts_lo =
                t_min + static_cast<int64_t>(b) * batch_step;
            const int64_t ts_hi =
                (b + 1 == batch_divider)
                    ? (t_max + 1)
                    : (t_min + static_cast<int64_t>(b + 1) * batch_step);

            const auto it_lo =
                std::lower_bound(timestamps.begin(), timestamps.end(), ts_lo);
            const auto it_hi =
                std::lower_bound(timestamps.begin(), timestamps.end(), ts_hi);
            const size_t batch_start =
                static_cast<size_t>(std::distance(timestamps.begin(), it_lo));
            const size_t batch_count =
                static_cast<size_t>(std::distance(it_lo, it_hi));

            if (batch_count == 0) {
                std::cout << "  batch " << (b + 1) << "/" << batch_divider
                          << ": empty, skipping\n";
                continue;
            }

            const auto t1 = std::chrono::high_resolution_clock::now();
            walker.add_multiple_edges(
                sources.data()    + batch_start,
                targets.data()    + batch_start,
                timestamps.data() + batch_start,
                batch_count);
            const auto t2 = std::chrono::high_resolution_clock::now();
            const double ingest_b =
                std::chrono::duration<double>(t2 - t1).count();
            ingest_total_s += ingest_b;

            const auto t3 = std::chrono::high_resolution_clock::now();
            last_walks_with_edge_features = run_walks();
            const auto t4 = std::chrono::high_resolution_clock::now();
            const double walk_b =
                std::chrono::duration<double>(t4 - t3).count();
            walk_total_s += walk_b;

            const auto& walk_set = last_walks_with_edge_features.walk_set;
            const size_t batch_walks = walk_set.num_walks();
            const size_t batch_steps = sum_walk_lens(walk_set);
            total_walks += batch_walks;
            total_steps += batch_steps;

            std::cout << std::fixed << std::setprecision(4)
                      << "  batch " << (b + 1) << "/" << batch_divider
                      << ": edges=" << batch_count
                      << "  ingest=" << ingest_b << "s"
                      << "  walk="   << walk_b   << "s"
                      << "  walks="  << batch_walks
                      << "  steps="  << batch_steps
                      << "\n";
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "Ingest time: " << ingest_total_s << " seconds\n";
        std::cout << "Graph at end. Nodes: "
                  << walker.get_node_count()
                  << " Edges: "
                  << walker.get_edge_count() << "\n";
    }

    std::cout << std::fixed << std::setprecision(6)
              << "Walk time: " << walk_total_s << " seconds\n";

    print_walk_performance_stats(total_walks, total_steps, walk_total_s);

    // ============================
    // Optional Dump (last batch in streaming mode)
    // ============================

    if (!walk_dump_file.empty()) {
        dump_walks_to_file(
            last_walks_with_edge_features.walk_set,
            max_walk_length,
            walk_dump_file);
    }

    const size_t memory_footprint_bytes =
        walker.get_memory_used()
        + last_walks_with_edge_features.walk_set.get_memory_used();
    const double memory_gb =
        static_cast<double>(memory_footprint_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Memory used (GB): " << memory_gb << std::endl;

    return 0;
}
