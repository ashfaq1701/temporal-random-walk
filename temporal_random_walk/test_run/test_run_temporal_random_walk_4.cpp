#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#endif

#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../test/test_utils.h"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool DEFAULT_USE_GPU = true;
#else
constexpr bool DEFAULT_USE_GPU = false;
#endif

// ------------------------------
// NVTX helper (no-op on CPU)
// ------------------------------
struct NvtxRange {
#ifdef HAS_CUDA
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~NvtxRange() {
        nvtxRangePop();
    }
#else
    explicit NvtxRange(const char*) {}
#endif
};

RandomPickerType parse_picker(const std::string &s) {
    if (s == "uniform") return RandomPickerType::Uniform;
    if (s == "linear") return RandomPickerType::Linear;
    if (s == "exponential_index") return RandomPickerType::ExponentialIndex;
    if (s == "exponential_weight") return RandomPickerType::ExponentialWeight;
    throw std::runtime_error("Invalid picker type");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <file_path>"
                  << " [use_gpu=1]"
                  << " [picker=exponential_index]"
                  << " [num_walks_per_node=1]"
                  << " [max_walk_len=80]\n";
        return 1;
    }

    const std::string file_path = argv[1];
    bool use_gpu = (argc > 2) ? std::stoi(argv[2]) : DEFAULT_USE_GPU;
    std::string picker_str = (argc > 3) ? argv[3] : "exponential_index";
    int num_walks_per_node = (argc > 4) ? std::stoi(argv[4]) : 1;
    int max_walk_len = (argc > 5) ? std::stoi(argv[5]) : 80;

    const RandomPickerType hop_picker = parse_picker(picker_str);
    constexpr RandomPickerType start_picker = RandomPickerType::Uniform;

    std::cout << "=== RNG Microbenchmark ===\n"
              << "File: " << file_path << "\n"
              << "Device: " << (use_gpu ? "GPU" : "CPU") << "\n"
              << "Hop picker: " << picker_str << "\n"
              << "Walks per node: " << num_walks_per_node << "\n"
              << "Max walk length: " << max_walk_len << "\n";

    // ------------------------------
    // Edge loading (CPU)
    // ------------------------------
    std::vector<int> sources, targets;
    std::vector<int64_t> timestamps;
    {
        NvtxRange r("edge_load");
        const auto edge_infos = read_edges_from_csv(file_path, -1);
        std::tie(sources, targets, timestamps) =
            convert_edge_tuples_to_components(edge_infos);
        std::cout << "Edges loaded: " << edge_infos.size() << "\n";
    }

    // ------------------------------
    // Construct TRW
    // ------------------------------
    const bool use_weight = hop_picker == RandomPickerType::ExponentialWeight;

    TemporalRandomWalk trw(
        false,      // undirected
        use_gpu,
        -1,         // no sliding window
        use_weight,
        /*timescale_bound=*/0.0
    );

    // ------------------------------
    // Ingestion
    // ------------------------------
    double ingestion_time = 0.0;
    {
        NvtxRange r("ingestion");
        const auto ingest_start = std::chrono::high_resolution_clock::now();
        trw.add_multiple_edges(
            sources.data(),
            targets.data(),
            timestamps.data(),
            timestamps.size());

#ifdef HAS_CUDA
        if (use_gpu) cudaDeviceSynchronize();
#endif

        const auto ingest_end = std::chrono::high_resolution_clock::now();
        ingestion_time =
            std::chrono::duration<double>(ingest_end - ingest_start).count();
    }

    std::cout << "Ingestion time: " << ingestion_time << " sec\n";

    // ------------------------------
    // Walk sampling
    // ------------------------------
    double walk_time = 0.0;
    size_t num_walks = 0;
    double avg_len = 0.0;

    {
        NvtxRange r("walk_sampling");
        const auto walk_start = std::chrono::high_resolution_clock::now();

        const auto walks = trw.get_random_walks_and_times_for_all_nodes(
            max_walk_len,
            &hop_picker,
            num_walks_per_node,
            &start_picker,
            WalkDirection::Forward_In_Time);

#ifdef HAS_CUDA
        if (use_gpu) cudaDeviceSynchronize();
#endif

        const auto walk_end = std::chrono::high_resolution_clock::now();
        walk_time =
            std::chrono::duration<double>(walk_end - walk_start).count();
        num_walks = walks.size();
        avg_len = get_average_walk_length(walks);
    }

    std::cout << "Walks generated: " << num_walks << "\n"
              << "Average walk length: " << avg_len << "\n"
              << "Walk generation time: " << walk_time << " sec\n"
              << "Throughput: " << (num_walks / walk_time) << " walks/sec\n";

    return 0;
}
