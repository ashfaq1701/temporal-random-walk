// Streaming walk-sampling benchmark over the Alibaba microservices dataset,
// implemented entirely in C++ to avoid the Python-side cost that noises the
// tempest-benchmarks/alibaba_benchmark/test_alibaba_dataset.py runner.
//
// Expects the dataset as `data_{0..total_minutes-1}.csv` under <dataset_dir>,
// each file holding one minute of edges with columns `u,i,ts` (header row).
// Use whatever quick parquet→CSV pre-pass fits your workflow.
//
// Mirrors the Python script:
//   * Reads files in groups of `minutes_per_step` minutes, for
//     `total_minutes` minutes total.
//   * Each step: ingest edges into a TemporalRandomWalk (timed), then
//     sample walks from the unique source nodes of the most recently
//     added batch via `get_random_walks_and_times_for_last_batch` (timed).
//   * Warmup pass runs the first step untimed before the main loop to
//     absorb one-shot CUDA/CUB init costs (same pattern as
//     ablation_streaming.cpp).

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <string>
#include <tuple>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../src/common/nvtx.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../test/test_utils.h"
#include "test_utils.h"

#ifdef HAS_CUDA
constexpr bool DEFAULT_USE_GPU = true;
#else
constexpr bool DEFAULT_USE_GPU = false;
#endif

constexpr int     DEFAULT_TOTAL_MINUTES      = 60;
constexpr int     DEFAULT_MINUTES_PER_STEP   = 3;
constexpr int64_t DEFAULT_WINDOW_MS          = 1'800'000;   // 30 min
constexpr int     DEFAULT_NUM_WALKS_PER_NODE = 20;
constexpr int     DEFAULT_MAX_WALK_LEN       = 100;

RandomPickerType parse_picker(const std::string& s) {
    if (s == "uniform")            return RandomPickerType::Uniform;
    if (s == "linear")             return RandomPickerType::Linear;
    if (s == "exponential_index")  return RandomPickerType::ExponentialIndex;
    if (s == "exponential_weight") return RandomPickerType::ExponentialWeight;
    throw std::runtime_error("Invalid picker type: " + s);
}

KernelLaunchType parse_kernel_launch_type(const std::string& s) {
    if (s == "full_walk")                return KernelLaunchType::FULL_WALK;
    if (s == "node_grouped")             return KernelLaunchType::NODE_GROUPED;
    if (s == "node_grouped_global_only") return KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY;
    throw std::runtime_error(
        "Invalid kernel launch type — expected one of: "
        "full_walk, node_grouped, node_grouped_global_only");
}

struct EdgeBatch {
    std::vector<int>     src;
    std::vector<int>     dst;
    std::vector<int64_t> ts;
};

// Load data_{minute_begin..minute_begin + minutes_per_step - 1}.csv from
// the dataset directory, clipping at total_minutes. Edges are concatenated
// into a single batch in file order (file i appears before file i+1).
EdgeBatch load_step(const std::string& dataset_dir,
                    int minute_begin,
                    int minutes_per_step,
                    int total_minutes) {
    EdgeBatch batch;
    for (int j = 0; j < minutes_per_step && (minute_begin + j) < total_minutes; ++j) {
        const std::string path =
            dataset_dir + "/data_" + std::to_string(minute_begin + j) + ".csv";
        const auto edges = read_edges_from_csv(path);
        batch.src.reserve(batch.src.size() + edges.size());
        batch.dst.reserve(batch.dst.size() + edges.size());
        batch.ts.reserve(batch.ts.size() + edges.size());
        for (const auto& e : edges) {
            batch.src.push_back(std::get<0>(e));
            batch.dst.push_back(std::get<1>(e));
            batch.ts.push_back(std::get<2>(e));
        }
    }
    return batch;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0] << " <dataset_dir>"
            << " [use_gpu=1]"
            << " [picker=exponential_index]"
            << " [kernel_launch_type=full_walk|node_grouped|node_grouped_global_only]"
            << " [num_walks_per_node=" << DEFAULT_NUM_WALKS_PER_NODE << "]"
            << " [minutes_per_step=" << DEFAULT_MINUTES_PER_STEP << "]"
            << " [window_ms=" << DEFAULT_WINDOW_MS << "]"
            << " [max_walk_len=" << DEFAULT_MAX_WALK_LEN << "]"
            << " [total_minutes=" << DEFAULT_TOTAL_MINUTES << "]"
            << " [timescale_bound=-1]\n"
            << "\n"
            << "<dataset_dir> must contain data_0.csv .. data_{total-1}.csv,\n"
            << "each a header-prefixed CSV with u,i,ts columns.\n";
        return 1;
    }

    const std::string dataset_dir         = argv[1];
    const bool        use_gpu             = (argc > 2) ? std::stoi(argv[2]) != 0 : DEFAULT_USE_GPU;
    const std::string picker_str          = (argc > 3) ? argv[3] : "exponential_index";
    const std::string klt_str             = (argc > 4) ? argv[4] : "node_grouped";
    const int         num_walks_per_node  = (argc > 5) ? std::stoi(argv[5]) : DEFAULT_NUM_WALKS_PER_NODE;
    const int         minutes_per_step    = (argc > 6) ? std::stoi(argv[6]) : DEFAULT_MINUTES_PER_STEP;
    const int64_t     window_ms           = (argc > 7) ? std::stoll(argv[7]) : DEFAULT_WINDOW_MS;
    const int         max_walk_len        = (argc > 8) ? std::stoi(argv[8]) : DEFAULT_MAX_WALK_LEN;
    const int         total_minutes       = (argc > 9) ? std::stoi(argv[9]) : DEFAULT_TOTAL_MINUTES;
    const double      timescale_bound     = (argc > 10) ? std::stod(argv[10]) : -1.0;

    const RandomPickerType hop_picker = parse_picker(picker_str);
    const KernelLaunchType kernel_launch_type = parse_kernel_launch_type(klt_str);
    constexpr RandomPickerType start_picker = RandomPickerType::Uniform;

    std::cout << "=== Alibaba streaming benchmark (C++) ===\n"
              << "Dataset dir:       " << dataset_dir << "\n"
              << "Device:            " << (use_gpu ? "GPU" : "CPU") << "\n"
              << "Hop picker:        " << picker_str << "\n"
              << "Kernel type:       " << klt_str << "\n"
              << "Walks per node:    " << num_walks_per_node << "\n"
              << "Minutes per step:  " << minutes_per_step << "\n"
              << "Window (ms):       " << window_ms << "\n"
              << "Max walk len:      " << max_walk_len << "\n"
              << "Total minutes:     " << total_minutes << "\n"
              << "Timescale bound:   " << timescale_bound << "\n";

    // -----------------------------------------------------------------
    // Construct TRW (directed, windowed by window_ms).
    // -----------------------------------------------------------------
    const bool use_weight = (hop_picker == RandomPickerType::ExponentialWeight);

    TemporalRandomWalk trw(
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/window_ms,
        /*enable_weight_computation=*/use_weight,
        /*enable_temporal_node2vec=*/false,
        /*timescale_bound=*/timescale_bound);

    // -----------------------------------------------------------------
    // Warmup: load + ingest + walk-sample the first step, untimed and
    // discarded. The main loop below re-ingests the same files for
    // iteration 0 (same "repeat batch 0" pattern as ablation_streaming).
    // -----------------------------------------------------------------
    {
        NvtxRange r("warmup");
        EdgeBatch warm = load_step(dataset_dir, 0, minutes_per_step, total_minutes);
        trw.add_multiple_edges(
            warm.src.data(), warm.dst.data(), warm.ts.data(), warm.ts.size());
        auto warm_walks = trw.get_random_walks_and_times_for_last_batch(
            max_walk_len, &hop_picker, num_walks_per_node, &start_picker,
            WalkDirection::Forward_In_Time, kernel_launch_type);
#ifdef HAS_CUDA
        if (use_gpu) cudaDeviceSynchronize();
#endif
        std::cout << "\n[Warmup] Ingested " << warm.src.size()
                  << " edges, sampled " << warm_walks.walk_set.size()
                  << " walks (discarded).\n";
    }

    // -----------------------------------------------------------------
    // Streaming loop: i = 0, step, 2*step, ... < total_minutes.
    // -----------------------------------------------------------------
    std::vector<double> ingestion_times;
    std::vector<double> walk_times;
    std::vector<size_t> walks_per_step;
    size_t total_walks = 0;
    double total_walk_len_sum = 0.0;
    size_t total_edges_added  = 0;

    for (int i = 0; i < total_minutes; i += minutes_per_step) {
        EdgeBatch batch = load_step(dataset_dir, i, minutes_per_step, total_minutes);
        total_edges_added += batch.src.size();

        std::cout << "\n[Step t=" << i << "..+" << minutes_per_step
                  << "min] Edges: " << batch.src.size() << "\n";

        // Ingest (timed).
        double ingest_time = 0.0;
        {
            NvtxRange r("ingestion_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();
            trw.add_multiple_edges(
                batch.src.data(), batch.dst.data(),
                batch.ts.data(), batch.ts.size());
#ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
#endif
            const auto t1 = std::chrono::high_resolution_clock::now();
            ingest_time = std::chrono::duration<double>(t1 - t0).count();
        }
        ingestion_times.push_back(ingest_time);

        // Walk sample (timed). Walks start from the unique source nodes
        // of the edges just added.
        double walk_time = 0.0;
        size_t walks_this_step = 0;
        double avg_len_this_step = 0.0;
        {
            NvtxRange r("walk_sampling_batch");
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto walks = trw.get_random_walks_and_times_for_last_batch(
                max_walk_len, &hop_picker, num_walks_per_node, &start_picker,
                WalkDirection::Forward_In_Time, kernel_launch_type);
#ifdef HAS_CUDA
            if (use_gpu) cudaDeviceSynchronize();
#endif
            const auto t1 = std::chrono::high_resolution_clock::now();
            walk_time = std::chrono::duration<double>(t1 - t0).count();

            walks_this_step    = walks.walk_set.size();
            avg_len_this_step  = get_average_walk_length(walks.walk_set);
            total_walks        += walks_this_step;
            total_walk_len_sum += avg_len_this_step * walks_this_step;
        }
        walk_times.push_back(walk_time);
        walks_per_step.push_back(walks_this_step);

        const size_t active_edges = trw.get_edge_count();

        std::cout << "  Ingest time: " << ingest_time << " sec\n"
                  << "  Walk time:   " << walk_time << " sec\n"
                  << "  Walks:       " << walks_this_step << "\n"
                  << "  Avg length:  " << avg_len_this_step << "\n"
                  << "  Active edges in TRW: " << active_edges << "\n"
                  << "  Total edges added:   " << total_edges_added << "\n";
    }

    // -----------------------------------------------------------------
    // Summary.
    // -----------------------------------------------------------------
    const size_t num_steps = walk_times.size();
    const double total_ingestion =
        std::accumulate(ingestion_times.begin(), ingestion_times.end(), 0.0);
    const double total_walk =
        std::accumulate(walk_times.begin(), walk_times.end(), 0.0);
    const double final_avg_len =
        (total_walks > 0) ? (total_walk_len_sum / static_cast<double>(total_walks)) : 0.0;
    const double inv_steps = num_steps > 0 ? 1.0 / static_cast<double>(num_steps) : 0.0;
    const double walks_per_sec =
        (total_walk > 0.0) ? (static_cast<double>(total_walks) / total_walk) : 0.0;

    std::cout << "\n=== Summary ===\n"
              << "Steps measured:       " << num_steps << "\n"
              << "Total ingestion time: " << total_ingestion << " sec\n"
              << "Mean ingestion time:  " << (total_ingestion * inv_steps) << " sec\n"
              << "Total walk time:      " << total_walk << " sec\n"
              << "Mean walk time/step:  " << (total_walk * inv_steps) << " sec\n"
              << "Total walks:          " << total_walks << "\n"
              << "Final avg walk length:" << final_avg_len << "\n"
              << "Throughput:           " << walks_per_sec << " walks/sec\n";

    return 0;
}
