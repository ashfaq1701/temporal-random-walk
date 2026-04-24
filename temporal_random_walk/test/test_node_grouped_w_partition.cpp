// GPU-only tests for the NODE_GROUPED scheduler's W-partition stage.
//
// The W-partition (task 5) classifies each unique node at a step into one
// of three tiers based on its walk count W:
//   W <= W_THRESHOLD_WARP   (==1)   -> solo_walks
//   W_THRESHOLD_WARP <  W <= BLOCK_DIM         (<=255) -> warp_nodes + walk_starts/counts
//   W  >  BLOCK_DIM                  (>=256) -> block_nodes + walk_starts/counts
//
// Scope covered here:
//   - disjoint coverage: the three tier lists partition exactly the active
//     walks, no duplicates, no terminated leaks,
//   - count identity: num_solo + sum(warp_counts) + sum(block_counts)
//     == num_active,
//   - tier boundaries: W={1,2} (solo/warp) and W={BLOCK_DIM, BLOCK_DIM+1}
//     (warp/block),
//   - termination filtering: walks with walk_padding_value at the current
//     step never reach the partition,
//   - walk-idx preservation: solo_walks carries original sparse indices.
//
// Scope deferred to sibling files:
//   - G-partition of warp/block into (smem, global) variants by per-node
//     timestamp-group count ->  test_node_grouped_g_partition.cpp
//   - Block-task expansion of W>W_THRESHOLD_MULTI_BLOCK nodes into multiple tasks
//     ->                         test_node_grouped_block_task_expansion.cpp
//
// This file is guarded by HAS_CUDA. It is NOT paired with a CPU variant
// because the scheduler only exists on GPU.

#ifdef HAS_CUDA

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "../src/common/const.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/core/node_grouped/scheduler.cuh"
#include "../src/data/walk_set/walk_set_view.cuh"

namespace {

using temporal_random_walk::NodeGroupedScheduler;

// --------------------------------------------------------------------------
// Controllable WalkSetView builder.
//
// Allocates the four device buffers the view points at, fills `nodes` with
// walk_padding_value everywhere except the caller-specified step_number
// slot (which receives last_nodes_at_step[walk_idx]). The scheduler only
// reads walk_set.nodes at that offset for its filter/gather stages, so the
// other buffers can stay uninitialized.
// --------------------------------------------------------------------------
struct DeviceWalkSet {
    int*     nodes              = nullptr;
    int64_t* timestamps         = nullptr;
    std::size_t* walk_lens      = nullptr;
    int64_t* edge_ids           = nullptr;
    std::size_t num_walks       = 0;
    std::size_t max_len         = 0;
    int walk_padding_value      = EMPTY_NODE_VALUE;

    ~DeviceWalkSet() {
        if (nodes)      cudaFree(nodes);
        if (timestamps) cudaFree(timestamps);
        if (walk_lens)  cudaFree(walk_lens);
        if (edge_ids)   cudaFree(edge_ids);
    }

    DeviceWalkSet() = default;
    DeviceWalkSet(const DeviceWalkSet&)            = delete;
    DeviceWalkSet& operator=(const DeviceWalkSet&) = delete;
};

std::unique_ptr<DeviceWalkSet> make_walk_set(
    const std::vector<int>& last_nodes_at_step,
    const int step_number,
    const std::size_t max_walk_len,
    const int padding_value) {

    auto ws = std::make_unique<DeviceWalkSet>();
    ws->num_walks = last_nodes_at_step.size();
    ws->max_len   = max_walk_len;
    ws->walk_padding_value = padding_value;

    const std::size_t total_nodes = ws->num_walks * max_walk_len;
    std::vector<int> h_nodes(total_nodes, padding_value);
    for (std::size_t w = 0; w < ws->num_walks; ++w) {
        h_nodes[w * max_walk_len + static_cast<std::size_t>(step_number)] =
            last_nodes_at_step[w];
    }

    cudaMalloc(&ws->nodes, total_nodes * sizeof(int));
    cudaMemcpy(ws->nodes, h_nodes.data(),
               total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    // Other buffers: contents don't matter for scheduler-only tests.
    cudaMalloc(&ws->timestamps, total_nodes * sizeof(int64_t));
    cudaMalloc(&ws->walk_lens,  ws->num_walks * sizeof(std::size_t));
    const std::size_t edge_total =
        ws->num_walks * (max_walk_len > 0 ? max_walk_len - 1 : 0);
    if (edge_total > 0) {
        cudaMalloc(&ws->edge_ids, edge_total * sizeof(int64_t));
    }
    return ws;
}

WalkSetView make_view(const DeviceWalkSet& ws) {
    WalkSetView v{};
    v.nodes              = ws.nodes;
    v.timestamps         = ws.timestamps;
    v.walk_lens          = ws.walk_lens;
    v.edge_ids           = ws.edge_ids;
    v.num_walks          = ws.num_walks;
    v.max_len            = ws.max_len;
    v.walk_padding_value = ws.walk_padding_value;
    return v;
}

// --------------------------------------------------------------------------
// SchedulerResult — host-side materialization of a run_step output. Keeps
// pointers to the scheduler's arena-backed buffers off the public surface
// of each test; tests read the host vectors directly.
// --------------------------------------------------------------------------
// Tests here pass a trivial (all-zero) count_ts_group_per_node, so every
// node sees G=0 <= g_cap and the G-partition routes every warp/block task
// into its smem variant (the global variants should stay empty). Under
// that setup, num_warp_tasks_host below == the number of warp_smem tasks
// (same for block). The G-partition's real routing is covered by a
// sibling file, test_node_grouped_g_partition.cpp.
struct SchedulerResult {
    int num_active_host          = 0;
    int num_solo_walks_host      = 0;
    int num_warp_tasks_host      = 0;   // warp_smem (global expected empty)
    int num_block_tasks_host     = 0;   // block_smem (global expected empty)

    std::vector<int> sorted_walk_idx;
    std::vector<int> solo_walks;
    std::vector<int> warp_nodes;
    std::vector<int> warp_walk_starts;
    std::vector<int> warp_walk_counts;
    std::vector<int> block_nodes;
    std::vector<int> block_walk_starts;
    std::vector<int> block_walk_counts;
};

std::vector<int> download_ints(const int* device_ptr, const std::size_t n) {
    std::vector<int> h(n);
    if (n > 0 && device_ptr != nullptr) {
        cudaMemcpy(h.data(), device_ptr,
                   n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return h;
}

// Trivial count_ts_group_per_node — all zeros, so every node reports
// G = 0 groups, forcing the G-partition to route every task to the smem
// variant (G <= g_cap for any g_cap >= 0). Owned device buffer, freed
// when the returned unique_ptr drops.
struct DeviceG {
    std::size_t* ptr = nullptr;
    ~DeviceG() { if (ptr) cudaFree(ptr); }
};

std::unique_ptr<DeviceG> make_trivial_count_ts_group(int max_node_id) {
    auto g = std::make_unique<DeviceG>();
    // Size: (max_node_id + 2) so indices [0..max_node_id+1] are addressable
    // (the G-partition kernel reads [node] and [node+1]). Zero-initialize.
    const std::size_t n = static_cast<std::size_t>(max_node_id) + 2;
    cudaMalloc(&g->ptr, n * sizeof(std::size_t));
    cudaMemset(g->ptr, 0, n * sizeof(std::size_t));
    return g;
}

SchedulerResult run_and_materialize(
    const std::vector<int>& last_nodes_at_step,
    const int step_number,
    const std::size_t max_walk_len,
    const int padding_value,
    const dim3 block_dim,
    const cudaStream_t stream) {

    auto ws = make_walk_set(last_nodes_at_step, step_number,
                            max_walk_len, padding_value);
    const WalkSetView view = make_view(*ws);

    // Largest node_id that might appear in last_nodes_at_step (ignore
    // padding_value). Use that to size the count_ts_group_per_node array.
    int max_node_id = 0;
    for (int n : last_nodes_at_step) {
        if (n != padding_value && n > max_node_id) max_node_id = n;
    }
    auto g = make_trivial_count_ts_group(max_node_id);

    NodeGroupedScheduler scheduler(ws->num_walks, block_dim, stream);
    auto outs = scheduler.run_step(
        view, step_number, static_cast<int>(max_walk_len),
        g->ptr,
        RandomPickerType::Linear);   // picker class doesn't matter when G=0.

    // run_step already does its own stream sync. A final sync here is
    // defensive — guarantees the arena's buffers are stable before
    // we copy them down to host.
    cudaStreamSynchronize(stream);

    // With G=0 everywhere, the global tiers MUST be empty. Catching this
    // here flags a future change that breaks the W-partition tests'
    // setup assumption.
    EXPECT_EQ(outs.warp_global.num_tasks_host, 0)
        << "test setup invariant broken: warp_global should be empty "
           "when count_ts_group_per_node is all zeros";
    EXPECT_EQ(outs.block_global.num_tasks_host, 0)
        << "test setup invariant broken: block_global should be empty "
           "when count_ts_group_per_node is all zeros";

    SchedulerResult r;
    r.num_active_host      = outs.num_active_host;
    r.num_solo_walks_host  = outs.num_solo_walks_host;
    r.num_warp_tasks_host  = outs.warp_smem.num_tasks_host;
    r.num_block_tasks_host = outs.block_smem.num_tasks_host;

    r.sorted_walk_idx = download_ints(
        outs.sorted_walk_idx, static_cast<std::size_t>(outs.num_active_host));
    r.solo_walks = download_ints(
        outs.solo_walks, static_cast<std::size_t>(outs.num_solo_walks_host));
    r.warp_nodes = download_ints(
        outs.warp_smem.nodes, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.warp_walk_starts = download_ints(
        outs.warp_smem.walk_starts, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.warp_walk_counts = download_ints(
        outs.warp_smem.walk_counts, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.block_nodes = download_ints(
        outs.block_smem.nodes, static_cast<std::size_t>(r.num_block_tasks_host));
    r.block_walk_starts = download_ints(
        outs.block_smem.walk_starts, static_cast<std::size_t>(r.num_block_tasks_host));
    r.block_walk_counts = download_ints(
        outs.block_smem.walk_counts, static_cast<std::size_t>(r.num_block_tasks_host));

    return r;
}

// Collects the set of walk indices covered by the three tier outputs.
// Asserts no duplicates as it accumulates. Returns the set for caller
// to do additional coverage assertions.
std::set<int> collect_covered_walks(const SchedulerResult& r,
                                    std::string* duplicate_detail) {
    std::set<int> covered;
    auto insert_unique = [&](const int w, const char* tier) {
        if (!covered.insert(w).second) {
            if (duplicate_detail)
                *duplicate_detail = std::string("walk ") + std::to_string(w)
                    + " appears in " + tier + " and another tier";
        }
    };
    for (int w : r.solo_walks) insert_unique(w, "solo");
    for (std::size_t t = 0; t < r.warp_walk_counts.size(); ++t) {
        const int start = r.warp_walk_starts[t];
        const int count = r.warp_walk_counts[t];
        for (int k = 0; k < count; ++k) {
            insert_unique(r.sorted_walk_idx[start + k], "warp");
        }
    }
    for (std::size_t t = 0; t < r.block_walk_counts.size(); ++t) {
        const int start = r.block_walk_starts[t];
        const int count = r.block_walk_counts[t];
        for (int k = 0; k < count; ++k) {
            insert_unique(r.sorted_walk_idx[start + k], "block");
        }
    }
    return covered;
}

constexpr int PAD = EMPTY_NODE_VALUE;

}  // namespace

class GpuSchedulingTest : public ::testing::Test {
protected:
    void SetUp() override {
        block_dim_ = dim3(static_cast<unsigned>(BLOCK_DIM));
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    dim3 block_dim_{};
    cudaStream_t stream_ = nullptr;

    // step_number=1 means we need max_walk_len >= 2 for a valid slot.
    static constexpr int STEP_NUMBER   = 1;
    static constexpr int MAX_WALK_LEN  = 4;
};

// ==========================================================================
// Edge cases.
// ==========================================================================

TEST_F(GpuSchedulingTest, AllWalksTerminated_AllTierCountsZero) {
    // Every walk has walk_padding_value at step_number -> filtered out.
    std::vector<int> last_nodes(10, PAD);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 0);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);
}

TEST_F(GpuSchedulingTest, SingleActiveWalk_IsSolo) {
    // One active walk at node 42, rest terminated -> a single solo walk.
    std::vector<int> last_nodes(5, PAD);
    last_nodes[2] = 42;
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 1);
    EXPECT_EQ(r.num_solo_walks_host, 1);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);
    ASSERT_EQ(r.solo_walks.size(), 1u);
    EXPECT_EQ(r.solo_walks[0], 2);  // original walk_idx preserved
}

// ==========================================================================
// Tier routing by W.
// ==========================================================================

TEST_F(GpuSchedulingTest, AllUniqueNodes_AllWalksGoToSolo) {
    // Each walk at a distinct node -> every run has W=1 -> solo for all.
    std::vector<int> last_nodes;
    for (int i = 0; i < 10; ++i) last_nodes.push_back(100 + i);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 10);
    EXPECT_EQ(r.num_solo_walks_host, 10);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    // Each walk_idx [0..10) appears exactly once.
    std::set<int> solo_set(r.solo_walks.begin(), r.solo_walks.end());
    EXPECT_EQ(solo_set.size(), 10u);
    for (int w : solo_set) {
        EXPECT_GE(w, 0);
        EXPECT_LT(w, 10);
    }
}

TEST_F(GpuSchedulingTest, AllSameNodeWarpSize_SingleWarpTask) {
    // W=10 at node 42 -> single warp task.
    constexpr int W = 10;
    std::vector<int> last_nodes(W, 42);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    ASSERT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    EXPECT_EQ(r.warp_nodes[0], 42);
    EXPECT_EQ(r.warp_walk_starts[0], 0);
    EXPECT_EQ(r.warp_walk_counts[0], W);
}

TEST_F(GpuSchedulingTest, AllSameNodeBlockSize_SingleBlockTask) {
    // W>BLOCK_DIM -> single block task.
    const int W = static_cast<int>(BLOCK_DIM) + 1;  // 256, just over the boundary
    std::vector<int> last_nodes(W, 7);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    ASSERT_EQ(r.num_block_tasks_host, 1);

    EXPECT_EQ(r.block_nodes[0], 7);
    EXPECT_EQ(r.block_walk_starts[0], 0);
    EXPECT_EQ(r.block_walk_counts[0], W);
}

// ==========================================================================
// Tier boundaries — the partition's if-else cascade must be correct at the
// exact threshold values, not just comfortably above/below them.
// ==========================================================================

TEST_F(GpuSchedulingTest, BoundaryW1IsSolo_W2IsWarp) {
    // Run at node 100 with W=1 -> solo. Run at node 200 with W=2 -> warp.
    std::vector<int> last_nodes;
    last_nodes.push_back(100);
    last_nodes.push_back(200);
    last_nodes.push_back(200);

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 3);
    EXPECT_EQ(r.num_solo_walks_host, 1);
    EXPECT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    EXPECT_EQ(r.solo_walks[0], 0);       // walk_idx 0 was the W=1 one
    EXPECT_EQ(r.warp_nodes[0], 200);
    EXPECT_EQ(r.warp_walk_counts[0], 2);
}

TEST_F(GpuSchedulingTest, BoundaryW_BLOCK_DIM_IsWarp_AndW_BLOCK_DIMPlus1_IsBlock) {
    // The warp tier's upper bound (BLOCK_DIM) is inclusive: W=BLOCK_DIM
    // stays in warp; W=BLOCK_DIM+1 goes to block.
    const int W_warp  = static_cast<int>(BLOCK_DIM);
    const int W_block = static_cast<int>(BLOCK_DIM) + 1;

    std::vector<int> last_nodes;
    for (int i = 0; i < W_warp;  ++i) last_nodes.push_back(10);
    for (int i = 0; i < W_block; ++i) last_nodes.push_back(20);

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W_warp + W_block);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    ASSERT_EQ(r.num_warp_tasks_host, 1);
    ASSERT_EQ(r.num_block_tasks_host, 1);

    EXPECT_EQ(r.warp_walk_counts[0], W_warp);
    EXPECT_EQ(r.block_walk_counts[0], W_block);
}

// ==========================================================================
// Compositional invariants — count identity + disjoint coverage.
// ==========================================================================

TEST_F(GpuSchedulingTest, MixedTiers_CountIdentityHolds) {
    // Solo: 5 walks at 5 distinct nodes.
    // Warp: 10 walks at node 42.
    // Block: 300 walks at node 7.
    std::vector<int> last_nodes;
    for (int i = 0; i < 5;   ++i) last_nodes.push_back(201 + i);
    for (int i = 0; i < 10;  ++i) last_nodes.push_back(42);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(7);
    const int expected_active = 5 + 10 + 300;

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, expected_active);
    EXPECT_EQ(r.num_solo_walks_host, 5);
    EXPECT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 1);

    int total = r.num_solo_walks_host;
    total += std::accumulate(r.warp_walk_counts.begin(),
                             r.warp_walk_counts.end(), 0);
    total += std::accumulate(r.block_walk_counts.begin(),
                             r.block_walk_counts.end(), 0);
    EXPECT_EQ(total, expected_active)
        << "num_solo + sum(warp_counts) + sum(block_counts) "
        << "must equal num_active";
}

TEST_F(GpuSchedulingTest, MixedWithTermination_DisjointUnionEqualsActive) {
    // 5 solo (unique nodes), 10 warp at node 42, 5 terminated, 10 warp at 99.
    // Active = 25; partition must cover exactly those 25 with no duplicates
    // and no terminated walks.
    std::vector<int> last_nodes(30);
    std::vector<int> alive_indices;
    for (int i = 0; i < 5;  ++i) { last_nodes[i] = 201 + i;   alive_indices.push_back(i); }
    for (int i = 5; i < 15; ++i) { last_nodes[i] = 42;        alive_indices.push_back(i); }
    for (int i = 15; i < 20; ++i) last_nodes[i] = PAD;
    for (int i = 20; i < 30; ++i) { last_nodes[i] = 99;       alive_indices.push_back(i); }

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 25);
    EXPECT_EQ(r.num_solo_walks_host, 5);
    EXPECT_EQ(r.num_warp_tasks_host, 2);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    std::string dup_detail;
    auto covered = collect_covered_walks(r, &dup_detail);
    EXPECT_TRUE(dup_detail.empty()) << dup_detail;

    EXPECT_EQ(covered.size(), 25u);
    for (int i : alive_indices) {
        EXPECT_EQ(covered.count(i), 1u)
            << "alive walk " << i << " missing from partition output";
    }
    for (int i = 15; i < 20; ++i) {
        EXPECT_EQ(covered.count(i), 0u)
            << "terminated walk " << i << " leaked into partition output";
    }
}

TEST_F(GpuSchedulingTest, SoloTierPreservesOriginalWalkIdx) {
    // All walks at distinct nodes -> every run goes to solo and every
    // walk_idx [0..N) must appear exactly once in solo_walks. Also
    // verifies that solo_walks carries the ORIGINAL walk_idx (not any
    // dense renumbering) even when some walks are terminated.
    std::vector<int> last_nodes(20, PAD);
    std::vector<int> expected_solo;
    const int alive_indices[] = {3, 5, 8, 11, 13, 17, 19};
    int node = 500;
    for (int i : alive_indices) {
        last_nodes[i] = node++;   // unique node per alive walk
        expected_solo.push_back(i);
    }

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, static_cast<int>(expected_solo.size()));
    EXPECT_EQ(r.num_solo_walks_host, static_cast<int>(expected_solo.size()));

    std::set<int> got(r.solo_walks.begin(), r.solo_walks.end());
    std::set<int> expected(expected_solo.begin(), expected_solo.end());
    EXPECT_EQ(got, expected);
}

#endif  // HAS_CUDA
