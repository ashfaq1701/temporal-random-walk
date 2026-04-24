// End-to-end tests that drive each NODE_GROUPED cooperative kernel body
// (block-smem, block-global, warp-smem, warp-global) by constructing
// graphs that force the W-partition/G-partition to route at least one
// task into the target tier. The test body doesn't directly assert
// "this tier ran" — the scheduler's routing isn't exposed through the
// public walk API — but under the crafted graph topology, the only way
// all walks can come out valid is if the target kernel executed without
// corrupting walks. A tier-body bug (wrong smem indexing, off-by-one in
// the stride loop, missing __syncwarp, etc.) shows up as a walk with a
// sentinel node in the middle or a non-monotone timestamp.
//
// Topology used across the tier tests is a convergence graph:
//   - N source nodes (0 .. N-1), each with exactly one edge to the hub
//     at the SAME timestamp (ts=0).
//   - Hub node H with G outbound edges at G distinct timestamps
//     (ts = 1 .. G), each going to a unique destination.
//
// With num_walks_per_node = 1, every source spawns one walk that arrives
// at H at step 1. Together they create a (node=H, step=1) group with
// W = N. The W and G values determine which tier services that task:
//
//   Warp-smem   : W ∈ [2, BLOCK_DIM] AND G ≤ 340   (index-picker cap)
//   Warp-global : W ∈ [2, BLOCK_DIM]     AND G >  340
//   Block-smem  : W >  BLOCK_DIM          AND G ≤ 2800
//   Block-global: W >  BLOCK_DIM          AND G >  2800
//
// This file also contains a Node2Vec smoke test — the dispatcher gates
// Node2Vec out of the cooperative pipeline entirely and runs it on
// per_walk_step_kernel, so tier routing doesn't apply, but the walk
// validity story is the same.
//
// All tests use the Linear picker (index-based, doesn't need weights)
// except the Node2Vec one. Picker × tier parity is covered by
// test_node_grouped_parity.cpp on a realistic graph; this file's job
// is to guarantee each tier kernel executes at least once in CI.

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"

namespace {

constexpr int      MAX_WALK_LEN      = 10;
constexpr uint64_t TIER_SEED         = 0xC0FFEEULL;
constexpr int      HUB_NODE          = 100000;   // well above source/dest ranges
constexpr int      DEST_NODE_BASE    = 200000;

constexpr RandomPickerType LINEAR_PICKER    = RandomPickerType::Linear;
constexpr RandomPickerType NODE2VEC_PICKER  = RandomPickerType::TemporalNode2Vec;

#ifdef HAS_CUDA
using TIER_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using TIER_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

template <typename T>
class NodeGroupedTierRoutingTest : public ::testing::Test {
protected:
    // Build a convergence graph (see file header). Returns the edge list
    // ready to hand to TemporalRandomWalk::add_multiple_edges.
    static std::vector<std::tuple<int, int, int64_t>>
    convergence_graph(const int num_sources, const int hub_groups) {
        std::vector<std::tuple<int, int, int64_t>> edges;
        edges.reserve(static_cast<size_t>(num_sources + hub_groups));
        for (int s = 0; s < num_sources; ++s) {
            edges.emplace_back(s, HUB_NODE, int64_t{0});
        }
        for (int g = 0; g < hub_groups; ++g) {
            edges.emplace_back(HUB_NODE, DEST_NODE_BASE + g,
                               static_cast<int64_t>(g + 1));
        }
        return edges;
    }

    std::unique_ptr<TemporalRandomWalk> make_trw(
        const bool enable_weights,
        const bool enable_node2vec = false) const {
        return std::make_unique<TemporalRandomWalk>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/enable_weights,
            /*enable_temporal_node2vec=*/enable_node2vec,
            DEFAULT_TIMESCALE_BOUND,
            DEFAULT_NODE2VEC_P,
            DEFAULT_NODE2VEC_Q,
            EMPTY_NODE_VALUE,
            /*global_seed=*/TIER_SEED,
            /*shuffle_walk_order=*/false);
    }
};

TYPED_TEST_SUITE(NodeGroupedTierRoutingTest, TIER_BACKENDS);

// Walk validity: every hop is a real node and timestamps are monotone
// non-decreasing. Both forward and backward walks are reversed in place
// post-sampling, so the caller-visible timestamp order is the same
// (earliest real ts first, latest real ts last, sentinel at whichever
// end is free). cur >= prev holds either way.
void assert_all_walks_valid(const WalksWithEdgeFeaturesHost& walks,
                            const size_t max_walk_len) {
    const int*     nodes = walks.walk_set.nodes_ptr();
    const int64_t* ts    = walks.walk_set.timestamps_ptr();
    const size_t*  lens  = walks.walk_set.walk_lens_ptr();
    const size_t num_walks = walks.walk_set.num_walks();

    size_t produced = 0;
    for (size_t w = 0; w < num_walks; ++w) {
        const size_t wl = lens[w];
        if (wl == 0) continue;
        ++produced;
        const size_t base = w * max_walk_len;
        for (size_t i = 0; i < wl; ++i) {
            ASSERT_NE(nodes[base + i], EMPTY_NODE_VALUE)
                << "walk " << w << " hop " << i << " is the -1 sentinel";
        }
        for (size_t i = 1; i < wl; ++i) {
            ASSERT_GE(ts[base + i], ts[base + i - 1])
                << "walk " << w << " ts non-monotone between hop "
                << (i - 1) << " (ts=" << ts[base + i - 1]
                << ") and hop " << i << " (ts=" << ts[base + i] << ")";
        }
    }
    ASSERT_GT(produced, 0u) << "no walks produced — kernel almost certainly broke.";
}

}  // namespace

// ==========================================================================
// Warp-smem tier: W ∈ [2, 255] AND G ≤ 340.
//
// 10 sources converge at HUB_NODE at step 1. W = 10 ∈ [2, 255].
// Hub has 20 distinct-timestamp outbound groups. G = 20 ≤ 340.
// Scheduler's W-partition -> warp task list, G-partition -> warp_smem.
// ==========================================================================
TYPED_TEST(NodeGroupedTierRoutingTest, WarpSmem_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/20));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// ==========================================================================
// Warp-global tier: W ∈ [2, 255] AND G > 340.
//
// 10 sources -> HUB with 400 distinct-timestamp outbound groups.
// G = 400 > G_THRESHOLD_WARP_INDEX (340) -> warp_global.
// ==========================================================================
TYPED_TEST(NodeGroupedTierRoutingTest, WarpGlobal_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/400));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// ==========================================================================
// Block-smem tier: W > 255 AND G ≤ 2800.
//
// 300 sources -> HUB with 100 distinct-timestamp outbound groups.
// W = 300 > BLOCK_DIM (255); G = 100 ≤ 2800 -> block_smem.
// ==========================================================================
TYPED_TEST(NodeGroupedTierRoutingTest, BlockSmem_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/300, /*hub_groups=*/100));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// ==========================================================================
// Block-global tier: W > 255 AND G > 2800.
//
// 300 sources -> HUB with 3000 distinct-timestamp outbound groups.
// W = 300 > BLOCK_DIM; G = 3000 > G_THRESHOLD_BLOCK_INDEX (2800)
// -> block_global. This tier had zero end-to-end coverage before.
// ==========================================================================
TYPED_TEST(NodeGroupedTierRoutingTest, BlockGlobal_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/300, /*hub_groups=*/3000));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// ==========================================================================
// Node2Vec smoke test: the dispatcher gates Node2Vec out of the
// cooperative pipeline entirely (prev_node-dependent sampling breaks
// panel sharing) and routes every intermediate step through
// per_walk_step_kernel instead. Tier routing doesn't apply, so the
// convergence shape is used only to give Node2Vec walks a non-trivial
// step-1 transition. Parity with FULL_WALK is not asserted here; the
// two paths step Philox counters differently. This is just a "Node2Vec
// NODE_GROUPED runs end-to-end and produces valid walks" smoke check.
// ==========================================================================
TYPED_TEST(NodeGroupedTierRoutingTest, Node2Vec_NodeGroupedSmokeTest) {
    auto trw = this->make_trw(/*enable_weights=*/true,
                              /*enable_node2vec=*/true);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/20));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &NODE2VEC_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}
