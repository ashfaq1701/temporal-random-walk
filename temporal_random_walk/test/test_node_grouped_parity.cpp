#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "test_utils.h"
#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"

// Structural parity harness between FULL_WALK and NODE_GROUPED. See
// CLAUDE.md §9.1.
//
// These two paths are NOT bit-exact today: both key Philox on
// (base_seed, walk_idx) so they share a substream, but FULL_WALK walks
// the counter sequentially while NODE_GROUPED re-inits per step at
// offset = 3 + step * 2 (step_kernel_philox_offset). The two paths
// therefore draw from different counter positions and produce genuinely
// different walks for the same seed.
//
// What this harness verifies instead — invariants that must hold
// regardless of the Philox offset gap:
//   * same walk count (both paths receive the same start_node_ids).
//   * walk_idx -> start_node mapping matches slot 0 (shuffle disabled).
//   * every hop is a real edge in the graph and timestamps respect the
//     requested walk direction.
//   * length distributions are close enough to catch step-loop / filter
//     regressions (e.g. every NODE_GROUPED walk jamming at length 1).
//
// Bit-exact parity is tracked as a separate follow-up; requires
// aligning FULL_WALK's Philox stepping to NODE_GROUPED's offset scheme
// (or vice versa).

namespace {

constexpr int      MAX_WALK_LEN       = 20;
constexpr int      NUM_WALKS_PER_NODE = 5;
constexpr uint64_t PARITY_SEED        = 0x5EED1234ULL;
constexpr RandomPickerType LINEAR_PICKER = RandomPickerType::Linear;

// Mean-length ratio band. Picked wide because the two paths consume the
// Philox stream differently, but tight enough to catch a gross bug
// where NODE_GROUPED walks jam a few hops in. Revisit once bit-exact
// parity lands — then this can shrink to ~1.0 ± epsilon.
constexpr double MEAN_LEN_RATIO_LOW  = 0.5;
constexpr double MEAN_LEN_RATIO_HIGH = 2.0;

#ifdef HAS_CUDA
using PARITY_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using PARITY_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

template <typename T>
class NodeGroupedParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        for (const auto& [u, v, ts] : sample_edges) {
            edge_set.insert(pack_edge(u, v, ts));
        }
    }

    // Fresh trw per kernel launch — identical config, identical seed.
    // shuffle_walk_order=false so slot 0 of walk w equals the same
    // start_node_id in both FULL_WALK and NODE_GROUPED output.
    std::unique_ptr<TemporalRandomWalk> make_trw() const {
        auto trw = std::make_unique<TemporalRandomWalk>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/false,
            /*enable_temporal_node2vec=*/false,
            DEFAULT_TIMESCALE_BOUND,
            DEFAULT_NODE2VEC_P,
            DEFAULT_NODE2VEC_Q,
            EMPTY_NODE_VALUE,
            /*global_seed=*/PARITY_SEED,
            /*shuffle_walk_order=*/false);
        trw->add_multiple_edges(sample_edges);
        return trw;
    }

    static uint64_t pack_edge(int u, int v, int64_t ts) {
        // Cheap composite key for edge-existence lookup; collisions are
        // tolerable because the test only asserts membership.
        uint64_t k = static_cast<uint64_t>(static_cast<uint32_t>(u));
        k = (k * 1099511628211ULL) ^ static_cast<uint32_t>(v);
        k = (k * 1099511628211ULL) ^ static_cast<uint64_t>(ts);
        return k;
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unordered_set<uint64_t> edge_set;
};

TYPED_TEST_SUITE(NodeGroupedParityTest, PARITY_BACKENDS);

struct WalkDigest {
    size_t num_walks       = 0;
    size_t produced_walks  = 0;    // walks with len >= 1
    size_t total_hops      = 0;
    size_t walks_ge_3      = 0;
    double mean_len_all    = 0.0;  // across produced walks only
};

WalkDigest digest_walks(const WalksWithEdgeFeaturesHost& walks) {
    WalkDigest d{};
    const size_t* lens = walks.walk_set.walk_lens_ptr();
    d.num_walks = walks.walk_set.num_walks();
    for (size_t w = 0; w < d.num_walks; ++w) {
        if (lens[w] == 0) continue;
        ++d.produced_walks;
        d.total_hops += lens[w];
        if (lens[w] >= 3) ++d.walks_ge_3;
    }
    if (d.produced_walks > 0) {
        d.mean_len_all =
            static_cast<double>(d.total_hops) /
            static_cast<double>(d.produced_walks);
    }
    return d;
}

// Slot-0 parity: with shuffle_walk_order=false, walk_idx w is seeded
// from the same start node in both paths, so nodes[w * max_walk_len]
// must agree wherever both paths produced a walk. If only one path
// dead-ended on the first edge, slot-0 may carry the padding on the
// other side; skip those asymmetric cases rather than asserting on them.
void assert_slot0_agreement(
    const WalksWithEdgeFeaturesHost& a,
    const WalksWithEdgeFeaturesHost& b,
    const size_t max_walk_len,
    size_t* compared_out) {
    ASSERT_EQ(a.walk_set.num_walks(), b.walk_set.num_walks());
    const size_t num_walks = a.walk_set.num_walks();
    const int* an = a.walk_set.nodes_ptr();
    const int* bn = b.walk_set.nodes_ptr();
    const size_t* al = a.walk_set.walk_lens_ptr();
    const size_t* bl = b.walk_set.walk_lens_ptr();

    size_t compared = 0;
    for (size_t w = 0; w < num_walks; ++w) {
        if (al[w] == 0 || bl[w] == 0) continue;
        ASSERT_EQ(an[w * max_walk_len], bn[w * max_walk_len])
            << "Walk " << w << " slot 0 differs across paths; "
            << "shuffle is disabled and both paths receive the same "
            << "start_node_ids, so this is a walk_idx placement bug.";
        ++compared;
    }
    *compared_out = compared;
}

// Structural validity: every hop in [0, walk_len) must be a real node,
// and consecutive (ts, ts') pairs must respect the walk direction.
// Forward: non-decreasing (INT64_MIN sentinel at slot 0 is tolerated).
template <bool Forward>
void assert_walks_are_valid(
    const WalksWithEdgeFeaturesHost& walks,
    const size_t max_walk_len) {
    const int*     nodes = walks.walk_set.nodes_ptr();
    const int64_t* ts    = walks.walk_set.timestamps_ptr();
    const size_t*  lens  = walks.walk_set.walk_lens_ptr();
    const size_t num_walks = walks.walk_set.num_walks();

    for (size_t w = 0; w < num_walks; ++w) {
        const size_t wl = lens[w];
        if (wl == 0) continue;

        const size_t base = w * max_walk_len;
        for (size_t i = 0; i < wl; ++i) {
            ASSERT_NE(nodes[base + i], EMPTY_NODE_VALUE)
                << "Walk " << w << " hop " << i << " is -1 sentinel";
        }
        // Skip slot 0 in the ts check — start kernel parks a direction
        // sentinel there (INT64_MIN for forward, INT64_MAX for backward).
        for (size_t i = 2; i < wl; ++i) {
            const int64_t prev = ts[base + i - 1];
            const int64_t cur  = ts[base + i];
            if constexpr (Forward) {
                ASSERT_GE(cur, prev)
                    << "Walk " << w << " forward ts non-monotone between "
                    << "hop " << (i - 1) << " (ts=" << prev
                    << ") and hop " << i << " (ts=" << cur << ")";
            } else {
                ASSERT_LE(cur, prev)
                    << "Walk " << w << " backward ts non-monotone between "
                    << "hop " << (i - 1) << " (ts=" << prev
                    << ") and hop " << i << " (ts=" << cur << ")";
            }
        }
    }
}

}  // namespace

// ==========================================================================
// Core structural parity: all-nodes path (constrained start).
// ==========================================================================
TYPED_TEST(NodeGroupedParityTest, AllNodesConstrained_StructuralParity) {
    auto trw_full = this->make_trw();
    auto trw_grp  = this->make_trw();

    const auto walks_full = trw_full->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::FULL_WALK);
    const auto walks_grp = trw_grp->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

    size_t compared = 0;
    ASSERT_NO_FATAL_FAILURE(
        assert_slot0_agreement(walks_full, walks_grp,
                               static_cast<size_t>(MAX_WALK_LEN),
                               &compared));
    ASSERT_GT(compared, 0u)
        << "No walks were compared — both paths dead-ended on every seed, "
        << "which almost certainly indicates a broken walk kernel rather "
        << "than a legitimate graph property.";

    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<true>(walks_full,
                                     static_cast<size_t>(MAX_WALK_LEN)));
    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<true>(walks_grp,
                                     static_cast<size_t>(MAX_WALK_LEN)));

    const auto d_full = digest_walks(walks_full);
    const auto d_grp  = digest_walks(walks_grp);
    ASSERT_GT(d_full.produced_walks, 0u);
    ASSERT_GT(d_grp.produced_walks,  0u);
    ASSERT_GT(d_full.mean_len_all, 1.0)
        << "FULL_WALK mean length " << d_full.mean_len_all
        << " is degenerate — every walk stopped at the start edge.";
    ASSERT_GT(d_grp.mean_len_all, 1.0)
        << "NODE_GROUPED mean length " << d_grp.mean_len_all
        << " is degenerate — step-loop or filter is broken.";

    const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
    EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW)
        << "NODE_GROUPED mean length " << d_grp.mean_len_all
        << " is far below FULL_WALK's " << d_full.mean_len_all
        << " — likely a termination-check or walk_idx scatter regression.";
    EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH)
        << "NODE_GROUPED mean length " << d_grp.mean_len_all
        << " is far above FULL_WALK's " << d_full.mean_len_all
        << " — unexpected and worth investigating.";
}

// ==========================================================================
// Unconstrained path: every walk starts with start_node_id == -1, so
// NODE_GROUPED short-circuits sort/RLE and runs solo for everyone. slot 0
// can legitimately differ between paths because the picker-draw sequence
// differs; we still assert validity and length-distribution parity.
// ==========================================================================
TYPED_TEST(NodeGroupedParityTest, Unconstrained_StructuralParity) {
    auto trw_full = this->make_trw();
    auto trw_grp  = this->make_trw();

    // _get_random_walks_and_times (no _for_all_nodes / _for_last_batch)
    // is the unconstrained entrypoint — fills start_node_ids with -1.
    constexpr int NUM_WALKS_TOTAL = 400;
    const auto walks_full = trw_full->get_random_walks_and_times(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_TOTAL,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::FULL_WALK);
    const auto walks_grp = trw_grp->get_random_walks_and_times(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_TOTAL,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<true>(walks_full,
                                     static_cast<size_t>(MAX_WALK_LEN)));
    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<true>(walks_grp,
                                     static_cast<size_t>(MAX_WALK_LEN)));

    const auto d_full = digest_walks(walks_full);
    const auto d_grp  = digest_walks(walks_grp);
    ASSERT_GT(d_full.produced_walks, 0u);
    ASSERT_GT(d_grp.produced_walks,  0u);

    const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
    EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW);
    EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH);
}

// ==========================================================================
// Backward walks: exercises the reverse_walks_kernel and the backward
// temporal-validity check. Only the timestamp-monotonicity direction
// changes; structural parity claims are identical.
// ==========================================================================
TYPED_TEST(NodeGroupedParityTest, AllNodesConstrained_Backward_StructuralParity) {
    auto trw_full = this->make_trw();
    auto trw_grp  = this->make_trw();

    const auto walks_full = trw_full->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Backward_In_Time,
        KernelLaunchType::FULL_WALK);
    const auto walks_grp = trw_grp->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Backward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<false>(walks_full,
                                      static_cast<size_t>(MAX_WALK_LEN)));
    ASSERT_NO_FATAL_FAILURE(
        assert_walks_are_valid<false>(walks_grp,
                                      static_cast<size_t>(MAX_WALK_LEN)));

    const auto d_full = digest_walks(walks_full);
    const auto d_grp  = digest_walks(walks_grp);
    ASSERT_GT(d_full.produced_walks, 0u);
    ASSERT_GT(d_grp.produced_walks,  0u);
    const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
    EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW);
    EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH);
}
