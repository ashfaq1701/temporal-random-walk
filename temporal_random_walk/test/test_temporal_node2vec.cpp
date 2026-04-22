#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "../src/core/temporal_random_walk.cuh"
#include "../src/graph/edge_data.cuh"
#include "../src/graph/temporal_node2vec_helpers.cuh"
#include "test_temporal_graph_utils.h"

namespace {

bool is_sentinel(const Edge& e) {
    return e.u == -1 && e.i == -1 && e.ts == -1;
}

bool contains_edge(const std::vector<Edge>& v, const Edge& e) {
    return std::any_of(v.begin(), v.end(), [&](const Edge& x) {
        return x.u == e.u && x.i == e.i && x.ts == e.ts;
    });
}

template<typename UseGpu>
class TemporalNode2VecTest : public ::testing::Test {
protected:
    static constexpr bool use_gpu = UseGpu::value;

    core::TemporalRandomWalk graph{
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/2.0,
        /*node2vec_q=*/0.5
    };

    void SetUp() override {
        test_util::add_edges(graph, {
            Edge{0, 5, 10},
            Edge{0, 42, 10},
            Edge{0, 1000, 15},
            Edge{0, 7, 20},
            Edge{5, 42, 5},
            Edge{1000, 1, 6},
            Edge{7, 3, 8}
        });
    }

    [[nodiscard]] std::vector<Edge> collect_outbound_edges(const int u) const {
        const auto all = graph.get_edges();
        std::vector<Edge> out;
        for (const auto& e : all) {
            if (e.u == u) out.push_back(e);
        }
        return out;
    }
};

#ifdef HAS_CUDA
using Backends = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using Backends = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

TYPED_TEST_SUITE(TemporalNode2VecTest, Backends);

TYPED_TEST(TemporalNode2VecTest, BetaRulesFollowNode2VecDefinition) {
    const int prev = 5;
    EXPECT_DOUBLE_EQ(test_util::compute_node2vec_beta(this->graph.data(), prev, prev), 1.0 / 2.0);
    EXPECT_DOUBLE_EQ(test_util::compute_node2vec_beta(this->graph.data(), prev, 42),   1.0);
    EXPECT_DOUBLE_EQ(test_util::compute_node2vec_beta(this->graph.data(), prev, 7),    1.0 / 0.5);
}

TYPED_TEST(TemporalNode2VecTest, Tn2vWithoutPrevNodeReturnsValidOutboundEdge) {
    const auto outbound = this->collect_outbound_edges(0);
    ASSERT_FALSE(outbound.empty());

    const Edge picked = test_util::get_node_edge_at(
        this->graph.data(), 0, RandomPickerType::TemporalNode2Vec, -1, -1, true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

TYPED_TEST(TemporalNode2VecTest, Tn2vWithValidPrevNodeReturnsValidOutboundEdge) {
    const auto outbound = this->collect_outbound_edges(0);
    ASSERT_FALSE(outbound.empty());

    const int prev_node = outbound.front().i;
    const Edge picked = test_util::get_node_edge_at(
        this->graph.data(), 0, RandomPickerType::TemporalNode2Vec, -1, prev_node, true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

TYPED_TEST(TemporalNode2VecTest, Tn2vWithUnrelatedPrevNodeIsSafe) {
    const auto outbound = this->collect_outbound_edges(0);
    ASSERT_FALSE(outbound.empty());

    const Edge picked = test_util::get_node_edge_at(
        this->graph.data(), 0, RandomPickerType::TemporalNode2Vec, -1, 9999, true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

TYPED_TEST(TemporalNode2VecTest, BackwardTn2vFromNodeWithNoInboundReturnsSentinel) {
    const Edge picked = test_util::get_node_edge_at(
        this->graph.data(), 0, RandomPickerType::TemporalNode2Vec, -1, 5, false);
    EXPECT_TRUE(is_sentinel(picked));
}

TYPED_TEST(TemporalNode2VecTest, NodeAdjacencyCSRIsValid) {
    const auto snap = edge_data::snapshot(this->graph.data());
    const auto& offsets = snap.node_adj_offsets;
    const auto& neighbors = snap.node_adj_neighbors;

    ASSERT_FALSE(offsets.empty());
    ASSERT_EQ(offsets.back(), neighbors.size());

    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        ASSERT_LE(offsets[i], offsets[i + 1]);
    }

    for (int v : neighbors) {
        ASSERT_GE(v, 0);
        ASSERT_LT(static_cast<size_t>(v), offsets.size() - 1);
    }
}

// ==========================================================================
// Distribution tests — the earlier tests only verify "returns valid edge";
// these verify the p/q bias actually produces the correct sampling
// distribution. If the beta multiplier were dropped or the wrong
// prev_node were consulted, these assertions would fail by a wide margin.
//
// Topology: V has outbound (Forward test) or inbound (Backward test)
// edges to three targets at the same timestamp so they share a
// ts-group and the per-group exp-weight factor cancels out. The three
// targets cover each beta class:
//   - P itself   → return case,   beta = 1/p
//   - N          → 1-step neighbor of P (via P-N edge), beta = 1
//   - F          → non-neighbor,   beta = 1/q
//
// With p=2, q=0.5: ratios (1/p : 1 : 1/q) = (0.5 : 1 : 2) →
// normalized (1/7, 2/7, 4/7) ≈ (0.143, 0.286, 0.571).
//
// 10_000 samples → σ ≈ √(p(1-p)/N) ≈ 0.005 for the largest mass; a
// tolerance of ±0.02 is ~4σ — safe against normal sampling noise,
// tight enough that a missing or miscomputed bias would fail.
// ==========================================================================

namespace dist_test {
constexpr int V = 0;   // current node
constexpr int P = 1;   // prev_node (also a target in forward test)
constexpr int N = 2;   // 1-step neighbor of P
constexpr int F = 3;   // non-neighbor of P

constexpr int      NUM_SAMPLES = 10000;
constexpr int64_t  EDGE_TS     = 100;
constexpr int64_t  ADJ_EDGE_TS = 50;  // P-N adjacency edge; must NOT
                                      // appear at V (so V's ts-group
                                      // stays a single group at EDGE_TS).
constexpr double   EXPECTED_P  = 1.0 / 7.0;
constexpr double   EXPECTED_N  = 2.0 / 7.0;
constexpr double   EXPECTED_F  = 4.0 / 7.0;
constexpr double   TOLERANCE   = 0.02;
}  // namespace dist_test

TYPED_TEST(TemporalNode2VecTest, ForwardDistributionFollowsPQBias) {
    using namespace dist_test;
    constexpr bool use_gpu = TypeParam::value;

    core::TemporalRandomWalk bias_graph{
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/2.0,
        /*node2vec_q=*/0.5,
    };

    test_util::add_edges(bias_graph, {
        // V's outbound edges — all in a single ts-group at EDGE_TS so
        // the per-group exp-weight is identical across them.
        Edge{V, P, EDGE_TS},
        Edge{V, N, EDGE_TS},
        Edge{V, F, EDGE_TS},
        // P-N adjacency edge; makes is_node_adjacent_to(P, N) == true.
        Edge{P, N, ADJ_EDGE_TS},
    });

    int count_p = 0, count_n = 0, count_f = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        const Edge picked = test_util::get_node_edge_at(
            bias_graph.data(), V,
            RandomPickerType::TemporalNode2Vec,
            /*timestamp=*/-1, /*prev_node=*/P, /*forward=*/true);
        ASSERT_FALSE(is_sentinel(picked));
        ASSERT_EQ(picked.u, V);
        if      (picked.i == P) ++count_p;
        else if (picked.i == N) ++count_n;
        else if (picked.i == F) ++count_f;
        else FAIL() << "Unexpected target: " << picked.i;
    }

    const double frac_p = static_cast<double>(count_p) / NUM_SAMPLES;
    const double frac_n = static_cast<double>(count_n) / NUM_SAMPLES;
    const double frac_f = static_cast<double>(count_f) / NUM_SAMPLES;

    EXPECT_NEAR(frac_p, EXPECTED_P, TOLERANCE)
        << "Return edge frac mismatch: got " << frac_p
        << ", expected " << EXPECTED_P;
    EXPECT_NEAR(frac_n, EXPECTED_N, TOLERANCE)
        << "Neighbor-of-prev edge frac mismatch: got " << frac_n
        << ", expected " << EXPECTED_N;
    EXPECT_NEAR(frac_f, EXPECTED_F, TOLERANCE)
        << "Non-adjacent edge frac mismatch: got " << frac_f
        << ", expected " << EXPECTED_F;
}

TYPED_TEST(TemporalNode2VecTest, BackwardDistributionFollowsPQBias) {
    using namespace dist_test;
    constexpr bool use_gpu = TypeParam::value;

    core::TemporalRandomWalk bias_graph{
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/2.0,
        /*node2vec_q=*/0.5,
    };

    test_util::add_edges(bias_graph, {
        // V's inbound edges — candidate for each is the SOURCE in the
        // backward path. Same ts-group so exp-weight is identical.
        Edge{P, V, EDGE_TS},
        Edge{N, V, EDGE_TS},
        Edge{F, V, EDGE_TS},
        // P-N adjacency edge.
        Edge{P, N, ADJ_EDGE_TS},
    });

    int count_p = 0, count_n = 0, count_f = 0;
    // Backward filter keeps groups with ts < timestamp; pass a large
    // timestamp so EDGE_TS (100) passes the cutoff.
    constexpr int64_t BACKWARD_CUTOFF = 10'000;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        const Edge picked = test_util::get_node_edge_at(
            bias_graph.data(), V,
            RandomPickerType::TemporalNode2Vec,
            /*timestamp=*/BACKWARD_CUTOFF, /*prev_node=*/P, /*forward=*/false);
        ASSERT_FALSE(is_sentinel(picked));
        ASSERT_EQ(picked.i, V);  // target is V in the inbound set
        // Candidate (source) is what determines the beta class.
        if      (picked.u == P) ++count_p;
        else if (picked.u == N) ++count_n;
        else if (picked.u == F) ++count_f;
        else FAIL() << "Unexpected source: " << picked.u;
    }

    const double frac_p = static_cast<double>(count_p) / NUM_SAMPLES;
    const double frac_n = static_cast<double>(count_n) / NUM_SAMPLES;
    const double frac_f = static_cast<double>(count_f) / NUM_SAMPLES;

    EXPECT_NEAR(frac_p, EXPECTED_P, TOLERANCE);
    EXPECT_NEAR(frac_n, EXPECTED_N, TOLERANCE);
    EXPECT_NEAR(frac_f, EXPECTED_F, TOLERANCE);
}

} // namespace
