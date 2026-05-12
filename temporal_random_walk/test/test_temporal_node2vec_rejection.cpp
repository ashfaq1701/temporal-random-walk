// Hardening tests for the Stage-2 temporal-node2vec rejection sampler.
//
// The base test_temporal_node2vec.cpp already covers the headline
// invariants (β rules, sentinel handling, adjacency-CSR validity, and
// distribution-following under the default p, q).  These additional
// tests cover the parts of the rejection-sampling implementation that
// the base tests don't directly exercise:
//
//   1. Retry-loop termination under degenerate (p, q) — confirms the
//      retry cap + defensive accept-last keeps the picker returning
//      valid edges (no sentinels, no infinite loops) even when the
//      acceptance ratio is bad.  Performance-regression guard.
//
//   2. Distribution-following across a (p, q) sweep — generalises the
//      base PQBias test to the full node2vec literature parameter
//      sweep {(1,1), (1,2), (0.5,2), (2,0.5), (0.25,4)}.  Catches β-
//      handling bugs at the parameter boundaries that the default
//      (p=2, q=0.5) cell happens to miss.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/core/temporal_random_walk.cuh"
#include "test_temporal_graph_utils.h"

namespace {

constexpr bool kIsDirected      = true;
constexpr int64_t kSentinelTs   = -1;
constexpr int  kSentinelNode    = -1;

inline bool is_sentinel(const Edge& e) {
    return e.u == kSentinelNode && e.i == kSentinelNode && e.ts == kSentinelTs;
}

// ---------------------------------------------------------------------------
// Shared n2v graph layout — three outbound destinations from V, one of which
// (P) is also adjacent to one of the others (N) via an extra edge.  Same
// shape as the base PQBias test (dist_test in test_temporal_node2vec.cpp)
// so the analytic distribution is straightforward to compute.
// ---------------------------------------------------------------------------
constexpr int      V            = 0;
constexpr int      P            = 1;     // "previous": β = 1/p
constexpr int      N            = 2;     // "neighbor of P": β = 1
constexpr int      F            = 3;     // "far from P":   β = 1/q
constexpr int64_t  EDGE_TS      = 100;
constexpr int64_t  ADJ_EDGE_TS  = 50;

template <bool UseGpu>
core::TemporalRandomWalk make_pq_graph(double p, double q) {
    core::TemporalRandomWalk g{
        /*is_directed=*/         kIsDirected,
        /*use_gpu=*/             UseGpu,
        /*max_time_capacity=*/   -1,
        /*enable_weight_computation=*/ true,
        /*enable_temporal_node2vec=*/  true,
        /*timescale_bound=*/     -1,
        /*node2vec_p=*/          p,
        /*node2vec_q=*/          q,
    };
    test_util::add_edges(g, {
        Edge{V, P, EDGE_TS},
        Edge{V, N, EDGE_TS},
        Edge{V, F, EDGE_TS},
        Edge{P, N, ADJ_EDGE_TS},   // makes N adjacent to P
    });
    return g;
}

// ---------------------------------------------------------------------------
// Helpers to compute analytic per-edge probability under TN2V.  The three
// outbound destinations from V all share timestamp EDGE_TS, so under the
// strict-paper exp bias their static weights are equal (= exp(0)=1 after
// the per-vertex t_max shift); only the β factor matters.
//
//   target weight  ∝  β(prev=P, dest)
//                  =  1/p  for dest == P
//                     1    for dest == N (in N(P) via the ADJ edge)
//                     1/q  for dest == F
// ---------------------------------------------------------------------------
struct PQProbs { double p_dest, n_dest, f_dest; };

PQProbs analytic_pq_probs(const double p, const double q) {
    const double w_p = 1.0 / p;
    const double w_n = 1.0;
    const double w_f = 1.0 / q;
    const double tot = w_p + w_n + w_f;
    return {w_p / tot, w_n / tot, w_f / tot};
}

// ---------------------------------------------------------------------------
// Typed test fixture: host- and (when HAS_CUDA) device-backed variants.
// ---------------------------------------------------------------------------
template<typename UseGpu>
class Node2VecRejectionTest : public ::testing::Test {};

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

TYPED_TEST_SUITE(Node2VecRejectionTest, Backends);

// ===========================================================================
// Test 1 — retry-loop termination under degenerate (p, q).
//
// Picks (p, q) far from 1 so β_max / E[β] is large (acceptance per-attempt
// ≤ ~0.1).  With NODE2VEC_MAX_RETRIES=64, even degenerate workloads must
// either accept on retry (most of the time) or defensively accept the last
// proposal — but never return a sentinel for a non-empty candidate set.
// ===========================================================================
TYPED_TEST(Node2VecRejectionTest, TerminatesWithValidEdgeUnderDegeneratePQ) {
    constexpr bool use_gpu = TypeParam::value;
    // p=0.1 → 1/p = 10 (huge return weight), q=10 → 1/q = 0.1 (tiny far
    // weight).  β_max = 10 (the return weight); E[β] ≈ (10 + 1 + 0.1) / 3
    // ≈ 3.7 → per-attempt accept ratio ≈ 0.37, expected ~3 retries.  This
    // is far enough from 1 to exercise the retry path repeatedly without
    // hitting the cap.
    auto graph = make_pq_graph<use_gpu>(/*p=*/0.1, /*q=*/10.0);

    constexpr int N_HOPS = 2'000;
    int sentinels = 0;
    int valid     = 0;
    for (int i = 0; i < N_HOPS; ++i) {
        const Edge picked = test_util::get_node_edge_at(
            graph.data(), V, RandomPickerType::TemporalNode2Vec,
            /*timestamp=*/-1, /*prev_node=*/P, /*forward=*/true);
        if (is_sentinel(picked)) {
            ++sentinels;
        } else {
            ++valid;
            ASSERT_EQ(picked.u, V);
            ASSERT_TRUE(picked.i == P || picked.i == N || picked.i == F)
                << "Unexpected destination " << picked.i;
        }
    }
    EXPECT_EQ(sentinels, 0)
        << "Rejection loop must return a valid edge for a non-empty candidate set, "
           "even under degenerate (p, q); got " << sentinels << " sentinels in "
        << N_HOPS << " hops.";
    EXPECT_EQ(valid, N_HOPS);
}

// ===========================================================================
// Test 2 — distribution-following across a (p, q) sweep.
//
// The base PQBias test only checks one (p, q) cell.  This sweep covers the
// full node2vec literature parameter set, catching β-handling regressions
// at parameter boundaries.  For each (p, q) we compute the analytic per-
// edge probability and check the empirical pick frequencies against it.
//
// Per-cell budget: 10 000 samples × 3 buckets ≈ ~3-sigma stdev of 0.005;
// allow ±0.025 (= 5σ for the rarest bucket under the most peaked case
// p=0.25, q=4) so the test is robust.
// ===========================================================================
TYPED_TEST(Node2VecRejectionTest, DistributionMatchesAnalyticAcrossPQValues) {
    constexpr bool use_gpu = TypeParam::value;
    constexpr int  N_SAMPLES = 10'000;
    constexpr double TOLERANCE = 0.025;

    struct PQ { double p, q; };
    const std::vector<PQ> sweep = {
        {1.0,  1.0},   // uniform — every β is 1
        {1.0,  2.0},   // DeepWalk default-ish
        {0.5,  2.0},   // BFS-leaning
        {2.0,  0.5},   // DFS-leaning
        {0.25, 4.0},   // node2vec-extreme corner
    };

    for (const auto& pq : sweep) {
        SCOPED_TRACE(::testing::Message() << "(p=" << pq.p << ", q=" << pq.q << ")");
        auto graph = make_pq_graph<use_gpu>(pq.p, pq.q);

        int c_p = 0, c_n = 0, c_f = 0;
        for (int i = 0; i < N_SAMPLES; ++i) {
            const Edge picked = test_util::get_node_edge_at(
                graph.data(), V, RandomPickerType::TemporalNode2Vec,
                /*timestamp=*/-1, /*prev_node=*/P, /*forward=*/true);
            ASSERT_FALSE(is_sentinel(picked));
            ASSERT_EQ(picked.u, V);
            if      (picked.i == P) ++c_p;
            else if (picked.i == N) ++c_n;
            else if (picked.i == F) ++c_f;
            else FAIL() << "Unexpected destination " << picked.i;
        }
        const double frac_p = static_cast<double>(c_p) / N_SAMPLES;
        const double frac_n = static_cast<double>(c_n) / N_SAMPLES;
        const double frac_f = static_cast<double>(c_f) / N_SAMPLES;
        const PQProbs expected = analytic_pq_probs(pq.p, pq.q);

        EXPECT_NEAR(frac_p, expected.p_dest, TOLERANCE)
            << "P fraction off at (p=" << pq.p << ", q=" << pq.q << ")";
        EXPECT_NEAR(frac_n, expected.n_dest, TOLERANCE)
            << "N fraction off at (p=" << pq.p << ", q=" << pq.q << ")";
        EXPECT_NEAR(frac_f, expected.f_dest, TOLERANCE)
            << "F fraction off at (p=" << pq.p << ", q=" << pq.q << ")";
    }
}

}  // namespace
