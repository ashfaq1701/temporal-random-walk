#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"
#include "../src/stores/edge_data.cuh"

namespace {

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

inline bool is_sentinel(const Edge& e) {
    return e.u == -1 && e.i == -1 && e.ts == -1;
}

inline std::vector<Edge> collect_outbound_edges(
    const TemporalGraphStore* s, int u)
{
    std::vector<Edge> out;
    const auto edges = edge_data::get_edges(s->edge_data);

    for (size_t i = 0; i < edges.size; ++i) {
        const Edge& e = edges.data[i];
        if (e.u == u) out.push_back(e);
    }
    return out;
}

inline bool contains_edge(const std::vector<Edge>& v, const Edge& e) {
    return std::any_of(v.begin(), v.end(), [&](const Edge& x) {
        return x.u == e.u && x.i == e.i && x.ts == e.ts;
    });
}

// ------------------------------------------------------------
// CPU Tests
// ------------------------------------------------------------

class TemporalNode2VecCpuTest : public ::testing::Test {
protected:
    TemporalGraph graph{
        /*is_directed=*/true,
        /*use_gpu=*/false,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/2.0,
        /*node2vec_q=*/0.5
    };

    void SetUp() override {
        graph.add_multiple_edges({
            // Node 0: sparse, multi-timestamp
            Edge{0, 5, 10},
            Edge{0, 42, 10},
            Edge{0, 1000, 15},
            Edge{0, 7, 20},

            // Structure for TN2V semantics
            Edge{5, 42, 5},     // neighbor-of-prev
            Edge{1000, 1, 6},   // out-edge
            Edge{7, 3, 8}       // out-edge
        });
    }

    const TemporalGraphStore* store() const {
        return graph.get_graph();
    }
};

/* ------------------------------------------------------------
 * β-rule correctness
 * ------------------------------------------------------------ */

TEST_F(TemporalNode2VecCpuTest, BetaRulesFollowNode2VecDefinition) {
    const auto* s = store();

    const int prev = 5;

    // return
    EXPECT_DOUBLE_EQ(
        temporal_graph::compute_node2vec_beta_host(s, prev, prev),
        1.0 / 2.0   // p = 2.0
    );

    // neighbor (5 -> 42 exists)
    EXPECT_DOUBLE_EQ(
        temporal_graph::compute_node2vec_beta_host(s, prev, 42),
        1.0
    );

    // out-node
    EXPECT_DOUBLE_EQ(
        temporal_graph::compute_node2vec_beta_host(s, prev, 7),
        1.0 / 0.5   // q = 0.5
    );
}

/* ------------------------------------------------------------
 * First-order fallback (prev_node = -1)
 * ------------------------------------------------------------ */

TEST_F(TemporalNode2VecCpuTest, Tn2vWithoutPrevNodeReturnsValidOutboundEdge) {
    const auto* s = store();
    const auto outbound = collect_outbound_edges(s, 0);
    ASSERT_FALSE(outbound.empty());

    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            -1,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

/* ------------------------------------------------------------
 * Second-order TN2V (valid prev_node)
 * ------------------------------------------------------------ */

TEST_F(TemporalNode2VecCpuTest, Tn2vWithValidPrevNodeReturnsValidOutboundEdge) {
    const auto* s = store();
    const auto outbound = collect_outbound_edges(s, 0);
    ASSERT_FALSE(outbound.empty());

    // Choose a real predecessor dynamically
    const int prev_node = outbound.front().i;

    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            prev_node,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

/* ------------------------------------------------------------
 * Unrelated prev_node ⇒ safe fallback
 * ------------------------------------------------------------ */

TEST_F(TemporalNode2VecCpuTest, Tn2vWithUnrelatedPrevNodeIsSafe) {
    const auto* s = store();
    const auto outbound = collect_outbound_edges(s, 0);
    ASSERT_FALSE(outbound.empty());

    constexpr int unrelated_prev = 9999;

    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            unrelated_prev,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

/* ------------------------------------------------------------
 * Backward walk with no inbound edges
 * ------------------------------------------------------------ */

TEST_F(TemporalNode2VecCpuTest, BackwardTn2vFromNodeWithNoInboundReturnsSentinel) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            5,
            false);

    EXPECT_TRUE(is_sentinel(picked));
}

#ifdef HAS_CUDA

// ------------------------------------------------------------
// GPU Tests (semantic parity)
// ------------------------------------------------------------

class TemporalNode2VecGpuTest : public ::testing::Test {
protected:
    TemporalGraph graph{
        /*is_directed=*/true,
        /*use_gpu=*/true,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/2.0,
        /*node2vec_q=*/0.5
    };

    void SetUp() override {
        graph.add_multiple_edges({
            Edge{0, 5, 10},
            Edge{0, 42, 10},
            Edge{0, 1000, 15},
            Edge{0, 7, 20},
            Edge{5, 42, 5},
            Edge{1000, 1, 6},
            Edge{7, 3, 8}
        });
    }

    const TemporalGraphStore* store() const {
        return graph.get_graph();
    }
};

TEST_F(TemporalNode2VecGpuTest, GpuTn2vReturnsValidOutboundEdge) {
    const auto* s = store();
    const auto outbound = collect_outbound_edges(s, 0);
    ASSERT_FALSE(outbound.empty());

    const int prev_node = outbound.front().i;

    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            prev_node,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_EQ(picked.u, 0);
    EXPECT_TRUE(contains_edge(outbound, picked));
}

TEST_F(TemporalNode2VecGpuTest, DeterministicPickerWorksOnGpu) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TEST_LAST,
            -1,
            -1,
            true);

    EXPECT_EQ(picked.u, 0);
    EXPECT_EQ(picked.i, 7);
    EXPECT_EQ(picked.ts, 20);
}

#endif  // HAS_CUDA

} // namespace
