#include <gtest/gtest.h>

#include <vector>
#include <utility>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/proxies/NodeEdgeIndex.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"

namespace {

// ------------------------
// Utility helpers
// ------------------------

inline bool is_sentinel(const Edge& e) {
    return e.u == -1 && e.i == -1 && e.ts == -1;
}

inline bool is_one_of(const Edge& e, const std::vector<Edge>& edges) {
    for (const auto& x : edges) {
        if (e.u == x.u && e.i == x.i && e.ts == x.ts) return true;
    }
    return false;
}

// Outbound edges from node 0 in the test graph (directed)
const std::vector<Edge> kNode0OutboundEdges = {
    Edge{0, 1, 10},
    Edge{0, 2, 10},
    Edge{0, 3, 20},
};

// Deterministic reference for edge-level beta sampling (CPU)
long expected_edge_pick_reference(
    const TemporalGraphStore* graph,
    const int node_id,
    const int prev_node,
    const size_t edge_start,
    const size_t edge_end,
    const size_t* node_ts_sorted_indices,
    const double edge_selector_rand_num) {

    double beta_sum = 0.0;
    for (size_t i = edge_start; i < edge_end; ++i) {
        const size_t edge_idx = node_ts_sorted_indices[i];
        const int w =
            temporal_graph::get_node2vec_candidate_node<true, true>(graph, node_id, edge_idx);
        beta_sum += temporal_graph::compute_node2vec_beta_host(graph, prev_node, w);
    }

    if (beta_sum <= 0.0) return -1;

    const double target = edge_selector_rand_num * beta_sum;
    double running_sum = 0.0;

    for (size_t i = edge_start; i < edge_end; ++i) {
        const size_t edge_idx = node_ts_sorted_indices[i];
        const int w =
            temporal_graph::get_node2vec_candidate_node<true, true>(graph, node_id, edge_idx);
        running_sum += temporal_graph::compute_node2vec_beta_host(graph, prev_node, w);
        if (running_sum >= target) {
            return static_cast<long>(edge_idx);
        }
    }

    return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
}

// ------------------------
// CPU tests
// ------------------------

class TemporalNode2VecCpuTest : public ::testing::Test {
protected:
    // directed=true, gpu=false, enable_weight_computation=true, p=2.0, q=0.5
    TemporalGraph graph{
        true,
        false,
        -1,
        false,
        true,
        -1,
        2.0,
        0.5
    };

    void SetUp() override {
        graph.add_multiple_edges({
            Edge{0, 1, 10},
            Edge{0, 2, 10},
            Edge{0, 3, 20},
            Edge{1, 2, 5},
            Edge{4, 1, 6}
        });
    }

    [[nodiscard]] TemporalGraphStore* store() const {
        return graph.get_graph();
    }

    [[nodiscard]] NodeEdgeIndex index() const {
        return NodeEdgeIndex(store()->node_edge_index);
    }

    [[nodiscard]] std::pair<size_t, size_t> outbound_group_range(const int node_id) const {
        const auto ranges = index().count_ts_group_per_node_outbound();
        return {ranges[node_id], ranges[node_id + 1]};
    }
};

/* ---------- Helper-level tests ---------- */

TEST_F(TemporalNode2VecCpuTest, BetaRulesAreCorrect) {
    const auto* s = store();

    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(s, 1, 1), 0.5); // return
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(s, 1, 2), 1.0); // neighbor
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(s, 1, 3), 2.0); // out
}

TEST_F(TemporalNode2VecCpuTest, GroupWeightFromCumulativeHandlesSubranges) {
    const std::vector<double> cumulative = {0.10, 0.35, 0.65, 1.00};

    EXPECT_NEAR(
        temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 1, 1),
        0.25, 1e-12);
    EXPECT_NEAR(
        temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 2, 1),
        0.30, 1e-12);
    EXPECT_NEAR(
        temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 3, 1),
        0.35, 1e-12);
}

TEST_F(TemporalNode2VecCpuTest, GroupPickerPrefersHigherBetaWeightedMass) {
    const auto* s = store();
    const auto idx = index();
    const auto [group_start, group_end] = outbound_group_range(0);
    ASSERT_EQ(group_end - group_start, 2);

    auto group_offsets = idx.node_ts_group_outbound_offsets();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();
    std::vector<double> cumulative = {0.2, 1.0};

    const int early =
        temporal_graph::pick_random_temporal_node2vec_host<true, true>(
            s, 0, 1, group_start, group_end, group_end,
            group_offsets.data(), sorted_indices.data(),
            cumulative.data(), 0.10);

    const int late =
        temporal_graph::pick_random_temporal_node2vec_host<true, true>(
            s, 0, 1, group_start, group_end, group_end,
            group_offsets.data(), sorted_indices.data(),
            cumulative.data(), 0.90);

    EXPECT_EQ(early, static_cast<int>(group_start));
    EXPECT_EQ(late, static_cast<int>(group_start + 1));
}

TEST_F(TemporalNode2VecCpuTest, EdgePickerMatchesReferencePrefixSampling) {
    const auto* s = store();
    const auto idx = index();

    auto group_offsets = idx.node_ts_group_outbound_offsets();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();
    auto group_ranges = idx.count_ts_group_per_node_outbound();

    const size_t group_start = group_ranges[0];
    const size_t edge_start = group_offsets[group_start];
    const size_t edge_end = group_offsets[group_start + 1];
    ASSERT_GT(edge_end, edge_start);

    for (double r : {0.10, 0.55, 0.90}) {
        const long picked =
            temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
                s, 0, 1, edge_start, edge_end, sorted_indices.data(), r);

        const long expected =
            expected_edge_pick_reference(
                s, 0, 1, edge_start, edge_end, sorted_indices.data(), r);

        EXPECT_EQ(picked, expected);
    }
}

/* ---------- Selector-level semantics ---------- */

TEST_F(TemporalNode2VecCpuTest, TemporalNode2VecWithoutPrevNodeFallsBack) {
    // prev_node == -1 ⇒ first-order temporal behavior
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            -1,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_TRUE(is_one_of(picked, kNode0OutboundEdges));
}

TEST_F(TemporalNode2VecCpuTest, TemporalNode2VecWithPrevNodeReturnsValidOutboundEdge) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            1,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_TRUE(is_one_of(picked, kNode0OutboundEdges));
}

TEST_F(TemporalNode2VecCpuTest, TemporalNode2VecBackwardFromNodeWithNoInboundReturnsSentinel) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            1,
            false); // backward

    EXPECT_TRUE(is_sentinel(picked));
}

// ------------------------
// GPU tests
// ------------------------

#ifdef HAS_CUDA

class TemporalNode2VecGpuTest : public ::testing::Test {
protected:
    TemporalGraph graph{true, true, -1, true, -1, 2.0, 0.5};

    void SetUp() override {
        graph.add_multiple_edges({
            Edge{0, 1, 10},
            Edge{0, 2, 10},
            Edge{0, 3, 20},
            Edge{1, 2, 5},
            Edge{4, 1, 6}
        });
    }
};

TEST_F(TemporalNode2VecGpuTest, TemporalNode2VecWithoutPrevNodeFallsBack) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            -1,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_TRUE(is_one_of(picked, kNode0OutboundEdges));
}

TEST_F(TemporalNode2VecGpuTest, TemporalNode2VecWithPrevNodeReturnsValidOutboundEdge) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            1,
            true);

    EXPECT_FALSE(is_sentinel(picked));
    EXPECT_TRUE(is_one_of(picked, kNode0OutboundEdges));
}

TEST_F(TemporalNode2VecGpuTest, TemporalNode2VecBackwardFromNodeWithNoInboundReturnsSentinel) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TemporalNode2Vec,
            -1,
            1,
            false);

    EXPECT_TRUE(is_sentinel(picked));
}

TEST_F(TemporalNode2VecGpuTest, DeterministicPickerRunsOnGpuGraph) {
    const Edge picked =
        graph.get_node_edge_at(
            0,
            RandomPickerType::TEST_LAST,
            -1,
            -1,
            true);

    EXPECT_EQ(picked.u, 0);
    EXPECT_EQ(picked.i, 3);
    EXPECT_EQ(picked.ts, 20);
}

#endif

} // namespace
