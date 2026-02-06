#include <gtest/gtest.h>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/proxies/NodeEdgeIndex.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"

namespace {

class TemporalNode2VecCpuTest : public ::testing::Test {
protected:
    TemporalGraph graph{true, false, -1, true, -1, 2.0, 0.5};

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

TEST_F(TemporalNode2VecCpuTest, BetaRulesAreCorrect) {
    const auto* graph_store = store();

    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 1), 0.5);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 3), 2.0);
}

TEST_F(TemporalNode2VecCpuTest, GroupPickerPrefersHigherBetaWeightedMass) {
    const auto* graph_store = store();
    const auto idx = index();
    const auto [group_start, group_end] = outbound_group_range(0);
    ASSERT_EQ(group_end - group_start, 2);

    auto group_offsets = idx.node_ts_group_outbound_offsets();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();
    std::vector<double> cumulative{0.2, 1.0};

    const int first_group = temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        graph_store,
        0,
        1,
        group_start,
        group_end,
        group_end,
        group_offsets.data(),
        sorted_indices.data(),
        cumulative.data(),
        0.10);

    const int second_group = temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        graph_store,
        0,
        1,
        group_start,
        group_end,
        group_end,
        group_offsets.data(),
        sorted_indices.data(),
        cumulative.data(),
        0.90);

    EXPECT_EQ(first_group, static_cast<int>(group_start));
    EXPECT_EQ(second_group, static_cast<int>(group_start + 1));
}

TEST_F(TemporalNode2VecCpuTest, EdgePickerReturnsSentinelForInvalidInput) {
    const auto* graph_store = store();
    const auto idx = index();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();

    EXPECT_EQ((temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        graph_store,
        0,
        -1,
        0,
        1,
        sorted_indices.data(),
        0.5)), -1);

    EXPECT_EQ((temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        graph_store,
        0,
        1,
        1,
        1,
        sorted_indices.data(),
        0.5)), -1);
}

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

TEST_F(TemporalNode2VecGpuTest, TemporalNode2VecWithoutPrevNodeReturnsSentinelEdge) {
    const Edge picked = graph.get_node_edge_at(0, RandomPickerType::TemporalNode2Vec, -1, true);

    EXPECT_EQ(picked.u, -1);
    EXPECT_EQ(picked.i, -1);
    EXPECT_EQ(picked.ts, -1);
}

TEST_F(TemporalNode2VecGpuTest, DeterministicPickerRunsOnGpuGraph) {
    const Edge picked = graph.get_node_edge_at(0, RandomPickerType::TEST_LAST, -1, true);

    EXPECT_EQ(picked.u, 0);
    EXPECT_EQ(picked.i, 3);
    EXPECT_EQ(picked.ts, 20);
}

#endif

} // namespace
