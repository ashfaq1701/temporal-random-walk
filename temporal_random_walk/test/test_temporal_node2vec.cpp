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

    [[nodiscard]] std::pair<size_t, size_t> group_edge_range(const int node_id, const size_t group_pos) const {
        const auto idx = index();
        const auto group_offsets = idx.node_ts_group_outbound_offsets();
        const auto [group_start, group_end] = outbound_group_range(node_id);

        const size_t edge_start = group_offsets[group_pos];
        const size_t edge_end = (group_pos + 1 < group_end)
            ? group_offsets[group_pos + 1]
            : idx.node_group_outbound_offsets()[node_id + 1];

        return {edge_start, edge_end};
    }
};

TEST_F(TemporalNode2VecCpuTest, BetaRulesAreCorrect) {
    const auto* graph_store = store();

    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 1), 0.5);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 3), 2.0);
}

TEST_F(TemporalNode2VecCpuTest, Node2VecGroupPickerUsesBetaWeightedExponentialMass) {
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

TEST_F(TemporalNode2VecCpuTest, Node2VecEdgePickerFavorsReturnAndNeighborBeforeDistant) {
    const auto* graph_store = store();
    const auto idx = index();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();

    const auto [edge_start, edge_end] = group_edge_range(0, outbound_group_range(0).first);
    ASSERT_EQ(edge_end - edge_start, 2);

    const long low_rand_pick = temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        graph_store,
        0,
        1,
        edge_start,
        edge_end,
        sorted_indices.data(),
        0.10);

    const long high_rand_pick = temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        graph_store,
        0,
        1,
        edge_start,
        edge_end,
        sorted_indices.data(),
        0.90);

    EXPECT_EQ(low_rand_pick, static_cast<long>(sorted_indices[edge_start]));
    EXPECT_EQ(high_rand_pick, static_cast<long>(sorted_indices[edge_start + 1]));
}

TEST_F(TemporalNode2VecCpuTest, InvalidNode2VecInputsReturnSentinel) {
    const auto* graph_store = store();
    const auto idx = index();
    const auto [group_start, group_end] = outbound_group_range(0);

    auto group_offsets = idx.node_ts_group_outbound_offsets();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();
    std::vector<double> cumulative{0.2, 1.0};

    EXPECT_EQ((temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        graph_store,
        0,
        -1,
        group_start,
        group_end,
        group_end,
        group_offsets.data(),
        sorted_indices.data(),
        cumulative.data(),
        0.5)), -1);

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
    TemporalGraph cpu_graph{true, false, -1, true, -1, 2.0, 0.5};
    TemporalGraph gpu_graph{true, true, -1, true, -1, 2.0, 0.5};

    void SetUp() override {
        const std::vector<Edge> edges{
            Edge{0, 1, 10},
            Edge{0, 2, 10},
            Edge{0, 3, 20},
            Edge{1, 2, 5},
            Edge{4, 1, 6}
        };

        cpu_graph.add_multiple_edges(edges);
        gpu_graph.add_multiple_edges(edges);
    }
};

TEST_F(TemporalNode2VecGpuTest, ProxyMatchesCpuForTemporalNode2VecWithoutTimestampConstraint) {
    const double random_nums[2] = {0.95, 0.95};

    const auto cpu_edge = cpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        1);

    const auto gpu_edge = gpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        1);

    EXPECT_EQ(gpu_edge.u, cpu_edge.u);
    EXPECT_EQ(gpu_edge.i, cpu_edge.i);
    EXPECT_EQ(gpu_edge.ts, cpu_edge.ts);
}

TEST_F(TemporalNode2VecGpuTest, ProxyMatchesCpuForTemporalNode2VecWithTimestampConstraint) {
    const double random_nums[2] = {0.25, 0.75};

    const auto cpu_edge = cpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        15,
        true,
        1);

    const auto gpu_edge = gpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        15,
        true,
        1);

    EXPECT_EQ(cpu_edge.u, 0);
    EXPECT_EQ(cpu_edge.i, 3);
    EXPECT_EQ(cpu_edge.ts, 20);

    EXPECT_EQ(gpu_edge.u, cpu_edge.u);
    EXPECT_EQ(gpu_edge.i, cpu_edge.i);
    EXPECT_EQ(gpu_edge.ts, cpu_edge.ts);
}

TEST_F(TemporalNode2VecGpuTest, ProxyHandlesMissingPrevNodeWithoutCrashing) {
    const double random_nums[2] = {0.6, 0.4};

    const auto cpu_edge = cpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        -1);

    const auto gpu_edge = gpu_graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        -1);

    EXPECT_EQ(gpu_edge.u, cpu_edge.u);
    EXPECT_EQ(gpu_edge.i, cpu_edge.i);
    EXPECT_EQ(gpu_edge.ts, cpu_edge.ts);
}

#endif

} // namespace
