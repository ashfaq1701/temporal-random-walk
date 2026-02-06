#include <gtest/gtest.h>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/proxies/NodeEdgeIndex.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"

namespace {

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>>;
#endif

template<typename T>
class TemporalNode2VecTest : public ::testing::Test {
protected:
    TemporalGraph graph{true, T::value, -1, true, -1, 2.0, 0.5};

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

TYPED_TEST_SUITE(TemporalNode2VecTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalNode2VecTest, BetaRulesAreCorrect) {
    const auto* graph_store = this->store();

    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 1), 0.5);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(graph_store, 1, 3), 2.0);
}

TYPED_TEST(TemporalNode2VecTest, Node2VecGroupPickerUsesBetaWeightedExponentialMass) {
    const auto* graph_store = this->store();
    const auto idx = this->index();
    const auto [group_start, group_end] = this->outbound_group_range(0);
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

TYPED_TEST(TemporalNode2VecTest, Node2VecEdgePickerFavorsReturnAndNeighborBeforeDistant) {
    const auto* graph_store = this->store();
    const auto idx = this->index();
    auto sorted_indices = idx.node_ts_sorted_outbound_indices();

    const auto [edge_start, edge_end] = this->group_edge_range(0, this->outbound_group_range(0).first);
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

TYPED_TEST(TemporalNode2VecTest, TemporalGraphProxyUsesPrevNodeWhenProvided) {
    const double random_nums[2] = {0.95, 0.95};

    const auto edge_without_prev = this->graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        -1);

    const auto edge_with_prev = this->graph.get_node_edge_at_with_provided_nums(
        0,
        RandomPickerType::TemporalNode2Vec,
        random_nums,
        -1,
        true,
        1);

    EXPECT_EQ(edge_without_prev.u, 0);
    EXPECT_EQ(edge_without_prev.i, 2);
    EXPECT_EQ(edge_without_prev.ts, 10);

    EXPECT_EQ(edge_with_prev.u, 0);
    EXPECT_EQ(edge_with_prev.i, 2);
    EXPECT_EQ(edge_with_prev.ts, 10);
}

TYPED_TEST(TemporalNode2VecTest, InvalidNode2VecInputsReturnSentinel) {
    const auto* graph_store = this->store();
    const auto idx = this->index();
    const auto [group_start, group_end] = this->outbound_group_range(0);

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

} // namespace
