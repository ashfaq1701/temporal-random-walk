#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"
#include "../src/proxies/EdgeData.cuh"

namespace {
    // ------------------------------------------------------------
    // Helpers (backend-agnostic)
    // ------------------------------------------------------------

    bool is_sentinel(const Edge &e) {
        return e.u == -1 && e.i == -1 && e.ts == -1;
    }

    bool contains_edge(const std::vector<Edge> &v, const Edge &e) {
        return std::any_of(v.begin(), v.end(), [&](const Edge &x) {
            return x.u == e.u && x.i == e.i && x.ts == e.ts;
        });
    }

    // ------------------------------------------------------------
    // Typed fixture: CPU + GPU
    // ------------------------------------------------------------

    template<typename UseGpu>
    class TemporalNode2VecTest : public ::testing::Test {
    protected:
        static constexpr bool use_gpu = UseGpu::value;

        TemporalGraph graph{
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

        [[nodiscard]] const TemporalGraphStore *store() const {
            return graph.get_graph();
        }

        // Always collect edges through EdgeData proxy (CPU/GPU safe)
        [[nodiscard]] std::vector<Edge> collect_outbound_edges(int u) const {
            EdgeData edges(store()->edge_data);
            const auto all = edges.get_edges();

            std::vector<Edge> out;
            for (const auto &e: all) {
                if (e.u == u) out.push_back(e);
            }
            return out;
        }
    };

    // ------------------------------------------------------------
    // Test instantiation
    // ------------------------------------------------------------

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

    // ------------------------------------------------------------
    // β-rule correctness
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, BetaRulesFollowNode2VecDefinition) {
        const int prev = 5;

        EXPECT_DOUBLE_EQ(
            this->graph.compute_node2vec_beta(prev, prev),
            1.0 / 2.0
        );

        EXPECT_DOUBLE_EQ(
            this->graph.compute_node2vec_beta(prev, 42),
            1.0
        );

        EXPECT_DOUBLE_EQ(
            this->graph.compute_node2vec_beta(prev, 7),
            1.0 / 0.5
        );
    }

    // ------------------------------------------------------------
    // First-order fallback
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, Tn2vWithoutPrevNodeReturnsValidOutboundEdge) {
        const auto outbound = this->collect_outbound_edges(0);
        ASSERT_FALSE(outbound.empty());

        const Edge picked =
                this->graph.get_node_edge_at(
                    0,
                    RandomPickerType::TemporalNode2Vec,
                    -1,
                    -1,
                    true);

        EXPECT_FALSE(is_sentinel(picked));
        EXPECT_EQ(picked.u, 0);
        EXPECT_TRUE(contains_edge(outbound, picked));
    }

    // ------------------------------------------------------------
    // Second-order TN2V
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, Tn2vWithValidPrevNodeReturnsValidOutboundEdge) {
        const auto outbound = this->collect_outbound_edges(0);
        ASSERT_FALSE(outbound.empty());

        const int prev_node = outbound.front().i;

        const Edge picked =
                this->graph.get_node_edge_at(
                    0,
                    RandomPickerType::TemporalNode2Vec,
                    -1,
                    prev_node,
                    true);

        EXPECT_FALSE(is_sentinel(picked));
        EXPECT_EQ(picked.u, 0);
        EXPECT_TRUE(contains_edge(outbound, picked));
    }

    // ------------------------------------------------------------
    // Unrelated prev_node fallback
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, Tn2vWithUnrelatedPrevNodeIsSafe) {
        const auto outbound = this->collect_outbound_edges(0);
        ASSERT_FALSE(outbound.empty());

        const Edge picked =
                this->graph.get_node_edge_at(
                    0,
                    RandomPickerType::TemporalNode2Vec,
                    -1,
                    9999,
                    true);

        EXPECT_FALSE(is_sentinel(picked));
        EXPECT_EQ(picked.u, 0);
        EXPECT_TRUE(contains_edge(outbound, picked));
    }

    // ------------------------------------------------------------
    // Backward walk safety
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, BackwardTn2vFromNodeWithNoInboundReturnsSentinel) {
        const Edge picked =
                this->graph.get_node_edge_at(
                    0,
                    RandomPickerType::TemporalNode2Vec,
                    -1,
                    5,
                    false);

        EXPECT_TRUE(is_sentinel(picked));
    }

    // ------------------------------------------------------------
    // CSR correctness (CPU + GPU)
    // ------------------------------------------------------------

    TYPED_TEST(TemporalNode2VecTest, NodeAdjacencyCSRIsValid) {
        EdgeData edges(this->store()->edge_data);

        const auto offsets = edges.node_adj_offsets();
        const auto neighbors = edges.node_adj_neighbors();

        ASSERT_FALSE(offsets.empty());
        ASSERT_EQ(offsets.back(), neighbors.size());

        for (size_t i = 0; i + 1 < offsets.size(); ++i) {
            ASSERT_LE(offsets[i], offsets[i + 1]);
        }

        for (int v: neighbors) {
            ASSERT_GE(v, 0);
            ASSERT_LT(static_cast<size_t>(v), offsets.size() - 1);
        }
    }
} // namespace
