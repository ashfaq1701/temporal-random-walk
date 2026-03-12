#include <gtest/gtest.h>

#include <set>
#include <vector>
#include <algorithm>

#include "../src/proxies/TemporalGraph.cuh"

bool is_sentinel(const Edge &e) {
    return e.u == -1 && e.i == -1 && e.ts == -1;
}

template<typename UseGpu>
class SpatioTemporalSamplerTest : public ::testing::Test {
protected:
    static constexpr bool use_gpu = UseGpu::value;

    TemporalGraph graph{
        /*is_directed=*/true,
        /*use_gpu=*/use_gpu,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/false,
        /*timescale_bound=*/-1,
        /*node2vec_p=*/1.0,
        /*node2vec_q=*/1.0,
        /*spatiotemporal_alpha=*/1.0,
        /*spatiotemporal_beta=*/1.0,
        /*spatiotemporal_gamma=*/1.0
    };
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

TYPED_TEST_SUITE(SpatioTemporalSamplerTest, Backends);

//
// ------------------------------------------------------------
// Basic Valid Sampling
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, ReturnsValidEdge) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 20},
        Edge{3, 0, 30}
    });

    const Edge e = this->graph.get_node_edge_at(
        0,
        RandomPickerType::SpatioTemporal,
        -1,
        -1,
        false
    );

    EXPECT_EQ(e.i, 0);
    EXPECT_NE(e.u, -1);
    EXPECT_NE(e.ts, -1);
}

//
// ------------------------------------------------------------
// Temporal Constraint
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, RespectsTimestampConstraint) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 20},
        Edge{3, 0, 30}
    });

    const Edge e = this->graph.get_node_edge_at(
        0,
        RandomPickerType::SpatioTemporal,
        25,
        -1,
        false
    );

    EXPECT_LT(e.ts, 25);
}

//
// ------------------------------------------------------------
// Sentinel When No Valid Edges
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, ReturnsSentinelWhenNoEdges) {
    this->graph.add_multiple_edges({
        Edge{0, 1, 10}
    });

    const Edge e = this->graph.get_node_edge_at(
        0,
        RandomPickerType::SpatioTemporal,
        -1,
        -1,
        false
    );

    EXPECT_TRUE(is_sentinel(e));
}

//
// ------------------------------------------------------------
// Randomness Within Timestamp Group
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, RandomSelectionWithinGroup) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 10},
        Edge{3, 0, 10}
    });

    std::set<int> seen;

    constexpr int NUM_TRIES = 200;

    for (int i = 0; i < NUM_TRIES; i++) {
        const Edge e = this->graph.get_node_edge_at(
            0,
            RandomPickerType::SpatioTemporal,
            -1,
            -1,
            false
        );

        EXPECT_EQ(e.i, 0);
        EXPECT_EQ(e.ts, 10);

        seen.insert(e.u);
    }

    EXPECT_GT(seen.size(), 1);
}

//
// ------------------------------------------------------------
// Exploration Bias
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, ExplorationBiasPenalizesVisitedNodes) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 10}
    });

    std::vector<int> walk = {0, 1, 0, 1};

    int visited_count = 0;
    int unvisited_count = 0;

    constexpr int NUM_TRIES = 500;

    for (int i = 0; i < NUM_TRIES; i++) {
        const Edge e = this->graph.get_node_edge_at(
            0,
            RandomPickerType::SpatioTemporal,
            -1,
            -1,
            false,
            walk,
            walk.size()
        );

        if (e.u == 1) visited_count++;
        if (e.u == 2) unvisited_count++;
    }

    EXPECT_GT(unvisited_count, visited_count);
}

//
// ------------------------------------------------------------
// Spatial Bias
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, SpatialBiasPrefersLowDegreeNodes) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 10},

        Edge{3, 1, 5},
        Edge{4, 1, 6},
        Edge{5, 1, 7}
    });

    int high_degree_node = 0;
    int low_degree_node = 0;

    constexpr int NUM_TRIES = 500;

    for (int i = 0; i < NUM_TRIES; i++) {
        const Edge e = this->graph.get_node_edge_at(
            0,
            RandomPickerType::SpatioTemporal,
            -1,
            -1,
            false
        );

        if (e.u == 1) high_degree_node++;
        if (e.u == 2) low_degree_node++;
    }

    EXPECT_GT(low_degree_node, high_degree_node);
}

//
// ------------------------------------------------------------
// Temporal Bias
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, TemporalBiasPrefersRecentEdges) {
    this->graph.add_multiple_edges({
        Edge{1, 0, 10},
        Edge{2, 0, 20}
    });

    int older = 0;
    int newer = 0;

    constexpr int NUM_TRIES = 500;

    for (int i = 0; i < NUM_TRIES; i++) {
        const Edge e = this->graph.get_node_edge_at(
            0,
            RandomPickerType::SpatioTemporal,
            -1,
            -1,
            false
        );

        if (e.u == 1) older++;
        if (e.u == 2) newer++;
    }

    EXPECT_GT(newer, older);
}

//
// ------------------------------------------------------------
// Forward Direction Sanity
// ------------------------------------------------------------
//

TYPED_TEST(SpatioTemporalSamplerTest, ForwardDirectionSanity) {
    this->graph.add_multiple_edges({
        Edge{0, 1, 10},
        Edge{0, 2, 20},
        Edge{0, 3, 30}
    });

    const Edge e = this->graph.get_node_edge_at(
        0,
        RandomPickerType::SpatioTemporal,
        -1,
        -1,
        true
    );

    EXPECT_EQ(e.u, 0);
    EXPECT_NE(e.i, -1);
}
