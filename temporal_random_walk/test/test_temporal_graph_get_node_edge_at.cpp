#include <gtest/gtest.h>
#include "../src/proxies/TemporalGraphProxy.cuh"


template<typename T>
class TemporalGraphGetNodeEdgeAtTest : public ::testing::Test {
protected:
    std::unique_ptr<TemporalGraphProxy> graph;

    void SetUp() override {
        graph = std::make_unique<TemporalGraphProxy>(true, T::value); // directed graph
    }

    // Helper to verify edge fields
    static void verify_edge(const Edge& edge, const int expected_src, const int expected_tgt, const int64_t expected_ts) {
        EXPECT_EQ(edge.u, expected_src);
        EXPECT_EQ(edge.i, expected_tgt);
        EXPECT_EQ(edge.ts, expected_ts);
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>   // CPU mode only
>;
#endif

TYPED_TEST_SUITE(TemporalGraphGetNodeEdgeAtTest, GPU_USAGE_TYPES);

// Test forward walks from a node
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ForwardWalkTest) {
    auto edges = std::vector<Edge>{
        Edge{10, 20, 100}, // Node 10 outbound group 1
        Edge{10, 30, 100},
        Edge{10, 40, 102}, // Node 10 outbound group 2
        Edge{10, 50, 104}, // Node 10 outbound group 3
        Edge{20, 10, 101}, // Node 10 inbound (should be ignored for forward)
        Edge{30, 10, 103}  // Node 10 inbound (should be ignored for forward)
    };
    this->graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, -1, true);
    EXPECT_EQ(edge.ts, 100); // Should select from first group
    EXPECT_EQ(edge.u, 10);  // Source should be 10

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_LAST, -1, true);
    EXPECT_EQ(edge.ts, 104); // Should select from last group
    EXPECT_EQ(edge.u, 10);  // Source should be 10

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 102, true);
    EXPECT_EQ(edge.ts, 104); // Should select first group after 102

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 103, true);
    EXPECT_EQ(edge.ts, 104); // Should select first group after 103

    // Test no available groups after timestamp
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 104, true);
    this->verify_edge(edge, -1, -1, -1); // No groups after 104
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, BackwardWalkTest) {
    auto edges = std::vector<Edge>{
        Edge{20, 10, 100}, // Node 10 inbound: ts 100 -> 103
        Edge{30, 10, 101}, // From upstream (source) nodes: 20,30,40,50
        Edge{40, 10, 102}, // To downstream node: 10
        Edge{50, 10, 103},
        Edge{10, 20, 101}, // Node 10 outbound (should be ignored for backward)
        Edge{10, 30, 102}  // Node 10 outbound (should be ignored for backward)
    };
    this->graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, -1, false);
    this->verify_edge(edge, 20, 10, 100); // Should select from first group with first_picker

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_LAST, -1, false);
    this->verify_edge(edge, 50, 10, 103); // Should select from last group with last_picker

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 104, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 104

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 103, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 103

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 102, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 102

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 101

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_LAST, 104, false);
    this->verify_edge(edge, 50, 10, 103); // Should select 103 as highest timestamp < 104

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_LAST, 103, false);
    this->verify_edge(edge, 40, 10, 102); // Should select 102 as highest timestamp < 103

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_LAST, 102, false);
    this->verify_edge(edge, 30, 10, 101); // Should select 101 as highest timestamp < 102

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as highest timestamp < 101

    // Test edge cases
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 100, false);
    this->verify_edge(edge, -1, -1, -1); // No timestamps < 100

    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 99, false);
    this->verify_edge(edge, -1, -1, -1); // No timestamps < 99
}

// Test edge cases and invalid inputs
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, EdgeCasesTest) {
    auto edges = std::vector<Edge>{
        Edge{10, 20, 100}, // Node 10 outbound: ts 100,101
        Edge{10, 30, 101}, // Node 10 -> Nodes 20,30
    };
    this->graph->add_multiple_edges(edges);

    // Test invalid node ID
    auto edge = this->graph->get_node_edge_at(-1, RandomPickerType::TEST_FIRST, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    // Test non-existent node
    edge = this->graph->get_node_edge_at(999, RandomPickerType::TEST_FIRST, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    // Test node with no edges in requested direction
    edge = this->graph->get_node_edge_at(20, RandomPickerType::TEST_FIRST, -1, true); // Node 20 has no outbound edges
    this->verify_edge(edge, -1, -1, -1);

    // Test backward walk for node with no inbound edges
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, -1, false); // Node 10 has no inbound edges
    this->verify_edge(edge, -1, -1, -1);

    // Test empty graph
    this->graph = std::make_unique<TemporalGraphProxy>(true, TypeParam::value);
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, -1, true);
    this->verify_edge(edge, -1, -1, -1);
}

// Test random selection within timestamp groups
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, RandomSelectionTest) {
    const auto edges = std::vector<Edge>{
        Edge{10, 20, 100}, // Group 1: ts 100
        Edge{10, 30, 100},
        Edge{10, 40, 100},
        Edge{10, 50, 101}  // Group 2: ts 101
    };
    this->graph->add_multiple_edges(edges);

    // Make multiple selections from first timestamp group
    std::set<int> seen_targets;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, -1, true);
        EXPECT_EQ(edge.ts, 100); // Should always be from first group
        seen_targets.insert(edge.i);
    }

    // Should see more than one target due to random selection within group ts=100
    EXPECT_GT(seen_targets.size(), 1);
}

// Test exact timestamp matching
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampTest) {
    auto edges = std::vector<Edge>{
        // Edges for backward walks (upstream -> node 10)
        Edge{20, 10, 100}, // Upstream nodes 20,30,40 -> downstream node 10
        Edge{30, 10, 101},
        Edge{40, 10, 102},
        // Edges for forward walks (node 10 -> downstream)
        Edge{10, 50, 100}, // Upstream node 10 -> downstream nodes 50,60,70
        Edge{10, 60, 101},
        Edge{10, 70, 102}
    };
    this->graph->add_multiple_edges(edges);

    // Forward direction (node 10 to downstream)
    auto edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 100, true);
    this->verify_edge(edge, 10, 60, 101); // Should get next timestamp going downstream

    // Backward direction (node 10 looking upstream)
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should get previous timestamp from upstream
}

// Test exact timestamp matching for undirected graphs
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampUndirectedTest) {
    // Create undirected graph
    this->graph = std::make_unique<TemporalGraphProxy>(false, TypeParam::value);

    auto edges = std::vector<Edge>{
        // Edges connecting to node 10
        Edge{10, 20, 100}, // Will be normalized (stored as min source, max target)
        Edge{30, 10, 101}, // These edges connect node 10 with 20,30,40,50,60,70
        Edge{10, 40, 102}, // In both directions since it's undirected
        Edge{50, 10, 100},
        Edge{10, 60, 101},
        Edge{70, 10, 102},
        Edge{20, 30, 104}
    };
    this->graph->add_multiple_edges(edges);

    // Forward direction should work same as backward
    auto edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(edge.ts, 101);
    EXPECT_TRUE(edge.u == 10 && (edge.i == 30 || edge.i == 60));

    // Backward direction
    edge = this->graph->get_node_edge_at(10, RandomPickerType::TEST_FIRST, 101, false);
    EXPECT_EQ(edge.ts, 100);
    EXPECT_TRUE(edge.u == 10 && (edge.i == 20 || edge.i == 50));

    // Try from other node's perspective
    edge = this->graph->get_node_edge_at(20, RandomPickerType::TEST_FIRST, 100, true);
    this->verify_edge(edge, 20, 30, 104);
}
