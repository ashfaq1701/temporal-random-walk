#include <gtest/gtest.h>
#include "../src/proxies/TemporalGraph.cuh"

template<typename T>
class TemporalGraphTest : public ::testing::Test {
protected:
    std::unique_ptr<TemporalGraph> graph;

    void SetUp() override {
        // Create directed graph by default
        graph = std::make_unique<TemporalGraph>(true, T::value);
    }

    // Helper to create edge tuples
    static std::vector<std::tuple<int, int, int64_t>> create_edges(
        std::initializer_list<std::tuple<int, int, int64_t>> edges) {
        return edges;
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>   // CPU mode only
>;
#endif

TYPED_TEST_SUITE(TemporalGraphTest, GPU_USAGE_TYPES);

// Test empty state
TYPED_TEST(TemporalGraphTest, EmptyStateTest) {
    EXPECT_EQ(this->graph->get_node_ids().size(), 0);
    EXPECT_TRUE(this->graph->get_edges().empty());
}

// Test basic edge addition
TYPED_TEST(TemporalGraphTest, BasicEdgeAdditionTest) {
    std::vector<Edge> edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 200},
        Edge {3, 1, 300}
    };

    this->graph->add_multiple_edges(edges, 3);

    auto graph_edges = this->graph->get_edges();
    EXPECT_EQ(graph_edges.size(), 3);
    EXPECT_EQ(this->graph->get_node_ids().size(), 3);
}

TYPED_TEST(TemporalGraphTest, MaintainSortedOrderTest) {
    // First addition
    std::vector<Edge> edges1 = {
        Edge {10, 20, 200},  // Out of order timestamps
        Edge {20, 30, 100}
    };
    this->graph->add_multiple_edges(edges1, 30);

    // Check first addition is sorted
    auto sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 200);

    // Second addition with timestamps that need to be merged
    std::vector<Edge> edges2 = {
        Edge {30, 40, 150},
        Edge {40, 50, 250}
    };
    this->graph->add_multiple_edges(edges2, 50);

    // Verify all timestamps are still sorted
    sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 4);
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 150);
    EXPECT_EQ(sorted_edges[2].ts, 200);
    EXPECT_EQ(sorted_edges[3].ts, 250);

    // Third addition with duplicate timestamps
    std::vector<Edge> edges3 = {
        Edge {50, 60, 150},
        Edge {60, 70, 200},
        Edge {70, 80, 175}
    };
    this->graph->add_multiple_edges(edges3, 80);

    // Verify order is maintained with duplicates
    sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 7);
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 150);
    EXPECT_EQ(sorted_edges[2].ts, 150);
    EXPECT_EQ(sorted_edges[3].ts, 175);
    EXPECT_EQ(sorted_edges[4].ts, 200);
    EXPECT_EQ(sorted_edges[5].ts, 200);
    EXPECT_EQ(sorted_edges[6].ts, 250);

    // Verify node timestamp groups are correct
    // Node 30 has edges at 100 (inbound) and 150 (outbound)
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(30, 50), 1);   // Should see 150
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(30, 200), 1);     // Should see 100
}

// Test time window functionality
TYPED_TEST(TemporalGraphTest, TimeWindowTest) {
    // Create graph with 100 time unit window
    this->graph = std::make_unique<TemporalGraph>(true, TypeParam::value, 100);

    // Add edges spanning the time window
    std::vector<Edge> edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 150},
        Edge {3, 4, 249}  // This should cause deletion of first edge
    };

    this->graph->add_multiple_edges(edges, 4);
    const auto remaining_edges = this->graph->get_edges();

    EXPECT_EQ(remaining_edges.size(), 2);
    EXPECT_EQ(remaining_edges[0].ts, 150);
    EXPECT_EQ(remaining_edges[1].ts, 249);
}

// Test edge cases and corner cases
TYPED_TEST(TemporalGraphTest, EdgeAdditionEdgeCasesTest) {
    // Test empty edge list
    this->graph->add_multiple_edges({}, -1);
    EXPECT_TRUE(this->graph->get_edges().empty());

    // Test single edge
    std::vector<Edge> single_edge = {Edge {1, 2, 100}};
    this->graph->add_multiple_edges(single_edge, 2);
    EXPECT_EQ(this->graph->get_edges().size(), 1);

    // Test duplicate timestamps
    const std::vector<Edge> dup_time_edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 4, 100}
    };
    this->graph->add_multiple_edges(dup_time_edges, 4);
    EXPECT_EQ(this->graph->get_edges().size(), 4);

    // Test max int64 timestamp
    const std::vector<Edge> max_time_edge = {
        Edge {1, 2, INT64_MAX}
    };
    this->graph->add_multiple_edges(max_time_edge, 4);
    EXPECT_EQ(this->graph->get_edges().size(), 5);
}

// Test deletion of nodes when all their edges are removed
TYPED_TEST(TemporalGraphTest, NodeDeletionTest) {
    this->graph = std::make_unique<TemporalGraph>(true, TypeParam::value, 100);  // 100 time unit window

    // Add initial edges
    std::vector<Edge> edges1 = {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 1, 100}
    };
    this->graph->add_multiple_edges(edges1, 3);
    EXPECT_EQ(this->graph->get_node_ids().size(), 3);

    // Add edge that causes old edges to be deleted
    std::vector<Edge> edges2 = {
        Edge {4, 5, 250}  // Should cause deletion of all previous edges
    };
    this->graph->add_multiple_edges(edges2, 5);

    auto remaining_nodes = this->graph->get_node_ids();
    EXPECT_EQ(remaining_nodes.size(), 2);  // Only nodes 4 and 5 should remain
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 4) != remaining_nodes.end());
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 5) != remaining_nodes.end());
}

// Test undirected graph behavior
TYPED_TEST(TemporalGraphTest, UndirectedGraphEdgeAdditionTest) {
    this->graph = std::make_unique<TemporalGraph>(false, TypeParam::value);  // Undirected

    std::vector<Edge> edges = {
        Edge {2, 1, 100},  // Should be stored as (1,2,100)
        Edge {3, 1, 200},  // Should be stored as (1,3,200)
    };

    this->graph->add_multiple_edges(edges, 3);
    const auto stored_edges = this->graph->get_edges();

    // Verify edges are stored with lower node ID as source
    EXPECT_EQ(stored_edges[0].u, 1);
    EXPECT_EQ(stored_edges[0].i, 2);
    EXPECT_EQ(stored_edges[1].u, 1);
    EXPECT_EQ(stored_edges[1].i, 3);
}

TYPED_TEST(TemporalGraphTest, CountTimestampsTest) {
   // Set up a graph with carefully chosen timestamps
   std::vector<Edge> edges = {
       Edge {1, 2, 100},  // t0
       Edge {2, 3, 100},  // t0 duplicate
       Edge {1, 3, 200},  // t1
       Edge {2, 4, 300},  // t2
       Edge {3, 4, 300},  // t2 duplicate
       Edge {4, 1, 400}   // t3
   };
   this->graph->add_multiple_edges(edges, 4);

   // Test count_timestamps_less_than
   EXPECT_EQ(this->graph->count_timestamps_less_than(50), 0);   // Before first timestamp
   EXPECT_EQ(this->graph->count_timestamps_less_than(100), 0);  // At first timestamp
   EXPECT_EQ(this->graph->count_timestamps_less_than(150), 1);  // Between t0 and t1
   EXPECT_EQ(this->graph->count_timestamps_less_than(200), 1);  // At t1
   EXPECT_EQ(this->graph->count_timestamps_less_than(300), 2);  // At t2
   EXPECT_EQ(this->graph->count_timestamps_less_than(400), 3);  // At t3
   EXPECT_EQ(this->graph->count_timestamps_less_than(500), 4);  // After last timestamp

   // Test count_timestamps_greater_than
   EXPECT_EQ(this->graph->count_timestamps_greater_than(50), 4);   // Before first timestamp
   EXPECT_EQ(this->graph->count_timestamps_greater_than(100), 3);  // At first timestamp
   EXPECT_EQ(this->graph->count_timestamps_greater_than(150), 3);  // Between t0 and t1
   EXPECT_EQ(this->graph->count_timestamps_greater_than(200), 2);  // At t1
   EXPECT_EQ(this->graph->count_timestamps_greater_than(300), 1);  // At t2
   EXPECT_EQ(this->graph->count_timestamps_greater_than(400), 0);  // At t3
   EXPECT_EQ(this->graph->count_timestamps_greater_than(500), 0);  // After last timestamp

   // Test empty graph
   this->graph = std::make_unique<TemporalGraph>(true, TypeParam::value);
   EXPECT_EQ(this->graph->count_timestamps_less_than(100), 0);
   EXPECT_EQ(this->graph->count_timestamps_greater_than(100), 0);
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsDirectedTest) {
    // Set up directed graph with careful node-timestamp patterns
    std::vector<Edge> edges = {
        Edge {1, 2, 100},  // Node 1 out: t0  | Node 2 in: t0
        Edge {1, 3, 100},  // Node 1 out: t0  | Node 3 in: t0
        Edge {2, 1, 200},  // Node 2 out: t1  | Node 1 in: t1
        Edge {1, 2, 300},  // Node 1 out: t2  | Node 2 in: t2
        Edge {3, 1, 300},  // Node 3 out: t2  | Node 1 in: t2
        Edge {1, 4, 400}   // Node 1 out: t3  | Node 4 in: t3
    };
    this->graph->add_multiple_edges(edges, 4);

    // Test outbound edges for node 1 (count_node_timestamps_greater_than)
    // Node 1's outbound edges are at: 100(x2), 300, 400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 50), 3);    // Should see 100,300,400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 100), 2);   // Should see 300,400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 200), 2);   // Should see 300,400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 300), 1);   // Should see 400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 400), 0);   // Nothing after 400
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 500), 0);   // Nothing after 500

    // Test inbound edges for node 1 (count_node_timestamps_less_than)
    // Node 1's inbound edges are at: 200, 300
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 50), 0);     // Nothing before 50
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 200), 0);    // Nothing before 200
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 250), 1);    // Should see 200
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 400), 2);    // Should see 200,300
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 500), 2);    // Should see 200,300

    // Test node 2's timestamps
    // Node 2 outbound: 200
    // Node 2 inbound: 100, 300
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(2, 50), 1);   // Should see 200
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(2, 200), 0);  // Nothing after 200
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(2, 400), 2);    // Should see 100,300

    // Test node with no edges
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(5, 100), 0);
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(5, 100), 0);

    // Test invalid node ID
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(-1, 100), 0);
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(-1, 100), 0);
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsUndirectedTest) {
   // Create undirected graph
   this->graph = std::make_unique<TemporalGraph>(false, TypeParam::value);

   // Add edges - note that order will be normalized (smaller ID becomes source)
   std::vector<Edge> edges = {
       Edge {2, 1, 100},  // Will be stored as (1,2,100)
       Edge {3, 1, 100},  // Will be stored as (1,3,100)
       Edge {1, 2, 200},  // Will be stored as (1,2,200)
       Edge {4, 1, 300},  // Will be stored as (1,4,300)
       Edge {1, 3, 300}   // Will be stored as (1,3,300)
   };
   this->graph->add_multiple_edges(edges, 4);

   // Test timestamps for node 1
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 50), 3);   // Before first
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 100), 2);  // At first
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 200), 1);  // At middle
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 300), 0);  // At last
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 400), 0);  // After last

   EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 50), 0);    // Before first
   EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 100), 0);   // At first
   EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 150), 1);   // After first
   EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 400), 3);   // After last

   // Test edge target node (node 2)
   EXPECT_EQ(this->graph->count_node_timestamps_greater_than(2, 50), 2);   // Should see both t0 and t1
   EXPECT_EQ(this->graph->count_node_timestamps_less_than(2, 250), 2);     // Should see both t0 and t1
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsDuplicatesTest) {
    // Test handling of duplicate timestamps for a node
    std::vector<Edge> edges = {
        Edge {1, 2, 100},  // t0 outbound from node 1
        Edge {1, 3, 100},  // t0 duplicate outbound from node 1
        Edge {1, 4, 100},  // t0 triplicate outbound from node 1
        Edge {2, 1, 200},  // t1 inbound to node 1
        Edge {3, 1, 200},  // t1 duplicate inbound to node 1
    };
    this->graph->add_multiple_edges(edges, 4);

    // Test outbound timestamps for node 1
    // Node 1 has outbound edges only at t0 (100)
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 50), 1);    // Should see only t0
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(1, 100), 0);   // Nothing after t0

    // Test inbound timestamps for node 1
    // Node 1 has inbound edges only at t1 (200)
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 250), 1);     // Should see only t1
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(1, 150), 0);     // Nothing before t1

    // Add test for target nodes
    EXPECT_EQ(this->graph->count_node_timestamps_less_than(2, 150), 1);     // Node 2 receives at t0
    EXPECT_EQ(this->graph->count_node_timestamps_greater_than(2, 50), 1);   // Node 2 sends at t1
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtTest) {
    // Set up test graph with carefully structured timestamps
    const std::vector<Edge> edges = {
        Edge {10, 20, 100},  // Group 0: timestamp 100
        Edge {30, 40, 100},
        Edge {50, 60, 200},  // Group 1: timestamp 200
        Edge {70, 80, 300},  // Group 2: timestamp 300
        Edge {90, 100, 300},
        Edge {110, 120, 400}  // Group 3: timestamp 400
    };
    this->graph->add_multiple_edges(edges, 120);

    // Test forward direction (looking for timestamps > given)

    // Test with timestamp = -1 (no constraint)
    auto [src1, tgt1, ts1] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, -1, true);
    EXPECT_EQ(ts1, 100);  // Should select from first group

    auto [src2, tgt2, ts2] = this->graph->get_edge_at(RandomPickerType::TEST_LAST, -1, true);
    EXPECT_EQ(ts2, 400);  // Should select from last group

    // Test with timestamp constraints
    auto [src3, tgt3, ts3] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(ts3, 200);  // Should select first group after 100

    auto [src4, tgt4, ts4] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 300, true);
    EXPECT_EQ(ts4, 400);  // Should select first group after 300

    // Test backward direction (looking for timestamps < given)
    auto [src5, tgt5, ts5] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 400, false);
    EXPECT_EQ(ts5, 100);  // Should select first group before 400

    auto [src6, tgt6, ts6] = this->graph->get_edge_at(RandomPickerType::TEST_LAST, 250, false);
    EXPECT_EQ(ts6, 200);  // Should select latest group before 250

    // Test edge cases
    // No groups after timestamp
    auto [src7, tgt7, ts7] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 500, true);
    EXPECT_EQ(ts7, -1);  // Should return -1 when no valid groups

    // No groups before timestamp
    auto [src8, tgt8, ts8] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 50, false);
    EXPECT_EQ(ts8, -1);

    // Test with empty graph
    this->graph = std::make_unique<TemporalGraph>(true, TypeParam::value);
    auto [src9, tgt9, ts9] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(ts9, -1);
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtDuplicateTimestampsTest) {
    // Setup graph with multiple edges in same timestamp groups
    std::vector<Edge> edges = {
        Edge {10, 20, 100},  // Group 0: three edges at 100
        Edge {30, 40, 100},
        Edge {50, 60, 100},
        Edge {70, 80, 200},  // Group 1: two edges at 200
        Edge {90, 100, 200},
        Edge {110, 120, 300}  // Group 2: single edge at 300
    };
    this->graph->add_multiple_edges(edges, 120);

    // Test forward selection
    auto [src1, tgt1, ts1] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 50, true);
    EXPECT_EQ(ts1, 100);
    EXPECT_TRUE((src1 == 10 && tgt1 == 20) ||
                (src1 == 30 && tgt1 == 40) ||
                (src1 == 50 && tgt1 == 60));  // Should be one of the t=100 edges

    // Test backward selection
    auto [src2, tgt2, ts2] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 250, false);
    EXPECT_EQ(ts2, 100);
    EXPECT_TRUE((src2 == 10 && tgt2 == 20) ||
                (src2 == 30 && tgt2 == 40) ||
                (src2 == 50 && tgt2 == 60));  // Should be one of the t=100 edges
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtBoundaryConditionsTest) {
    // Test exact timestamp boundaries
    std::vector<Edge> edges = {
        Edge {10, 20, 100},
        Edge {30, 40, 200},
        Edge {50, 60, 300}
    };
    this->graph->add_multiple_edges(edges, 60);

    // Forward direction
    auto [src1, tgt1, ts1] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(ts1, 200);  // Should get next timestamp

    auto [src2, tgt2, ts2] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 300, true);
    EXPECT_EQ(ts2, -1);   // No timestamps after 300

    // Backward direction
    auto [src3, tgt3, ts3] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 200, false);
    EXPECT_EQ(ts3, 100);  // Should get previous timestamp

    auto [src4, tgt4, ts4] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 100, false);
    EXPECT_EQ(ts4, -1);   // No timestamps before 100
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtRandomSelectionTest) {
    // Test that we can get different edges from same group
    const std::vector<Edge> edges = {
        Edge {1, 2, 100},
        Edge {3, 4, 100},
        Edge {5, 6, 100}
    };
    this->graph->add_multiple_edges(edges, 6);

    // Make multiple selections and verify we can get different edges
    std::set<std::pair<int, int>> seen_edges;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto [src, tgt, ts] = this->graph->get_edge_at(RandomPickerType::TEST_FIRST, 50, true);
        EXPECT_EQ(ts, 100);
        seen_edges.insert({src, tgt});
    }

    // We should see more than one edge (due to random selection within group)
    EXPECT_GT(seen_edges.size(), 1);
}
