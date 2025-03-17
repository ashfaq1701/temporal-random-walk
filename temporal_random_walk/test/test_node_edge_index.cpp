#include <gtest/gtest.h>
#include "../src/proxies/NodeEdgeIndex.cuh"
#include "../src/proxies/EdgeData.cuh"
#include "../src/proxies/NodeMapping.cuh"

template<typename  T>
class NodeEdgeIndexTest : public ::testing::Test {
protected:
    NodeEdgeIndex index;
    EdgeData edges;
    NodeMapping mapping;

    NodeEdgeIndexTest(): index(T::value), edges(T::value), mapping(T::value) {}

    // Helper function to set up a simple directed graph
    void setup_simple_directed_graph() {
        // Add edges with timestamps
        edges.push_back(10, 20, 100);  // Edge 0
        edges.push_back(10, 30, 100);  // Edge 1 - same timestamp as Edge 0
        edges.push_back(10, 20, 200);  // Edge 2 - new timestamp
        edges.push_back(20, 30, 300);  // Edge 3
        edges.push_back(20, 10, 300);  // Edge 4 - same timestamp as Edge 3
        edges.update_timestamp_groups();

        // Update node mapping
        mapping.update(edges.edge_data, 0, edges.size());

        // Rebuild index
        index.rebuild(edges.edge_data, mapping.node_mapping, true);
    }

    // Helper function to set up a simple undirected graph
    void setup_simple_undirected_graph() {
        // Add edges with timestamps
        edges.push_back(100, 200, 1000);  // Edge 0
        edges.push_back(100, 300, 1000);  // Edge 1 - same timestamp as Edge 0
        edges.push_back(100, 200, 2000);  // Edge 2 - new timestamp
        edges.push_back(200, 300, 3000);  // Edge 3
        edges.update_timestamp_groups();

        // Update node mapping
        mapping.update(edges.edge_data, 0, edges.size());

        // Rebuild index
        index.rebuild(edges.edge_data, mapping.node_mapping, false);
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

TYPED_TEST_SUITE(NodeEdgeIndexTest, GPU_USAGE_TYPES);

// Test empty state
TYPED_TEST(NodeEdgeIndexTest, EmptyStateTest) {
    EXPECT_TRUE(this->index.outbound_offsets().empty());
    EXPECT_TRUE(this->index.outbound_indices().empty());
    EXPECT_TRUE(this->index.outbound_timestamp_group_offsets().empty());
    EXPECT_TRUE(this->index.outbound_timestamp_group_indices().empty());
    EXPECT_TRUE(this->index.inbound_offsets().empty());
    EXPECT_TRUE(this->index.inbound_indices().empty());
    EXPECT_TRUE(this->index.inbound_timestamp_group_offsets().empty());
    EXPECT_TRUE(this->index.inbound_timestamp_group_indices().empty());
}

// Test edge ranges in directed graph
TYPED_TEST(NodeEdgeIndexTest, DirectedEdgeRangeTest) {
    this->setup_simple_directed_graph();

    // Check outbound edges for node 10
    const int dense_node10 = this->mapping.to_dense(10);
    auto [out_start10, out_end10] = this->index.get_edge_range(dense_node10, true, true);
    EXPECT_EQ(out_end10 - out_start10, 3);  // Node 10 has 3 outbound edges (to 20,30,20)

    // Verify each outbound edge
    for (size_t i = out_start10; i < out_end10; i++) {
        size_t edge_idx = this->index.outbound_indices()[i];
        EXPECT_EQ(this->edges.sources()[edge_idx], 10);  // All should be from node 10
    }

    // Check inbound edges for node 20
    const int dense_node20 = this->mapping.to_dense(20);
    auto [in_start20, in_end20] = this->index.get_edge_range(dense_node20, false, true);
    EXPECT_EQ(in_end20 - in_start20, 2);  // Node 20 has 2 inbound edges from node 10

    // Verify each inbound edge
    for (size_t i = in_start20; i < in_end20; i++) {
        size_t edge_idx = this->index.inbound_indices()[i];
        EXPECT_EQ(this->edges.targets()[edge_idx], 20);  // All should be to node 20
    }

    // Check invalid node
    auto [inv_start, inv_end] = this->index.get_edge_range(-1, true, true);
    EXPECT_EQ(inv_start, 0);
    EXPECT_EQ(inv_end, 0);
}

// Test timestamp groups in directed graph
TYPED_TEST(NodeEdgeIndexTest, DirectedTimestampGroupTest) {
    this->setup_simple_directed_graph();

    const int dense_node10 = this->mapping.to_dense(10);

    // Check outbound groups for node 10
    EXPECT_EQ(this->index.get_timestamp_group_count(dense_node10, true, true), 2);  // Two groups: 100, 200

    // Check first group range (timestamp 100)
    auto [group1_start, group1_end] = this->index.get_timestamp_group_range(dense_node10, 0, true, true);
    EXPECT_EQ(group1_end - group1_start, 2);  // Two edges in timestamp 100

    // Verify all edges in first group
    for (size_t i = group1_start; i < group1_end; i++) {
        size_t edge_idx = this->index.outbound_indices()[i];
        EXPECT_EQ(this->edges.timestamps()[edge_idx], 100);
        EXPECT_EQ(this->edges.sources()[edge_idx], 10);
        EXPECT_TRUE(this->edges.targets()[edge_idx] == 20 || this->edges.targets()[edge_idx] == 30);
    }

    // Check second group range (timestamp 200)
    auto [group2_start, group2_end] = this->index.get_timestamp_group_range(dense_node10, 1, true, true);
    EXPECT_EQ(group2_end - group2_start, 1);  // One edge in timestamp 200

    // Verify edge in second group
    size_t edge_idx = this->index.outbound_indices()[group2_start];
    EXPECT_EQ(this->edges.timestamps()[edge_idx], 200);
    EXPECT_EQ(this->edges.sources()[edge_idx], 10);
    EXPECT_EQ(this->edges.targets()[edge_idx], 20);
}

// Test edge ranges in undirected graph
TYPED_TEST(NodeEdgeIndexTest, UndirectedEdgeRangeTest) {
    this->setup_simple_undirected_graph();

    // In undirected graph, all edges are stored as outbound
    const int dense_node100 = this->mapping.to_dense(100);
    auto [out_start100, out_end100] = this->index.get_edge_range(dense_node100, true, false);
    EXPECT_EQ(out_end100 - out_start100, 3);  // Node 100 has 3 edges (to 200,300,200)

    // Verify each edge for node 100
    for (size_t i = out_start100; i < out_end100; i++) {
        size_t edge_idx = this->index.outbound_indices()[i];
        EXPECT_TRUE(
            (this->edges.sources()[edge_idx] == 100 && (this->edges.targets()[edge_idx] == 200 || this->edges.targets()[edge_idx] == 300)) ||
            (this->edges.targets()[edge_idx] == 100 && (this->edges.sources()[edge_idx] == 200 || this->edges.sources()[edge_idx] == 300))
        );
    }

    const int dense_node200 = this->mapping.to_dense(200);
    auto [out_start200, out_end200] = this->index.get_edge_range(dense_node200, true, false);
    EXPECT_EQ(out_end200 - out_start200, 3);  // Node 200 has 3 edges (with 100,100,300)
}

// Test timestamp groups in undirected graph
TYPED_TEST(NodeEdgeIndexTest, UndirectedTimestampGroupTest) {
    this->setup_simple_undirected_graph();

    const int dense_node100 = this->mapping.to_dense(100);

    // Check timestamp groups for node 100
    EXPECT_EQ(this->index.get_timestamp_group_count(dense_node100, true, false), 2);  // Two timestamp groups (1000,2000)

    // Check first group (timestamp 1000)
    auto [group1_start, group1_end] = this->index.get_timestamp_group_range(dense_node100, 0, true, false);
    EXPECT_EQ(this->edges.timestamps()[this->index.outbound_indices()[group1_start]], 1000);
    EXPECT_EQ(group1_end - group1_start, 2);  // Two edges in timestamp 1000 group

    // Verify all edges in first group (timestamp 1000)
    for (size_t i = group1_start; i < group1_end; i++) {
        size_t edge_idx = this->index.outbound_indices()[i];
        EXPECT_EQ(this->edges.timestamps()[edge_idx], 1000);
        EXPECT_TRUE(
            (this->edges.sources()[edge_idx] == 100 && (this->edges.targets()[edge_idx] == 200 || this->edges.targets()[edge_idx] == 300)) ||
            (this->edges.targets()[edge_idx] == 100 && (this->edges.sources()[edge_idx] == 200 || this->edges.sources()[edge_idx] == 300))
        );
    }

    // Check second group (timestamp 2000)
    auto [group2_start, group2_end] = this->index.get_timestamp_group_range(dense_node100, 1, true, false);
    EXPECT_EQ(this->edges.timestamps()[this->index.outbound_indices()[group2_start]], 2000);
    EXPECT_EQ(group2_end - group2_start, 1);  // One edge in timestamp 2000 group

    // Verify edge in second group (timestamp 2000)
    size_t edge_idx = this->index.outbound_indices()[group2_start];
    EXPECT_EQ(this->edges.timestamps()[edge_idx], 2000);
    EXPECT_TRUE(
        (this->edges.sources()[edge_idx] == 100 && this->edges.targets()[edge_idx] == 200) ||
        (this->edges.targets()[edge_idx] == 100 && this->edges.sources()[edge_idx] == 200)
    );
}

// Test edge cases and invalid inputs
TYPED_TEST(NodeEdgeIndexTest, EdgeCasesTest) {
    this->setup_simple_directed_graph();

    // Test invalid node ID
    EXPECT_EQ(this->index.get_timestamp_group_count(-1, true, true), 0);

    // Test invalid group index
    auto [inv_start, inv_end] = this->index.get_timestamp_group_range(this->mapping.to_dense(1), 999, true, true);
    EXPECT_EQ(inv_start, 0);
    EXPECT_EQ(inv_end, 0);

    // Test node with no edges
    this->edges.push_back(4, 5, 400);  // Add isolated node
    this->edges.update_timestamp_groups();
    this->mapping.update(this->edges.edge_data, this->edges.size()-1, this->edges.size());
    this->index.rebuild(this->edges.edge_data, this->mapping.node_mapping, true);

    int isolated_node = this->mapping.to_dense(4);
    EXPECT_EQ(this->index.get_timestamp_group_count(isolated_node, false, true), 0);
}
