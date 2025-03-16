#include <gtest/gtest.h>

#include "../src/proxies/NodeMappingProxy.cuh"
#include "../src/proxies/EdgeDataProxy.cuh"

template<typename T>
class NodeMappingTest : public ::testing::Test {
protected:
    NodeMappingProxy mapping;
    EdgeDataProxy edges;

    NodeMappingTest(): mapping(T::value), edges(T::value) {}

    // Helper to verify bidirectional mapping
    void verify_mapping(int sparse_id, int expected_dense_idx) const {
        EXPECT_EQ(mapping.to_dense(sparse_id), expected_dense_idx);
        if (expected_dense_idx != -1) {
            EXPECT_EQ(mapping.to_sparse(expected_dense_idx), sparse_id);
        }
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

TYPED_TEST_SUITE(NodeMappingTest, GPU_USAGE_TYPES);

// Test empty state
TYPED_TEST(NodeMappingTest, EmptyStateTest) {
    EXPECT_EQ(this->mapping.size(), 0);
    EXPECT_EQ(this->mapping.active_size(), 0);
    EXPECT_TRUE(this->mapping.get_active_node_ids().empty());
    EXPECT_TRUE(this->mapping.get_all_sparse_ids().empty());

    // Test invalid mappings in empty state
    EXPECT_EQ(this->mapping.to_dense(0), -1);
    EXPECT_EQ(this->mapping.to_dense(-1), -1);
    EXPECT_EQ(this->mapping.to_sparse(0), -1);
    EXPECT_FALSE(this->mapping.has_node(0));
}

// Test basic update functionality
TYPED_TEST(NodeMappingTest, BasicUpdateTest) {
    this->edges.push_back(10, 20, 100);
    this->edges.push_back(20, 30, 200);
    this->mapping.update(this->edges.edge_data, 0, this->edges.size());

    // Verify sizes
    EXPECT_EQ(this->mapping.size(), 3);  // 3 unique nodes
    EXPECT_EQ(this->mapping.active_size(), 3);  // All nodes active

    // Verify mappings
    this->verify_mapping(10, 0);  // First node gets dense index 0
    this->verify_mapping(20, 1);  // Second node gets dense index 1
    this->verify_mapping(30, 2);  // Third node gets dense index 2

    // Verify node existence
    EXPECT_TRUE(this->mapping.has_node(10));
    EXPECT_TRUE(this->mapping.has_node(20));
    EXPECT_TRUE(this->mapping.has_node(30));
    EXPECT_FALSE(this->mapping.has_node(15));  // Non-existent node
}

// Test handling of gaps in sparse IDs
TYPED_TEST(NodeMappingTest, SparseGapsTest) {
    this->edges.push_back(10, 50, 100);  // Gap between 10 and 50
    this->mapping.update(this->edges.edge_data, 0, this->edges.size());

    // Verify size handling with gaps
    EXPECT_EQ(this->mapping.size(), 2);  // Only 2 actual nodes
    EXPECT_GE(this->mapping.sparse_to_dense().size(), 51);  // But space for all up to 50

    // Verify mappings
    this->verify_mapping(10, 0);
    this->verify_mapping(50, 1);

    // Verify nodes in gap don't exist
    for (int i = 11; i < 50; i++) {
        EXPECT_FALSE(this->mapping.has_node(i));
        EXPECT_EQ(this->mapping.to_dense(i), -1);
    }
}

// Test incremental updates
TYPED_TEST(NodeMappingTest, IncrementalUpdateTest) {
    // First update
    this->edges.push_back(10, 20, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);

    this->verify_mapping(10, 0);
    this->verify_mapping(20, 1);

    // Second update with new nodes
    this->edges.push_back(30, 40, 200);
    this->mapping.update(this->edges.edge_data, 1, 2);

    this->verify_mapping(30, 2);
    this->verify_mapping(40, 3);

    // Third update with existing nodes
    this->edges.push_back(20, 30, 300);  // Both nodes already exist
    this->mapping.update(this->edges.edge_data, 2, 3);

    EXPECT_EQ(this->mapping.size(), 4);  // No new nodes added
}

// Test node deletion
TYPED_TEST(NodeMappingTest, NodeDeletionTest) {
    this->edges.push_back(10, 20, 100);
    this->edges.push_back(20, 30, 200);
    this->mapping.update(this->edges.edge_data, 0, this->edges.size());

    // Delete node 20
    this->mapping.mark_node_deleted(20);

    // Verify counts
    EXPECT_EQ(this->mapping.size(), 3);        // Total size unchanged
    EXPECT_EQ(this->mapping.active_size(), 2);  // But one less active node

    // Verify active nodes list
    auto active_nodes = this->mapping.get_active_node_ids();
    EXPECT_EQ(active_nodes.size(), 2);
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 10) != active_nodes.end());
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 30) != active_nodes.end());
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 20) == active_nodes.end());

    // Mapping should still work for deleted nodes
    EXPECT_NE(this->mapping.to_dense(20), -1);
}

// Test edge cases and invalid inputs
TYPED_TEST(NodeMappingTest, EdgeCasesTest) {
    // Test with negative IDs
    this->edges.push_back(-1, -2, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);
    EXPECT_EQ(this->mapping.to_dense(-1), -1);  // Should not map negative IDs
    EXPECT_EQ(this->mapping.to_dense(-2), -1);

    // Test with very large sparse ID
    this->edges.clear();
    this->edges.push_back(1000000, 1, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);
    this->verify_mapping(1, 0);
    this->verify_mapping(1000000, 1);

    // Test invalid dense indices
    EXPECT_EQ(this->mapping.to_sparse(-1), -1);
    EXPECT_EQ(this->mapping.to_sparse(1000000), -1);

    // Test marking non-existent node as deleted
    this->mapping.mark_node_deleted(999);  // Should not crash

    // Test empty range update
    this->mapping.update(this->edges.edge_data, 0, 0);  // Should handle empty range gracefully
}

// Test reservation and clear
TYPED_TEST(NodeMappingTest, ReservationAndClearTest) {
    this->mapping.reserve(100);

    this->edges.push_back(10, 20, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);

    this->mapping.clear();
    EXPECT_EQ(this->mapping.size(), 0);
    EXPECT_EQ(this->mapping.active_size(), 0);
    EXPECT_FALSE(this->mapping.has_node(10));
}
