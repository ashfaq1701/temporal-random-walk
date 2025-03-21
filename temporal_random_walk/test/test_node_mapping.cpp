#include <unordered_set>
#include <gtest/gtest.h>

#include "../src/proxies/NodeMapping.cuh"
#include "../src/proxies/EdgeData.cuh"
#include "../src/common/const.cuh"

template<typename T>
class NodeMappingTest : public ::testing::Test {
protected:
    NodeMapping mapping;
    EdgeData edges;

    NodeMappingTest(): mapping(DEFAULT_NODE_COUNT_MAX_BOUND, T::value), edges(T::value) {}

    // Helper to verify bidirectional mapping
    void verify_mapping_exists(int sparse_id) const {
        EXPECT_NE(mapping.to_dense(sparse_id), -1);
    }

    void verify_mapping_no_duplicate(const std::vector<int>& sparse_ids) const {
        std::unordered_set<int> dense_ids;

        for (const int sparse_id : sparse_ids) {
            int dense_id = mapping.to_dense(sparse_id);
            EXPECT_NE(dense_id, -1) << "Sparse ID " << sparse_id << " has no mapping";

            // Check if this dense ID has already been seen
            EXPECT_EQ(dense_ids.count(dense_id), 0)
                << "Dense ID " << dense_id << " is duplicated (mapped from sparse ID " << sparse_id << ")";

            // Add to set of seen dense IDs
            dense_ids.insert(dense_id);
        }

        // Optional: verify count matches
        EXPECT_EQ(dense_ids.size(), sparse_ids.size())
            << "Number of unique dense IDs doesn't match number of sparse IDs";
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
    this->verify_mapping_exists(10);  // First node gets dense index 0
    this->verify_mapping_exists(20);  // Second node gets dense index 1
    this->verify_mapping_exists(30);  // Third node gets dense index 2

    this->verify_mapping_no_duplicate({10, 20, 30});

    // Verify node existence
    EXPECT_TRUE(this->mapping.has_node(10));
    EXPECT_TRUE(this->mapping.has_node(20));
    EXPECT_TRUE(this->mapping.has_node(30));
    EXPECT_FALSE(this->mapping.has_node(15));  // Non-existent node
}

// Test incremental updates
TYPED_TEST(NodeMappingTest, IncrementalUpdateTest) {
    // First update
    this->edges.push_back(10, 20, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);

    this->verify_mapping_exists(10);
    this->verify_mapping_exists(20);

    // Second update with new nodes
    this->edges.push_back(30, 40, 200);
    this->mapping.update(this->edges.edge_data, 1, 2);

    this->verify_mapping_exists(30);
    this->verify_mapping_exists(40);

    this->verify_mapping_no_duplicate({10, 20, 30, 40});

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
    this->verify_mapping_exists(1);
    this->verify_mapping_exists(1000000);

    this->verify_mapping_no_duplicate({1, 1000000});

    // Test marking non-existent node as deleted
    this->mapping.mark_node_deleted(999);  // Should not crash

    // Test empty range update
    this->mapping.update(this->edges.edge_data, 0, 0);  // Should handle empty range gracefully
}

// Test reservation and clear
TYPED_TEST(NodeMappingTest, ClearTest) {
    this->edges.push_back(10, 20, 100);
    this->mapping.update(this->edges.edge_data, 0, 1);

    this->mapping.clear();
    EXPECT_EQ(this->mapping.size(), 0);
    EXPECT_EQ(this->mapping.active_size(), 0);
    EXPECT_FALSE(this->mapping.has_node(10));
}
