#include <gtest/gtest.h>
#include "../src/proxies/EdgeDataProxy.cuh"

template<typename T>
class EdgeDataTest : public ::testing::Test {
protected:
    using EdgeDataType = EdgeData<T::value>;
    EdgeDataType edges;

   void verify_edge(const size_t index, const int expected_src, const int expected_tgt, const int64_t expected_ts) const {
       ASSERT_LT(index, this->edges.size());
       EXPECT_EQ(this->edges.sources()[index], expected_src);
       EXPECT_EQ(this->edges.targets()[index], expected_tgt);
       EXPECT_EQ(this->edges.timestamps()[index], expected_ts);
   }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>,
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_GPU>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>
>;
#endif

TYPED_TEST_SUITE(EdgeDataTest, GPU_USAGE_TYPES);

// Test empty state
TYPED_TEST(EdgeDataTest, EmptyStateTest) {
    EXPECT_TRUE(this->edges.empty());
    EXPECT_EQ(this->edges.size(), 0);
    EXPECT_TRUE(this->edges.timestamp_group_offsets().empty());
    EXPECT_TRUE(this->edges.unique_timestamps().empty());
}

// Test single edge
TYPED_TEST(EdgeDataTest, SingleEdgeTest) {
    this->edges.push_back(100, 200, 100);
    EXPECT_FALSE(this->edges.empty());
    EXPECT_EQ(this->edges.size(), 1);
    this->verify_edge(0, 100, 200, 100);

    this->edges.update_timestamp_groups();
    EXPECT_EQ(this->edges.unique_timestamps().size(), 1);
    EXPECT_EQ(this->edges.timestamp_group_offsets().size(), 2);  // n+1 offsets for n groups
    EXPECT_EQ(this->edges.timestamp_group_offsets()[0], 0);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[1], 1);
}

// Test multiple edges with same timestamp
TYPED_TEST(EdgeDataTest, SameTimestampEdgesTest) {
    this->edges.push_back(100, 200, 100);
    this->edges.push_back(200, 300, 100);
    this->edges.push_back(300, 400, 100);

    this->edges.update_timestamp_groups();
    EXPECT_EQ(this->edges.unique_timestamps().size(), 1);
    EXPECT_EQ(this->edges.timestamp_group_offsets().size(), 2);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[0], 0);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[1], 3);
}

// Test edges with different timestamps
TYPED_TEST(EdgeDataTest, DifferentTimestampEdgesTest) {
    this->edges.push_back(1, 2, 100);
    this->edges.push_back(2, 3, 200);
    this->edges.push_back(3, 4, 300);

    this->edges.update_timestamp_groups();
    EXPECT_EQ(this->edges.unique_timestamps().size(), 3);
    EXPECT_EQ(this->edges.timestamp_group_offsets().size(), 4);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[0], 0);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[1], 1);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[2], 2);
    EXPECT_EQ(this->edges.timestamp_group_offsets()[3], 3);
}

// Test find_group functions
TYPED_TEST(EdgeDataTest, FindGroupTest) {
    this->edges.push_back(100, 200, 100);
    this->edges.push_back(200, 300, 200);
    this->edges.push_back(300, 400, 300);
    this->edges.update_timestamp_groups();

    // Test find_group_after_timestamp
    EXPECT_EQ(this->edges.find_group_after_timestamp(50), 0);
    EXPECT_EQ(this->edges.find_group_after_timestamp(100), 1);
    EXPECT_EQ(this->edges.find_group_after_timestamp(150), 1);
    EXPECT_EQ(this->edges.find_group_after_timestamp(200), 2);
    EXPECT_EQ(this->edges.find_group_after_timestamp(300), 3);
    EXPECT_EQ(this->edges.find_group_after_timestamp(350), 3);

    // Test find_group_before_timestamp
    EXPECT_EQ(this->edges.find_group_before_timestamp(50), -1);
    EXPECT_EQ(this->edges.find_group_before_timestamp(150), 0);
    EXPECT_EQ(this->edges.find_group_before_timestamp(200), 0);
    EXPECT_EQ(this->edges.find_group_before_timestamp(300), 1);
    EXPECT_EQ(this->edges.find_group_before_timestamp(350), 2);
}

// Test timestamp group ranges
TYPED_TEST(EdgeDataTest, TimestampGroupRangeTest) {
    this->edges.push_back(100, 200, 100);  // Group 0
    this->edges.push_back(200, 300, 100);
    this->edges.push_back(300, 400, 200);  // Group 1
    this->edges.push_back(400, 500, 300);  // Group 2
    this->edges.push_back(500, 600, 300);
    this->edges.update_timestamp_groups();

    auto [start0, end0] = this->edges.get_timestamp_group_range(0);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 2);

    auto [start1, end1] = this->edges.get_timestamp_group_range(1);
    EXPECT_EQ(start1, 2);
    EXPECT_EQ(end1, 3);

    auto [start2, end2] = this->edges.get_timestamp_group_range(2);
    EXPECT_EQ(start2, 3);
    EXPECT_EQ(end2, 5);

    // Test invalid group index
    auto [invalid_start, invalid_end] = this->edges.get_timestamp_group_range(3);
    EXPECT_EQ(invalid_start, 0);
    EXPECT_EQ(invalid_end, 0);
}

// Test resize
TYPED_TEST(EdgeDataTest, ResizeTest) {
    this->edges.push_back(100, 200, 100);
    this->edges.push_back(200, 300, 200);

    this->edges.resize(1);
    EXPECT_EQ(this->edges.size(), 1);
    this->verify_edge(0, 100, 200, 100);

    this->edges.resize(3);
    EXPECT_EQ(this->edges.size(), 3);
    this->verify_edge(0, 100, 200, 100);
}
