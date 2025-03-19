#include <gtest/gtest.h>
#include "../src/data/IntHashMap.cuh"

template<typename T>
class IntHashMapTest : public ::testing::Test {
protected:
    IntHashMap hash_map;

    IntHashMapTest() : hash_map(16, T::value) {}
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

TYPED_TEST_SUITE(IntHashMapTest, GPU_USAGE_TYPES);

// Test empty state
TYPED_TEST(IntHashMapTest, EmptyStateTest) {
    EXPECT_FALSE(this->hash_map.has_value_host(1));
    EXPECT_EQ(this->hash_map.get_host(1, -999), -999);
}

// Test basic insert and get
TYPED_TEST(IntHashMapTest, BasicInsertGetTest) {
    this->hash_map.insert_host(10, 100);
    this->hash_map.insert_host(20, 200);
    this->hash_map.insert_host(30, 300);

    EXPECT_EQ(this->hash_map.get_host(10, -1), 100);
    EXPECT_EQ(this->hash_map.get_host(20, -1), 200);
    EXPECT_EQ(this->hash_map.get_host(30, -1), 300);
    EXPECT_EQ(this->hash_map.get_host(40, -1), -1);
}

// Test default value return
TYPED_TEST(IntHashMapTest, DefaultValueTest) {
    this->hash_map.insert_host(100, 1000);

    EXPECT_EQ(this->hash_map.get_host(100, -1), 1000);
    EXPECT_EQ(this->hash_map.get_host(999, -1), -1);
}

// Test value update
TYPED_TEST(IntHashMapTest, ValueUpdateTest) {
    this->hash_map.insert_host(50, 500);
    EXPECT_EQ(this->hash_map.get_host(50, -1), 500);

    // Update existing key
    this->hash_map.insert_host(50, 5000);
    EXPECT_EQ(this->hash_map.get_host(50, -1), 5000);
}

// Test exists check
TYPED_TEST(IntHashMapTest, ExistsCheckTest) {
    this->hash_map.insert_host(123, 456);

    EXPECT_TRUE(this->hash_map.has_value_host(123));
    EXPECT_FALSE(this->hash_map.has_value_host(456));
}

// Test collision handling
TYPED_TEST(IntHashMapTest, CollisionTest) {
    // Create a map with size 4 (will have capacity 4)
    IntHashMap small_map(4, TypeParam::value);

    // Insert keys that will collide in a small table
    small_map.insert_host(4, 400);   // Hash: 4 & 3 = 0
    small_map.insert_host(8, 800);   // Hash: 8 & 3 = 0 (collision)
    small_map.insert_host(12, 1200); // Hash: 12 & 3 = 0 (collision)
    small_map.insert_host(5, 500);   // Hash: 5 & 3 = 1

    // Verify all values are correctly retrievable
    EXPECT_EQ(small_map.get_host(4, -1), 400);
    EXPECT_EQ(small_map.get_host(8, -1), 800);
    EXPECT_EQ(small_map.get_host(12, -1), 1200);
    EXPECT_EQ(small_map.get_host(5, -1), 500);
}

// Test EMPTY_KEY handling
TYPED_TEST(IntHashMapTest, EmptyKeyTest) {
    // Trying to insert EMPTY_KEY should be a no-op
    this->hash_map.insert_host(EMPTY_KEY, 999);

    EXPECT_FALSE(this->hash_map.has_value_host(EMPTY_KEY));
    EXPECT_EQ(this->hash_map.get_host(EMPTY_KEY, -1), -1);
}

// Test with large keys (big integers)
TYPED_TEST(IntHashMapTest, LargeKeysTest) {
    this->hash_map.insert_host(1000000, 1);
    this->hash_map.insert_host(2000000, 2);
    this->hash_map.insert_host(1000000000, 3);

    EXPECT_EQ(this->hash_map.get_host(1000000, -1), 1);
    EXPECT_EQ(this->hash_map.get_host(2000000, -1), 2);
    EXPECT_EQ(this->hash_map.get_host(1000000000, -1), 3);
}

// Test capacity limits
TYPED_TEST(IntHashMapTest, CapacityTest) {
    // Create a small hash map and fill it to capacity
    IntHashMap small_map(4, TypeParam::value);

    // Insert values up to capacity
    for (int i = 0; i < 4; i++) {
        small_map.insert_host(i, i * 10);
    }

    // Verify all values are present
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(small_map.get_host(i, -1), i * 10);
    }

    // Even when full, we should be able to update existing keys
    small_map.insert_host(2, 25);
    EXPECT_EQ(small_map.get_host(2, -1), 25);
}

TYPED_TEST(IntHashMapTest, ElementCountTest) {
    // Initial count should be zero
    EXPECT_EQ(this->hash_map.size(), 0);

    // Insert new elements and check count increases
    this->hash_map.insert_host(1, 10);
    EXPECT_EQ(this->hash_map.size(), 1);

    this->hash_map.insert_host(2, 20);
    EXPECT_EQ(this->hash_map.size(), 2);

    // Updating existing element should not change count
    this->hash_map.insert_host(1, 100);
    EXPECT_EQ(this->hash_map.size(), 2);

    this->hash_map.insert_host(3, 30);
    EXPECT_EQ(this->hash_map.size(), 3);
}
