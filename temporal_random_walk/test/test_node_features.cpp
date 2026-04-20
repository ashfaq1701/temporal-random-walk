#include <gtest/gtest.h>

#include <vector>

#include "../src/core/temporal_random_walk.cuh"
#include "../src/graph/node_features.cuh"

TEST(NodeFeaturesTest, SetNodeFeaturesPreservesOldDataOnGrow) {
    // node features are host-only regardless of use_gpu.
    core::TemporalRandomWalk trw(/*is_directed=*/true, /*use_gpu=*/false);
    auto& data = trw.data();

    std::vector<int> first_ids{1, 3};
    std::vector<float> first_features{1.0f, 2.0f, 3.0f, 4.0f};
    node_features::set_node_features(
        data, /*max_node_id=*/3,
        first_ids.data(), first_ids.size(),
        first_features.data(), /*feature_dim=*/2);

    {
        const auto snap = node_features::snapshot(data);
        ASSERT_GE(snap.node_features.size(), 4u * 2u);
        EXPECT_FLOAT_EQ(snap.node_features[1 * 2],     1.0f);
        EXPECT_FLOAT_EQ(snap.node_features[1 * 2 + 1], 2.0f);
    }

    std::vector<int> second_ids{5};
    std::vector<float> second_features{9.0f, 10.0f};
    node_features::set_node_features(
        data, /*max_node_id=*/5,
        second_ids.data(), second_ids.size(),
        second_features.data(), 2);

    const auto snap = node_features::snapshot(data);
    EXPECT_FLOAT_EQ(snap.node_features[1 * 2],     1.0f);
    EXPECT_FLOAT_EQ(snap.node_features[1 * 2 + 1], 2.0f);
    EXPECT_FLOAT_EQ(snap.node_features[5 * 2],     9.0f);
    EXPECT_FLOAT_EQ(snap.node_features[5 * 2 + 1], 10.0f);
    EXPECT_EQ(snap.feature_dim, 2u);
}

TEST(NodeFeaturesTest, SetNodeFeaturesWithPointers) {
    core::TemporalRandomWalk trw(/*is_directed=*/true, /*use_gpu=*/false);
    auto& data = trw.data();

    const std::vector<int> node_ids{0, 2};
    const std::vector<float> node_features{7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};

    node_features::set_node_features(
        data, /*max_node_id=*/2,
        node_ids.data(), node_ids.size(),
        node_features.data(), /*feature_dim=*/3);

    const auto snap = node_features::snapshot(data);
    EXPECT_FLOAT_EQ(snap.node_features[2],     9.0f);
    EXPECT_FLOAT_EQ(snap.node_features[2 * 3], 1.0f);
    EXPECT_EQ(snap.feature_dim, 3u);
}
