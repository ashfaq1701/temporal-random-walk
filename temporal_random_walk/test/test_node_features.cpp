#include <gtest/gtest.h>

#include <vector>

#include "../src/proxies/NodeFeatures.cuh"
#include "../src/graph/edge_data.cuh"

TEST(NodeFeaturesTest, SetNodeFeaturesPreservesOldDataOnGrow) {
    EdgeDataStore edge_data(false, false, false);
    edge_data.max_node_id = 3;

    NodeFeatures nf;

    std::vector<int> first_ids{1, 3};
    std::vector<float> first_features{1.0f, 2.0f, 3.0f, 4.0f};
    nf.set_node_features(edge_data.max_node_id, first_ids.data(), first_ids.size(), first_features.data(), 2);

    const auto* store = nf.get_node_features();
    EXPECT_FLOAT_EQ(store->node_features[1 * 2], 1.0f);
    EXPECT_FLOAT_EQ(store->node_features[1 * 2 + 1], 2.0f);

    edge_data.max_node_id = 5;
    std::vector<int> second_ids{5};
    std::vector<float> second_features{9.0f, 10.0f};
    nf.set_node_features(edge_data.max_node_id, second_ids.data(), second_ids.size(), second_features.data(), 2);

    EXPECT_FLOAT_EQ(store->node_features[1 * 2], 1.0f);
    EXPECT_FLOAT_EQ(store->node_features[1 * 2 + 1], 2.0f);
    EXPECT_FLOAT_EQ(store->node_features[5 * 2], 9.0f);
    EXPECT_FLOAT_EQ(store->node_features[5 * 2 + 1], 10.0f);

    EXPECT_EQ(nf.max_node_id(), 5);
    EXPECT_EQ(nf.node_feature_dim(), 2);
}

TEST(NodeFeaturesTest, SetNodeFeaturesWithPointers) {
    EdgeDataStore edge_data(false, false, false);
    edge_data.max_node_id = 2;

    const NodeFeatures nf;

    const std::vector<int> node_ids{0, 2};
    const std::vector<float> node_features{7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};

    nf.set_node_features(edge_data.max_node_id, node_ids.data(), node_ids.size(), node_features.data(), 3);

    const auto* store = nf.get_node_features();
    EXPECT_FLOAT_EQ(store->node_features[2], 9.0f);
    EXPECT_FLOAT_EQ(store->node_features[2 * 3], 1.0f);
    EXPECT_EQ(nf.node_feature_dim(), 3);
}
