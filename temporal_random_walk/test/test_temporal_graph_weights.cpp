#include <gtest/gtest.h>

#include "../src/proxies/EdgeDataProxy.cuh"
#include "../src/proxies/NodeEdgeIndexProxy.cuh"
#include "../src/proxies/NodeMappingProxy.cuh"

#include "../src/proxies/TemporalGraphProxy.cuh"
#include "../src/proxies/RandomPickerProxies.cuh"

template<typename T>
class TemporalGraphWeightTest : public ::testing::Test {

protected:
    void SetUp() override {
        // Will be used in multiple tests
        test_edges = {
            Edge{1, 2, 10}, // source, target, timestamp
            Edge{1, 3, 10}, // same timestamp group
            Edge{2, 3, 20},
            Edge{2, 4, 20}, // same timestamp group
            Edge{3, 4, 30},
            Edge{4, 1, 40}
        };
    }

    // Helper to verify cumulative weights are properly normalized
    static void verify_cumulative_weights(const std::vector<double>& weights) {
        ASSERT_FALSE(weights.empty());
        for (size_t i = 0; i < weights.size(); i++) {
            EXPECT_GE(weights[i], 0.0);
            if (i > 0) {
                EXPECT_GE(weights[i], weights[i-1]);
            }
        }
        EXPECT_NEAR(weights.back(), 1.0, 1e-6);
    }

    // Helper to get individual weights from cumulative weights
    static std::vector<double> get_individual_weights(const std::vector<double>& cumulative) {
        std::vector<double> weights;
        weights.push_back(cumulative[0]);
        for (size_t i = 1; i < cumulative.size(); i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    }

    std::vector<Edge> test_edges;
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,  // CPU mode
    std::integral_constant<bool, true>    // GPU mode
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>   // CPU mode only
>;
#endif

TYPED_TEST_SUITE(TemporalGraphWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalGraphWeightTest, EdgeWeightComputation) {
    TemporalGraphProxy graph(/*is_directed=*/false, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(this->test_edges);

    EdgeDataProxy edge_data_proxy(graph.graph->edge_data);

    // Should have 4 timestamp groups (10,20,30,40)
    ASSERT_EQ(edge_data_proxy.forward_cumulative_weights_exponential().size(), 4);
    ASSERT_EQ(edge_data_proxy.backward_cumulative_weights_exponential().size(), 4);

    // Verify cumulative properties
    this->verify_cumulative_weights(edge_data_proxy.forward_cumulative_weights_exponential());
    this->verify_cumulative_weights(edge_data_proxy.backward_cumulative_weights_exponential());

    // Get individual weights
    auto forward_weights = this->get_individual_weights(edge_data_proxy.forward_cumulative_weights_exponential());
    auto backward_weights = this->get_individual_weights(edge_data_proxy.backward_cumulative_weights_exponential());

    // Forward weights should give higher probability to earlier timestamps
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GE(forward_weights[i], forward_weights[i+1])
            << "Forward weight at " << i << " should be >= weight at " << (i+1);
    }

    // Backward weights should give higher probability to later timestamps
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LE(backward_weights[i], backward_weights[i+1])
            << "Backward weight at " << i << " should be <= weight at " << (i+1);
    }
}

TYPED_TEST(TemporalGraphWeightTest, NodeWeightComputation) {
    TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(this->test_edges);

    const auto& node_index = NodeEdgeIndexProxy(graph.graph->node_edge_index);
    const auto& node_mapping = NodeMappingProxy(graph.graph->node_mapping);

    // Check weights for node 2 which has both in/out edges
    constexpr int node_id = 2;
    const int dense_idx = node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    // Get node's group range
    const size_t num_out_groups = node_index.get_timestamp_group_count(dense_idx, true, true);
    ASSERT_GT(num_out_groups, 0);

    // Get outbound group offsets
    const auto host_offsets = node_index.outbound_timestamp_group_offsets();
    const size_t start_pos = host_offsets[dense_idx];
    const size_t end_pos = host_offsets[dense_idx + 1];

    // Get weights for this node's range
    auto host_weights = node_index.outbound_forward_cumulative_weights_exponential();
    const std::vector<double> node_out_weights(
        host_weights.begin() + static_cast<int>(start_pos),
        host_weights.begin() + static_cast<int>(end_pos));

    // Verify weights
    ASSERT_FALSE(node_out_weights.empty());
    for (size_t i = 0; i < node_out_weights.size(); i++) {
        EXPECT_GE(node_out_weights[i], 0.0);
        if (i > 0) {
            EXPECT_GE(node_out_weights[i], node_out_weights[i-1]);
        }
    }
    EXPECT_NEAR(node_out_weights.back(), 1.0, 1e-6);
}

TYPED_TEST(TemporalGraphWeightTest, WeightBasedSampling) {
    TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(this->test_edges);

    // Test forward sampling after timestamp 20
    std::map<int64_t, int> forward_samples;
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(RandomPickerType::ExponentialWeight, 20, true);
        EXPECT_GT(ts, 20) << "Forward sampled timestamp should be > 20";
        forward_samples[ts]++;
    }
    EXPECT_GT(forward_samples[30], forward_samples[40])
        << "Earlier timestamp 30 should be sampled more than 40";

    // Test backward sampling before timestamp 30
    std::map<int64_t, int> backward_samples;
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(RandomPickerType::ExponentialWeight, 30, false);
        EXPECT_LT(ts, 30) << "Backward sampled timestamp should be < 30";
        backward_samples[ts]++;
    }
    EXPECT_GT(backward_samples[20], backward_samples[10])
        << "Later timestamp 30 should be sampled more than 10";

    backward_samples.clear();
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(RandomPickerType::ExponentialWeight, 50, false);
        backward_samples[ts]++;
    }
    EXPECT_GT(backward_samples[40], backward_samples[30])
        << "Later timestamp 40 should be sampled more than 30";
}

TYPED_TEST(TemporalGraphWeightTest, EdgeCases) {
    // Empty graph test
    {
        const TemporalGraphProxy empty_graph(false, TypeParam::value, -1, true);

        EXPECT_EQ(empty_graph.graph->edge_data->forward_cumulative_weights_exponential_size, 0);
        EXPECT_EQ(empty_graph.graph->edge_data->backward_cumulative_weights_exponential_size, 0);
    }

    // Single edge graph test
    {
        TemporalGraphProxy single_edge_graph(false, TypeParam::value, -1, true);
        single_edge_graph.add_multiple_edges({Edge{1, 2, 10}});

        EXPECT_EQ(single_edge_graph.graph->edge_data->forward_cumulative_weights_exponential_size, 1);
        EXPECT_EQ(single_edge_graph.graph->edge_data->backward_cumulative_weights_exponential_size, 1);

        EdgeDataProxy edge_data_proxy = EdgeDataProxy(single_edge_graph.graph->edge_data);
        EXPECT_NEAR(edge_data_proxy.forward_cumulative_weights_exponential()[0], 1.0, 1e-6);
        EXPECT_NEAR(edge_data_proxy.backward_cumulative_weights_exponential()[0], 1.0, 1e-6);
    }
}

TYPED_TEST(TemporalGraphWeightTest, TimescaleBoundZero) {
    TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, 0);
    graph.add_multiple_edges(this->test_edges);

    // Should behave like -1 (unscaled)
    EdgeDataProxy edge_data_proxy = EdgeDataProxy(graph.graph->edge_data);
    this->verify_cumulative_weights(edge_data_proxy.forward_cumulative_weights_exponential());
    this->verify_cumulative_weights(edge_data_proxy.backward_cumulative_weights_exponential());
}

TYPED_TEST(TemporalGraphWeightTest, TimescaleBoundSampling) {
    TemporalGraphProxy scaled_graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, 10.0);
    TemporalGraphProxy unscaled_graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, -1);

    scaled_graph.add_multiple_edges(this->test_edges);
    unscaled_graph.add_multiple_edges(this->test_edges);

    // Sample edges and verify temporal bias is preserved
    std::map<int64_t, int> scaled_samples, unscaled_samples;
    constexpr int num_samples = 1000;

    // Forward sampling
    for (int i = 0; i < num_samples; i++) {
        auto [u1, i1, ts1] = scaled_graph.get_edge_at(RandomPickerType::ExponentialWeight, 20, true);
        auto [u2, i2, ts2] = unscaled_graph.get_edge_at(RandomPickerType::ExponentialWeight, 20, true);
        scaled_samples[ts1]++;
        unscaled_samples[ts2]++;
    }

    // Both should maintain same ordering of preferences
    EXPECT_GT(scaled_samples[30], scaled_samples[40])
        << "Scaled sampling should prefer earlier timestamp";
    EXPECT_GT(unscaled_samples[30], unscaled_samples[40])
        << "Unscaled sampling should prefer earlier timestamp";
}

TYPED_TEST(TemporalGraphWeightTest, WeightScalingPrecision) {
    TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, 2.0);

    // Use exact timestamps for precise validation
    graph.add_multiple_edges({
        Edge{1, 2, 100},
        Edge{1, 3, 200},
        Edge{1, 4, 300}
    });

    // Get individual weights using helper
    EdgeDataProxy edge_data_proxy = EdgeDataProxy(graph.graph->edge_data);
    const auto weights = this->get_individual_weights(edge_data_proxy.forward_cumulative_weights_exponential());
    ASSERT_EQ(weights.size(), 3);

    // Time range is 200, scale = 2.0/200 = 0.01
    constexpr double time_scale = 2.0 / 200.0;

    // Check weight ratios
    for (size_t i = 0; i < weights.size() - 1; i++) {
        constexpr auto time_diff = 100.0;  // Fixed time difference
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = weights[i+1] / weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Weight ratio incorrect at index " << i;
    }

    auto node_mapping = NodeMappingProxy(graph.graph->node_mapping);

    // Check node weights
    constexpr int node_id = 1;
    const int dense_idx = node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    // Get node's group range
    auto node_edge_index = NodeEdgeIndexProxy(graph.graph->node_edge_index);
    const auto host_offsets = node_edge_index.outbound_timestamp_group_offsets();
    const size_t start = host_offsets[dense_idx];

    // Get node's weights
    const auto node_individual_weights = this->get_individual_weights(
        node_edge_index.outbound_forward_cumulative_weights_exponential());

    // Check for segment from start to end
    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(node_individual_weights[start + i], weights[i], 1e-6)
            << "Node weight mismatch at index " << i;
    }
}

TYPED_TEST(TemporalGraphWeightTest, DifferentTimescaleBounds) {
    const std::vector<double> bounds = {2.0, 5.0, 10.};

    for (double bound : bounds) {
        constexpr int num_samples = 10000;

        TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, bound);
        graph.add_multiple_edges(this->test_edges);

        std::map<int64_t, int> samples;
        for (int i = 0; i < num_samples; i++) {
            auto [u1, i1, ts] = graph.get_edge_at(RandomPickerType::ExponentialWeight, -1, true);
            samples[ts]++;
        }

        // Compare consecutive timestamps
        std::vector<int64_t> timestamps = {10, 20, 30, 40};
        for (size_t i = 0; i < timestamps.size() - 1; i++) {
            int count_curr = samples[timestamps[i]];
            int count_next = samples[timestamps[i + 1]];

            EXPECT_GT(count_curr, count_next)
                << "At bound " << bound
                << ", timestamp " << timestamps[i] << " (" << count_curr << " samples) vs "
                << timestamps[i+1] << " (" << count_next << " samples)";
        }
    }
}

TYPED_TEST(TemporalGraphWeightTest, SingleTimestampWithBounds) {
    const std::vector<Edge> single_ts_edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 4, 100}
    };

    // Test with different bounds
    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        TemporalGraphProxy graph(/*is_directed=*/true, /*use_gpu=*/TypeParam::value, /*max_time_capacity=*/-1, /*enable_weight_computation=*/true, bound);
        graph.add_multiple_edges(single_ts_edges);

        auto edge_data_proxy = EdgeDataProxy(graph.graph->edge_data);

        ASSERT_EQ(edge_data_proxy.forward_cumulative_weights_exponential().size(), 1);
        ASSERT_EQ(edge_data_proxy.backward_cumulative_weights_exponential().size(), 1);
        EXPECT_NEAR(edge_data_proxy.forward_cumulative_weights_exponential()[0], 1.0, 1e-6);
        EXPECT_NEAR(edge_data_proxy.backward_cumulative_weights_exponential()[0], 1.0, 1e-6);
    }
}
