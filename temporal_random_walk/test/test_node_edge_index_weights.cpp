#include <gtest/gtest.h>

#include "../src/proxies/NodeEdgeIndexProxy.cuh"
#include "../src/proxies/EdgeDataProxy.cuh"
#include "../src/proxies/NodeMappingProxy.cuh"

template<typename T>
class NodeEdgeIndexWeightTest : public ::testing::Test {
protected:

    NodeEdgeIndexWeightTest(): index(T::value) {}

    // Helper to verify weights are normalized and cumulative per node's group
    static void verify_node_weights(const std::vector<size_t>& group_offsets,
                                  const std::vector<double>& weights)
    {
        // For each node
        for (size_t node = 0; node < group_offsets.size() - 1; node++) {
            size_t start = group_offsets[node];
            size_t end = group_offsets[node + 1];

            if (start < end) {
                // Check weights are monotonically increasing
                for (size_t i = start; i < end; i++) {
                    EXPECT_GE(weights[i], 0.0);
                    if (i > start) {
                        EXPECT_GE(weights[i], weights[i-1]);
                    }
                }
                // Last weight in node's group should be normalized to 1.0
                EXPECT_NEAR(weights[end-1], 1.0, 1e-6);
            }
        }
    }

    // Helper to get individual weights for a node
    static std::vector<double> get_individual_weights(
        const std::vector<double> &cumulative,
        const std::vector<size_t> &offsets,
        const size_t node) {

        // Get node's range
        const size_t start = offsets[node];
        const size_t end = offsets[node + 1];

        // Calculate individual weights
        std::vector<double> weights;
        weights.push_back(cumulative[start]);
        for (size_t i = start; i < end; i++) {
            weights.push_back(cumulative[i] - cumulative[i - 1]);
        }

        return weights;
    }

    void setup_test_graph(bool directed = true) {
        const EdgeDataProxy edges(T::value);  // CPU mode
        // Add edges that create multiple timestamp groups per node
        edges.push_back(1, 2, 10);
        edges.push_back(1, 3, 10); // Same timestamp group for node 1
        edges.push_back(1, 4, 20); // New timestamp group for node 1
        edges.push_back(2, 3, 20);
        edges.push_back(2, 4, 30); // Different timestamp groups for node 2
        edges.push_back(3, 4, 40);
        edges.update_timestamp_groups();

        const NodeMappingProxy mapping(T::value);  // CPU mode
        mapping.update(edges.edge_data, 0, edges.size());

        index = NodeEdgeIndexProxy(T::value);  // CPU mode
        index.rebuild(edges.edge_data, mapping.node_mapping, directed);
        index.update_temporal_weights(edges.edge_data, -1);
    }

    NodeEdgeIndexProxy index;
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

TYPED_TEST_SUITE(NodeEdgeIndexWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(NodeEdgeIndexWeightTest, EmptyGraph) {
    EdgeDataProxy empty_edges(TypeParam::value);
    NodeMappingProxy empty_mapping(TypeParam::value);
    this->index = NodeEdgeIndexProxy(TypeParam::value);
    this->index.rebuild(empty_edges.edge_data, empty_mapping.node_mapping, true);
    this->index.update_temporal_weights(empty_edges.edge_data, -1);

    EXPECT_TRUE(this->index.outbound_forward_cumulative_weights_exponential().empty());
    EXPECT_TRUE(this->index.outbound_backward_cumulative_weights_exponential().empty());
    EXPECT_TRUE(this->index.inbound_backward_cumulative_weights_exponential().empty());
}

TYPED_TEST(NodeEdgeIndexWeightTest, DirectedWeightNormalization) {
    this->setup_test_graph(true);

    // Verify per-node weight normalization
    this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
        this->index.outbound_forward_cumulative_weights_exponential());
    this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
        this->index.outbound_backward_cumulative_weights_exponential());
    this->verify_node_weights(this->index.inbound_timestamp_group_offsets(),
        this->index.inbound_backward_cumulative_weights_exponential());
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightBiasPerNode) {

    EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 100);  // Known timestamps for precise verification
    edges.push_back(1, 3, 200);
    edges.push_back(1, 4, 300);
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);
    mapping.update(edges.edge_data, 0, edges.size());

    this->index = NodeEdgeIndexProxy(TypeParam::value);
    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);
    this->index.update_temporal_weights(edges.edge_data, -1); // No scaling

    // Forward weights: exp(-(t - t_min))
    auto forward = this->get_individual_weights(
        this->index.outbound_forward_cumulative_weights_exponential(),
        this->index.outbound_timestamp_group_offsets(), 1);

    for (size_t i = 0; i < forward.size() - 1; i++) {
        double expected_ratio = exp(-100); // Time diff between groups is 100
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    // Backward weights: exp(t - t_min)
    auto backward = this->get_individual_weights(
        this->index.outbound_backward_cumulative_weights_exponential(),
        this->index.outbound_timestamp_group_offsets(), 1);

    for (size_t i = 0; i < backward.size() - 1; i++) {
        double expected_ratio = exp(100); // Time diff between groups is 100
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, ScaledWeightRatios) {
    EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 100);
    edges.push_back(1, 3, 300);
    edges.push_back(1, 4, 500);
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);
    mapping.update(edges.edge_data, 0, edges.size());

    this->index = NodeEdgeIndexProxy(TypeParam::value);
    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);

    constexpr double timescale_bound = 2.0;
    this->index.update_temporal_weights(edges.edge_data, timescale_bound);

    const auto forward = this->get_individual_weights(
        this->index.outbound_forward_cumulative_weights_exponential(),
        this->index.outbound_timestamp_group_offsets(), 1);

    // Time range is 400, scale = 2.0/400 = 0.005
    constexpr double time_scale = timescale_bound / 400.0;

    for (size_t i = 0; i < forward.size() - 1; i++) {
        // Each step is 200 units
        double scaled_diff = 200 * time_scale;
        double expected_ratio = exp(-scaled_diff);
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6)
            << "Forward ratio incorrect at index " << i;
    }

    auto backward = this->get_individual_weights(
        this->index.outbound_backward_cumulative_weights_exponential(),
        this->index.outbound_timestamp_group_offsets(), 1);

    for (size_t i = 0; i < backward.size() - 1; i++) {
        constexpr double scaled_diff = 200 * time_scale;
        const double expected_ratio = exp(scaled_diff);
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6)
            << "Backward ratio incorrect at index " << i;
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, UndirectedWeightNormalization) {
   this->setup_test_graph(false);

   // For undirected, should only have outbound weights
   this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
                      this->index.outbound_forward_cumulative_weights_exponential());
   this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
                      this->index.outbound_backward_cumulative_weights_exponential());
   EXPECT_TRUE(this->index.inbound_backward_cumulative_weights_exponential().empty());
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightConsistencyAcrossUpdates) {
   this->setup_test_graph(true);

   // Store original weights
   auto original_out_forward = this->index.outbound_forward_cumulative_weights_exponential();
   auto original_out_backward = this->index.outbound_backward_cumulative_weights_exponential();
   auto original_in_backward = this->index.inbound_backward_cumulative_weights_exponential();

   // Rebuild and update weights again
   EdgeDataProxy edges(TypeParam::value);
   edges.push_back(1, 2, 10);
   edges.push_back(1, 3, 10);
   edges.update_timestamp_groups();

   NodeMappingProxy mapping(TypeParam::value);
   mapping.update(edges.edge_data, 0, edges.size());

   this->index.rebuild(edges.edge_data, mapping.node_mapping, true);
   this->index.update_temporal_weights(edges.edge_data, -1);

   // Weights should be different but still normalized
   EXPECT_NE(original_out_forward.size(), this->index.outbound_forward_cumulative_weights_exponential().size());
   this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
                      this->index.outbound_forward_cumulative_weights_exponential());
}

TYPED_TEST(NodeEdgeIndexWeightTest, SingleTimestampGroupPerNode) {
   EdgeDataProxy edges(TypeParam::value);
   // All edges in same timestamp group
   edges.push_back(1, 2, 10);
   edges.push_back(1, 3, 10);
   edges.push_back(2, 3, 10);
   edges.update_timestamp_groups();

   NodeMappingProxy mapping(TypeParam::value);
   mapping.update(edges.edge_data, 0, edges.size());

   this->index.rebuild(edges.edge_data, mapping.node_mapping, true);
   this->index.update_temporal_weights(edges.edge_data, -1);

   // Each node should have single weight of 1.0
   for (size_t node = 0; node < this->index.outbound_timestamp_group_offsets().size() - 1; node++) {
       const size_t start = this->index.outbound_timestamp_group_offsets()[node];
       size_t end = this->index.outbound_timestamp_group_offsets()[node + 1];
       if (start < end) {
           EXPECT_EQ(end - start, 1);
           EXPECT_NEAR(this->index.outbound_forward_cumulative_weights_exponential()[start], 1.0, 1e-6);
           EXPECT_NEAR(this->index.outbound_backward_cumulative_weights_exponential()[start], 1.0, 1e-6);
       }
   }
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleBoundZero) {
    EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);
    mapping.update(edges.edge_data, 0, edges.size());

    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);
    this->index.update_temporal_weights(edges.edge_data, 0);  // Should behave like -1

    this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
                       this->index.outbound_forward_cumulative_weights_exponential());
    this->verify_node_weights(this->index.outbound_timestamp_group_offsets(),
                       this->index.outbound_backward_cumulative_weights_exponential());
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleBoundWithSingleTimestamp) {
    EdgeDataProxy edges(TypeParam::value);
    // All edges for node 1 have same timestamp
    constexpr int node_id = 1;  // Original node ID
    edges.push_back(node_id, 2, 10);
    edges.push_back(node_id, 3, 10);
    edges.push_back(node_id, 4, 10);
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);
    mapping.update(edges.edge_data, 0, edges.size());

    // Get the dense index for node_id
    const int dense_idx = mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0) << "Node " << node_id << " not found in mapping";

    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);

    // Test with different bounds
    for (const double bound : {-1.0, 0.0, 10.0, 50.0}) {
        this->index.update_temporal_weights(edges.edge_data, bound);

        // Node should have single group with weight 1.0
        const size_t start = this->index.outbound_timestamp_group_offsets()[dense_idx];
        const size_t end = this->index.outbound_timestamp_group_offsets()[dense_idx + 1];
        ASSERT_EQ(end - start, 1) << "Node should have exactly one timestamp group";
        EXPECT_NEAR(this->index.outbound_forward_cumulative_weights_exponential()[start], 1.0, 1e-6);
        EXPECT_NEAR(this->index.outbound_backward_cumulative_weights_exponential()[start], 1.0, 1e-6);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightOrderPreservation) {
    EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);
    mapping.update(edges.edge_data, 0, edges.size());
    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);

    // Get unscaled weights
    this->index.update_temporal_weights(edges.edge_data, -1);
    const auto unscaled_forward = this->index.outbound_forward_cumulative_weights_exponential();
    const auto unscaled_backward = this->index.outbound_backward_cumulative_weights_exponential();

    // Get scaled weights
    this->index.update_temporal_weights(edges.edge_data, 10.0);

    // Check relative ordering is preserved for node 1
    const size_t start = this->index.outbound_timestamp_group_offsets()[1];
    const size_t end = this->index.outbound_timestamp_group_offsets()[2];
    for (size_t i = start + 1; i < end; i++) {
        // If unscaled weights were increasing/decreasing, scaled weights should follow
        EXPECT_EQ(unscaled_forward[i] > unscaled_forward[i-1],
                  this->index.outbound_forward_cumulative_weights_exponential()[i] > this->index.outbound_forward_cumulative_weights_exponential()[i-1]);
        EXPECT_EQ(unscaled_backward[i] > unscaled_backward[i-1],
                  this->index.outbound_backward_cumulative_weights_exponential()[i] > this->index.outbound_backward_cumulative_weights_exponential()[i-1]);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleNormalizationTest) {

    EdgeDataProxy edges(TypeParam::value);  // CPU mode
    // Create edges with widely varying time differences
    edges.push_back(1, 2, 100);       // Small gap
    edges.push_back(1, 3, 200);       // 100 units
    edges.push_back(1, 4, 1000);      // 800 units
    edges.push_back(1, 5, 100000);    // Large gap
    edges.update_timestamp_groups();

    NodeMappingProxy mapping(TypeParam::value);  // CPU mode
    mapping.update(edges.edge_data, 0, edges.size());

    this->index = NodeEdgeIndexProxy(TypeParam::value);  // CPU mode
    this->index.rebuild(edges.edge_data, mapping.node_mapping, true);

    constexpr double timescale_bound = 5.0;
    this->index.update_temporal_weights(edges.edge_data, timescale_bound);

    // Get host data for group offsets and start/end indices
    const auto host_offsets = this->index.outbound_timestamp_group_offsets();

    // Get individual weights using helper method
    const auto weights = this->get_individual_weights(
        this->index.outbound_forward_cumulative_weights_exponential(),
        this->index.outbound_timestamp_group_offsets(),
        1  // Node index
    );

    // Check that max weight difference is bounded by timescale_bound
    double max_weight_ratio = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights.size(); j++) {
            if (weights[j] > 0) {
                max_weight_ratio = std::max(max_weight_ratio, log(weights[i] / weights[j]));
            }
        }
    }
    EXPECT_LE(max_weight_ratio, timescale_bound)
        << "Maximum weight ratio exceeds timescale bound";

    // Get timestamps for node from EdgeData
    const auto host_timestamps = edges.unique_timestamps();

    // Verify that relative ordering matches time differences
    for (size_t i = 0; i < weights.size() - 1; i++) {
        const double time_ratio = static_cast<double>(host_timestamps[i+1] - host_timestamps[i]) /
                          static_cast<double>(host_timestamps[weights.size()-1] - host_timestamps[0]);
        double weight_ratio = weights[i] / weights[i+1];
        EXPECT_NEAR(log(weight_ratio), timescale_bound * time_ratio, 1e-6)
            << "Weight ratio doesn't match expected scaled time difference at " << i;
    }
}