#include <gtest/gtest.h>
#include "../src/proxies/EdgeDataProxy.cuh"
#include <cmath>

template<typename T>
class EdgeDataWeightTest : public ::testing::Test {

protected:
    static void verify_cumulative_weights(const std::vector<double> &weights) {
        ASSERT_FALSE(weights.empty());
        for (size_t i = 0; i < weights.size(); i++) {
            EXPECT_GE(weights[i], 0.0);
            if (i > 0) {
                EXPECT_GE(weights[i], weights[i-1]);
            }
        }
        EXPECT_NEAR(weights.back(), 1.0, 1e-6);
    }

    static void add_test_edges(const EdgeDataProxy& edges) {
        edges.push_back(1, 2, 10);
        edges.push_back(1, 3, 10);
        edges.push_back(2, 3, 20);
        edges.push_back(2, 4, 20);
        edges.push_back(3, 4, 30);
        edges.push_back(4, 1, 40);
        edges.update_timestamp_groups();
    }

    static std::vector<double> get_individual_weights(const std::vector<double>& cumulative) {
        std::vector<double> weights;
        weights.reserve(cumulative.size());
        weights.push_back(cumulative[0]);
        for (size_t i = 1; i < cumulative.size(); i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>   // CPU mode only
>;
#endif

TYPED_TEST_SUITE(EdgeDataWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(EdgeDataWeightTest, SingleTimestampGroup) {
    const EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 10);
    edges.push_back(2, 3, 10);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    ASSERT_EQ(edges.forward_cumulative_weights_exponential().size(), 1);
    ASSERT_EQ(edges.backward_cumulative_weights_exponential().size(), 1);

    // Single group should have normalized weight of 1.0
    EXPECT_NEAR(edges.forward_cumulative_weights_exponential()[0], 1.0, 1e-6);
    EXPECT_NEAR(edges.backward_cumulative_weights_exponential()[0], 1.0, 1e-6);
}

TYPED_TEST(EdgeDataWeightTest, WeightNormalization) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Should have 4 timestamp groups (10,20,30,40)
    ASSERT_EQ(edges.forward_cumulative_weights_exponential().size(), 4);
    ASSERT_EQ(edges.backward_cumulative_weights_exponential().size(), 4);

    this->verify_cumulative_weights(edges.forward_cumulative_weights_exponential());
    this->verify_cumulative_weights(edges.backward_cumulative_weights_exponential());
}

TYPED_TEST(EdgeDataWeightTest, ForwardWeightBias) {
    EdgeDataProxy edges(TypeParam::value);

    this->add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Forward weights should be higher for earlier timestamps
    const std::vector<double> forward_weights = this->get_individual_weights(edges.forward_cumulative_weights_exponential());

    // Earlier groups should have higher weights
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GT(forward_weights[i], forward_weights[i+1])
            << "Forward weight at index " << i << " should be greater than weight at " << i+1;
    }
}

TYPED_TEST(EdgeDataWeightTest, BackwardWeightBias) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Backward weights should be higher for later timestamps
    const std::vector<double> backward_weights = this->get_individual_weights(edges.backward_cumulative_weights_exponential());

    // Later groups should have higher weights
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LT(backward_weights[i], backward_weights[i+1])
            << "Backward weight at index " << i << " should be less than weight at " << i+1;
    }
}

TYPED_TEST(EdgeDataWeightTest, WeightExponentialDecay) {
    const EdgeDataProxy edges(TypeParam::value);

    edges.push_back(1, 2, 10);
    edges.push_back(2, 3, 20);
    edges.push_back(3, 4, 30);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    const auto forward_weights = this->get_individual_weights(edges.forward_cumulative_weights_exponential());
    const auto backward_weights = this->get_individual_weights(edges.backward_cumulative_weights_exponential());

    // For forward weights: log(w[i+1]/w[i]) = -Δt
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        const auto time_diff = edges.unique_timestamps()[i+1] - edges.unique_timestamps()[i];
        if (forward_weights[i+1] > 0 && forward_weights[i] > 0) {
            const double log_ratio = log(forward_weights[i+1]/forward_weights[i]);
            EXPECT_NEAR(log_ratio, -time_diff, 1e-6)
                << "Forward weight log ratio incorrect at index " << i;
        }
    }

    // For backward weights: log(w[i+1]/w[i]) = Δt
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        const auto time_diff = edges.unique_timestamps()[i+1] - edges.unique_timestamps()[i];
        if (backward_weights[i+1] > 0 && backward_weights[i] > 0) {
            const double log_ratio = log(backward_weights[i+1]/backward_weights[i]);
            EXPECT_NEAR(log_ratio, time_diff, 1e-6)
                << "Backward weight log ratio incorrect at index " << i;
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, UpdateWeights) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Store original weights
    const auto original_forward = edges.forward_cumulative_weights_exponential();
    const auto original_backward = edges.backward_cumulative_weights_exponential();

    // Add new edge with different timestamp
    edges.push_back(1, 4, 50);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    // Weights should be different after update
    EXPECT_NE(original_forward.size(), edges.forward_cumulative_weights_exponential().size());
    EXPECT_NE(original_backward.size(), edges.backward_cumulative_weights_exponential().size());

    // But should still maintain normalization
    this->verify_cumulative_weights(edges.forward_cumulative_weights_exponential());
    this->verify_cumulative_weights(edges.backward_cumulative_weights_exponential());
}

TYPED_TEST(EdgeDataWeightTest, TimescaleBoundZero) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);
    edges.update_temporal_weights(0);  // Should behave like -1

    this->verify_cumulative_weights(edges.forward_cumulative_weights_exponential());
    this->verify_cumulative_weights(edges.backward_cumulative_weights_exponential());
}

TYPED_TEST(EdgeDataWeightTest, TimescaleBoundPositive) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);
    constexpr double timescale_bound = 30.0;
    edges.update_temporal_weights(timescale_bound);

    // Check relative weights instead of absolute values
    std::vector<double> forward_weights;
    forward_weights.push_back(edges.forward_cumulative_weights_exponential()[0]);
    for (size_t i = 1; i < edges.forward_cumulative_weights_exponential().size(); i++) {
        forward_weights.push_back(
            edges.forward_cumulative_weights_exponential()[i] - edges.forward_cumulative_weights_exponential()[i-1]);
    }

    // Earlier timestamps should have higher weights for forward
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GT(forward_weights[i], forward_weights[i+1]);
    }

    // Later timestamps should have higher weights for backward
    std::vector<double> backward_weights;
    backward_weights.push_back(edges.backward_cumulative_weights_exponential()[0]);
    for (size_t i = 1; i < edges.backward_cumulative_weights_exponential().size(); i++) {
        backward_weights.push_back(
            edges.backward_cumulative_weights_exponential()[i] - edges.backward_cumulative_weights_exponential()[i-1]);
    }

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LT(backward_weights[i], backward_weights[i+1]);
    }
}

TYPED_TEST(EdgeDataWeightTest, ScalingComparison) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);

    // Test relative weight proportions are preserved
    std::vector<double> weights_unscaled, weights_scaled;

    edges.update_temporal_weights(-1);
    for (size_t i = 1; i < edges.forward_cumulative_weights_exponential().size(); i++) {
        weights_unscaled.push_back(
            edges.forward_cumulative_weights_exponential()[i] / edges.forward_cumulative_weights_exponential()[i-1]);
    }

    edges.update_temporal_weights(50.0);
    for (size_t i = 1; i < edges.forward_cumulative_weights_exponential().size(); i++) {
        weights_scaled.push_back(
            edges.forward_cumulative_weights_exponential()[i] / edges.forward_cumulative_weights_exponential()[i-1]);
    }

    // Compare ratios with some tolerance
    for (size_t i = 0; i < weights_unscaled.size(); i++) {
        EXPECT_NEAR(weights_scaled[i], weights_unscaled[i], 1e-2);
    }
}

TYPED_TEST(EdgeDataWeightTest, ScaledWeightBounds) {
    const EdgeDataProxy edges(TypeParam::value);
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    // Get weights using the helper method defined in the test fixture
    const auto forward_weights = this->get_individual_weights(edges.forward_cumulative_weights_exponential());
    const auto backward_weights = this->get_individual_weights(edges.backward_cumulative_weights_exponential());

    // Maximum log ratio should not exceed timescale_bound
    for (size_t i = 0; i < forward_weights.size(); i++) {
        for (size_t j = 0; j < forward_weights.size(); j++) {
            if (forward_weights[j] > 0 && forward_weights[i] > 0) {  // Add protection against zero weights
                const double log_ratio = log(forward_weights[i] / forward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6)
                    << "Forward weights ratio exceeded bound at i=" << i << ", j=" << j;
            }
        }
    }

    for (size_t i = 0; i < backward_weights.size(); i++) {
        for (size_t j = 0; j < backward_weights.size(); j++) {
            if (backward_weights[j] > 0 && backward_weights[i] > 0) {  // Add protection against zero weights
                const double log_ratio = log(backward_weights[i] / backward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6)
                    << "Backward weights ratio exceeded bound at i=" << i << ", j=" << j;
            }
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, DifferentTimescaleBounds) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);

    std::vector<double> bounds = {5.0, 10.0, 20.0};
    std::vector<std::vector<double>> scaled_ratios;

    // Collect weight ratios for different bounds
    for (const double bound : bounds) {
        edges.update_temporal_weights(bound);
        std::vector<double> ratios;
        for (size_t i = 1; i < edges.forward_cumulative_weights_exponential().size(); i++) {
            ratios.push_back(edges.forward_cumulative_weights_exponential()[i] /
                           edges.forward_cumulative_weights_exponential()[i-1]);
        }
        scaled_ratios.push_back(ratios);
    }

    // Relative ordering should be preserved across different bounds
    for (size_t i = 0; i < scaled_ratios[0].size(); i++) {
        for (size_t j = 1; j < scaled_ratios.size(); j++) {
            EXPECT_EQ(scaled_ratios[0][i] > 1.0, scaled_ratios[j][i] > 1.0)
                << "Weight ratio ordering should be consistent across different bounds";
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, SingleTimestampWithBounds) {
    const EdgeDataProxy edges(TypeParam::value);
    // All edges have same timestamp
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 100);
    edges.push_back(3, 4, 100);
    edges.update_timestamp_groups();

    // Test with different bounds
    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        edges.update_temporal_weights(bound);
        ASSERT_EQ(edges.forward_cumulative_weights_exponential().size(), 1);
        ASSERT_EQ(edges.backward_cumulative_weights_exponential().size(), 1);
        EXPECT_NEAR(edges.forward_cumulative_weights_exponential()[0], 1.0, 1e-6);
        EXPECT_NEAR(edges.backward_cumulative_weights_exponential()[0], 1.0, 1e-6);
    }
}

TYPED_TEST(EdgeDataWeightTest, WeightMonotonicity) {
    EdgeDataProxy edges(TypeParam::value);
    this->add_test_edges(edges);

    const double timescale_bound = 20.0;
    edges.update_temporal_weights(timescale_bound);

    // Forward weights should decrease monotonically
    for (size_t i = 1; i < edges.forward_cumulative_weights_exponential().size(); i++) {
        double prev_weight = i == 1 ? edges.forward_cumulative_weights_exponential()[0]
                                  : edges.forward_cumulative_weights_exponential()[i-1];
        double curr_weight = edges.forward_cumulative_weights_exponential()[i];
        double weight_diff = curr_weight - prev_weight;

        if (i > 1) {
            double prev_diff = edges.forward_cumulative_weights_exponential()[i-1] -
                             edges.forward_cumulative_weights_exponential()[i-2];
            EXPECT_GE(prev_diff, weight_diff)
                << "Forward weight differences should decrease monotonically";
        }
    }

    // Backward weights should increase monotonically
    for (size_t i = 1; i < edges.backward_cumulative_weights_exponential().size(); i++) {
        double prev_weight = i == 1 ? edges.backward_cumulative_weights_exponential()[0]
                                  : edges.backward_cumulative_weights_exponential()[i-1];
        double curr_weight = edges.backward_cumulative_weights_exponential()[i];
        double weight_diff = curr_weight - prev_weight;

        if (i > 1) {
            double prev_diff = edges.backward_cumulative_weights_exponential()[i-1] -
                             edges.backward_cumulative_weights_exponential()[i-2];
            EXPECT_LE(prev_diff, weight_diff)
                << "Backward weight differences should increase monotonically";
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, TimescaleScalingPrecision) {
    const EdgeDataProxy edges(TypeParam::value);
    // Use precise timestamps for exact validation
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    const auto forward_weights = this->get_individual_weights(edges.forward_cumulative_weights_exponential());
    const auto backward_weights = this->get_individual_weights(edges.backward_cumulative_weights_exponential());

    // Time range is 600, scale = 2.0/600
    constexpr double time_scale = timescale_bound / 600.0;

    // Check forward weights
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        const auto time_diff = static_cast<double>(
            edges.unique_timestamps()[i+1] - edges.unique_timestamps()[i]);
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = forward_weights[i+1] / forward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Forward weight ratio incorrect at index " << i;
    }

    // Check backward weights
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        const auto time_diff = static_cast<double>(
            edges.unique_timestamps()[i+1] - edges.unique_timestamps()[i]);
        const double expected_ratio = exp(time_diff * time_scale);
        const double actual_ratio = backward_weights[i+1] / backward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Backward weight ratio incorrect at index " << i;
    }
}
