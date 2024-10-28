#include <gtest/gtest.h>

#include "test_utils.h"
#include "../src/core/TemporalWalk.h"
#include "../src/random/ExponentialRandomPicker.h"
#include "../src/random/LinearRandomPicker.h"

constexpr int TEST_NODE_ID = 45965;
constexpr int LEN_WALK = 20;
constexpr int NUM_WALKS = 1000;

constexpr int RANDOM_START = 0;
constexpr int RANDOM_END = 10000;
constexpr int RANDOM_NUM_SAMPLES = 100000;

class RandomPickerTest : public ::testing::Test {
protected:

    LinearRandomPicker linear_picker;
    ExponentialRandomPicker exp_picker;

    double compute_average_picks(const bool use_exponential, const bool prioritize_end) {
        double sum = 0;
        for (int i = 0; i < RANDOM_NUM_SAMPLES; i++) {
            const int pick = use_exponential ?
                                 exp_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end) :
                                 linear_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end);
            sum += pick;
        }
        return sum / RANDOM_NUM_SAMPLES;
    }
};

class EmptyTemporalWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledTemporalWalkTest : public ::testing::Test {
protected:
    FilledTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<EdgeInfo> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

// Test that prioritize_end=true gives higher average than prioritize_end=false for both pickers
TEST_F(RandomPickerTest, PrioritizeEndGivesHigherAverage) {
    // For Linear Picker
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double linear_start_prioritized = compute_average_picks(false, false);
    EXPECT_GT(linear_end_prioritized, linear_start_prioritized)
        << "Linear picker with prioritize_end=true should give higher average ("
        << linear_end_prioritized << ") than prioritize_end=false ("
        << linear_start_prioritized << ")";

    // For Exponential Picker
    const double exp_end_prioritized = compute_average_picks(true, true);
    const double exp_start_prioritized = compute_average_picks(true, false);
    EXPECT_GT(exp_end_prioritized, exp_start_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than prioritize_end=false ("
        << exp_start_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing end
TEST_F(RandomPickerTest, ExponentialMoreExtremeForEnd) {
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double exp_end_prioritized = compute_average_picks(true, true);

    EXPECT_GT(exp_end_prioritized, linear_end_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than Linear picker ("
        << linear_end_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing start
TEST_F(RandomPickerTest, ExponentialMoreExtremeForStart) {
    const double linear_start_prioritized = compute_average_picks(false, false);
    const double exp_start_prioritized = compute_average_picks(true, false);

    EXPECT_LT(exp_start_prioritized, linear_start_prioritized)
        << "Exponential picker with prioritize_end=false should give lower average ("
        << exp_start_prioritized << ") than Linear picker ("
        << linear_start_prioritized << ")";
}

// Test that output is always within bounds
TEST_F(RandomPickerTest, BoundsTest) {
    const int start = 5;
    const int end = 10;
    const int num_tests = 1000;

    for (int i = 0; i < num_tests; i++) {
        int linear_result = linear_picker.pick_random(start, end, true);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        linear_result = linear_picker.pick_random(start, end, false);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        int exp_result = exp_picker.pick_random(start, end, true);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);

        exp_result = exp_picker.pick_random(start, end, false);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);
    }
}

// Test single-element range always returns that element
TEST_F(RandomPickerTest, SingleElementRangeTest) {
    constexpr int start = 5;
    constexpr int end = 6;  // Range of size 1

    // Should always return start for both true and false prioritize_end
    EXPECT_EQ(linear_picker.pick_random(start, end, true), start);
    EXPECT_EQ(linear_picker.pick_random(start, end, false), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, true), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, false), start);
}

// Test probabilities more deterministically for linear random picker and two elements.
TEST_F(RandomPickerTest, TwoElementRangeDistributionTestForLinearRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    const int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = linear_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = linear_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    // For linear picker with size 2:
    // When prioritize_end=true:
    //   weight(0) = 1, weight(1) = 2, total_weight = 3
    //   prob(0) = 1/3, prob(1) = 2/3
    // When prioritize_end=false:
    //   weight(0) = 2, weight(1) = 1, total_weight = 3
    //   prob(0) = 2/3, prob(1) = 1/3

    constexpr double expected_prob_end = 2.0 / 3.0;    // probability of getting 1 when prioritizing end
    constexpr double expected_prob_start = 1.0 / 3.0;  // probability of getting 1 when prioritizing start

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    // Allow for some statistical variation
    constexpr double tolerance = 0.02;  // 2% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test probabilities more deterministically for exponential random picker and two elements.
TEST_F(RandomPickerTest, TwoElementRangeDistributionTestForExponentialRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    constexpr int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = exp_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = exp_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    // For exponential picker with size 2:
    // When prioritize_end=true:
    //   weight(0) = e^0 = 1, weight(1) = e^1 = e
    //   total_weight = 1 + e
    //   prob(0) = 1/(1+e), prob(1) = e/(1+e)
    const double e = std::exp(1.0);
    const double expected_prob_end = e / (1.0 + e);    // probability of getting 1 when prioritizing end

    // When prioritize_end=false:
    //   weight(0) = e^0 = 1, weight(1) = e^(-1)
    //   total_weight = 1 + e^(-1)
    //   prob(0) = 1/(1+e^(-1)), prob(1) = e^(-1)/(1+e^(-1))
    const double exp_neg_one = std::exp(-1.0);
    const double expected_prob_start = exp_neg_one / (1.0 + exp_neg_one);  // probability of getting 1 when prioritizing start

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    // Allow for some statistical variation
    constexpr double tolerance = 0.02;  // 2% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test the constructor of TemporalWalk to ensure it initializes correctly.
TEST_F(EmptyTemporalWalkTest, ConstructorTest) {
    EXPECT_NO_THROW(temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Uniform));
    EXPECT_EQ(temporal_walk->get_len_walk(), LEN_WALK);
    EXPECT_EQ(temporal_walk->get_node_count(), 0); // Assuming initial node count is 0
}


// Test adding an edge to the TemporalWalk when it's empty.
TEST_F(EmptyTemporalWalkTest, AddEdgeTest) {
    temporal_walk->add_edge(1, 2, 100);
    temporal_walk->add_edge(2, 3, 101);
    temporal_walk->add_edge(7, 8, 102);
    temporal_walk->add_edge(1, 7, 103);
    temporal_walk->add_edge(3, 2, 103);
    temporal_walk->add_edge(10, 11, 104);
    EXPECT_EQ(temporal_walk->get_edge_count(), 6);
    EXPECT_EQ(temporal_walk->get_node_count(), 7);
}

// Test to check if a specific node ID is present in the filled TemporalWalk.
TEST_F(FilledTemporalWalkTest, TestNodeFoundTest) {
    const auto nodes = temporal_walk->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

// Test that the number of random walks generated matches the expected count and checks that no walk exceeds its length.
// Also test that the system can sample walks of length more than 1.
TEST_F(FilledTemporalWalkTest, WalkCountAndLensTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Random, TEST_NODE_ID);
    EXPECT_EQ(walks.size(), NUM_WALKS);

    int total_walk_lens = 0;

    for (const auto& walk : walks) {
        EXPECT_LE(walk.size(), LEN_WALK) << "A walk exceeds the maximum length of " << LEN_WALK;
        EXPECT_GT(walk.size(), 0);

        total_walk_lens += static_cast<int>(walk.size());
    }

    auto average_walk_len = static_cast<float>(total_walk_lens) / static_cast<float>(walks.size());
    EXPECT_GT(average_walk_len, 1) << "System could not sample any walk of length more than 1";
}

// Test that all walks starting from a specific node begin with that node.
TEST_F(FilledTemporalWalkTest, WalkStartTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Begin, TEST_NODE_ID);
    for (const auto& walk : walks) {
        EXPECT_EQ(walk[0].node, TEST_NODE_ID);
    }
}

// Test that all walks ending at a specific node conclude with that node.
TEST_F(FilledTemporalWalkTest, WalkEndTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::End, TEST_NODE_ID);
    for (const auto& walk : walks) {
        EXPECT_EQ(walk.back().node, TEST_NODE_ID);
    }
}

// Test to verify that the timestamps in each walk are strictly increasing.
TEST_F(FilledTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Random, TEST_NODE_ID);

    for (const auto& walk : walks) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify random walks for selected nodes and their properties.
TEST_F(FilledTemporalWalkTest, CheckWalksForNodes) {
    constexpr int num_selected_walks = 100;

    const auto nodes = temporal_walk->get_node_ids();
    const auto selected_nodes = std::vector<int>(nodes.begin(), nodes.begin() + num_selected_walks);

    const auto walks_for_nodes = temporal_walk->get_random_walks_for_nodes_with_times(WalkStartAt::Random, selected_nodes);
    EXPECT_EQ(walks_for_nodes.size(), num_selected_walks);

    for (const auto& node : selected_nodes) {
        auto it = walks_for_nodes.find(node);
        EXPECT_NE(it, walks_for_nodes.end()) << "Node " << node << " is not present in walks_for_nodes.";
        EXPECT_EQ(it->second.size(), NUM_WALKS) << "Node " << node << " does not have the expected number of walks.";
    }

    // Test that each walk for each node is strictly increasing in time.
    for (const auto& node : selected_nodes) {
        auto walks = walks_for_nodes.at(node);

        for (const auto& walk : walks) {
            for (size_t i = 1; i < walk.size(); ++i) {
                EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                    << "Timestamps are not strictly increasing in walk: "
                    << i << " with node: " << walk[i].node
                    << ", previous node: " << walk[i - 1].node;
            }
        }
    }
}
