#include <gtest/gtest.h>
#include "../src/proxies/RandomPickerProxies.cuh"

constexpr int RANDOM_START = 0;
constexpr int RANDOM_END = 10000;
constexpr int RANDOM_NUM_SAMPLES = 100000;

template<typename T>
class RandomPickerTest : public ::testing::Test {
protected:

    LinearRandomPickerProxy linear_picker;
    ExponentialIndexRandomPickerProxy exp_picker;

    RandomPickerTest(): linear_picker(T::value), exp_picker(T::value) {}

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

TYPED_TEST_SUITE(RandomPickerTest, GPU_USAGE_TYPES);

// Test that prioritize_end=true gives higher average than prioritize_end=false for both pickers
TYPED_TEST(RandomPickerTest, PrioritizeEndGivesHigherAverage) {
    // For Linear Picker
    const double linear_end_prioritized = this->compute_average_picks(false, true);
    const double linear_start_prioritized = this->compute_average_picks(false, false);
    EXPECT_GT(linear_end_prioritized, linear_start_prioritized)
        << "Linear picker with prioritize_end=true should give higher average ("
        << linear_end_prioritized << ") than prioritize_end=false ("
        << linear_start_prioritized << ")";

    // For Exponential Picker
    const double exp_end_prioritized = this->compute_average_picks(true, true);
    const double exp_start_prioritized = this->compute_average_picks(true, false);
    EXPECT_GT(exp_end_prioritized, exp_start_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than prioritize_end=false ("
        << exp_start_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing end
TYPED_TEST(RandomPickerTest, ExponentialMoreExtremeForEnd) {
    const double linear_end_prioritized = this->compute_average_picks(false, true);
    const double exp_end_prioritized = this->compute_average_picks(true, true);

    EXPECT_GT(exp_end_prioritized, linear_end_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than Linear picker ("
        << linear_end_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing start
TYPED_TEST(RandomPickerTest, ExponentialMoreExtremeForStart) {
    const double linear_start_prioritized = this->compute_average_picks(false, false);
    const double exp_start_prioritized = this->compute_average_picks(true, false);

    EXPECT_LT(exp_start_prioritized, linear_start_prioritized)
        << "Exponential picker with prioritize_end=false should give lower average ("
        << exp_start_prioritized << ") than Linear picker ("
        << linear_start_prioritized << ")";
}

// Test that output is always within bounds
TYPED_TEST(RandomPickerTest, BoundsTest) {
    const int start = 5;
    const int end = 10;
    const int num_tests = 1000;

    for (int i = 0; i < num_tests; i++) {
        int linear_result = this->linear_picker.pick_random(start, end, true);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        linear_result = this->linear_picker.pick_random(start, end, false);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        int exp_result = this->exp_picker.pick_random(start, end, true);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);

        exp_result = this->exp_picker.pick_random(start, end, false);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);
    }
}

// Test single-element range always returns that element
TYPED_TEST(RandomPickerTest, SingleElementRangeTest) {
    constexpr int start = 5;
    constexpr int end = 6;  // Range of size 1

    // Should always return start for both true and false prioritize_end
    EXPECT_EQ(this->linear_picker.pick_random(start, end, true), start);
    EXPECT_EQ(this->linear_picker.pick_random(start, end, false), start);
    EXPECT_EQ(this->exp_picker.pick_random(start, end, true), start);
    EXPECT_EQ(this->exp_picker.pick_random(start, end, false), start);
}

// Test probabilities more deterministically for linear random picker and two elements.
TYPED_TEST(RandomPickerTest, TwoElementRangeDistributionTestForLinearRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    const int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = this->linear_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = this->linear_picker.pick_random(start, end, false);
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
    constexpr double tolerance = 0.01;  // 1% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test probabilities more deterministically for exponential random picker and two elements.
TYPED_TEST(RandomPickerTest, TwoElementRangeDistributionTestForExponentialRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    constexpr int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = this->exp_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = this->exp_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    // For exponential picker with size 2:
    // When prioritize_end=true:
    //   P(0) = (e-1)/(e^2 - 1)
    //   P(1) = (e-1)e/(e^2 - 1)
    const double e = std::exp(1.0);
    const double e_squared = e * e;
    const double expected_prob_end = (e - 1.0) * e / (e_squared - 1.0);  // probability of getting 1

    // When prioritize_end=false:
    //   P(0) = (e-1)e/(e^2 - 1)
    //   P(1) = (e-1)/(e^2 - 1)
    const double expected_prob_start = (e - 1.0) / (e_squared - 1.0);  // probability of getting 1

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    // Allow for some statistical variation
    constexpr double tolerance = 0.01;  // 1% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}
