#include <gtest/gtest.h>
#include <random>

#include "../src/proxies/RandomPicker.cuh"
#include "../src/random/pickers.cuh"
#include "../src/utils/utils.cuh"
#include "../src/common/random_gen.cuh"

template<typename T>
class WeightBasedRandomPickerTest : public ::testing::Test
{
protected:

    WeightBasedRandomPicker picker;

    WeightBasedRandomPickerTest(): picker(T::value) {}

    // Helper to verify sampling is within correct range
    void verify_sampling_range(const std::vector<double>& weights,
                               const int start,
                               const int end,
                               const int num_samples = 1000)
    {
        std::map<int, int> sample_counts;

        Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, T::value);
        double* random_nums = random_nums_buf.data();

        for (int i = 0; i < num_samples; i++)
        {
            int picked = picker.pick_random_with_provided_number(weights, start, end, random_nums + i);
            EXPECT_GE(picked, start) << "Sampled index below start";
            EXPECT_LT(picked, end) << "Sampled index at or above end";
            ++sample_counts[picked];
        }

        // Verify all valid indices were sampled
        for (int i = start; i < end; i++)
        {
            EXPECT_GT(sample_counts[i], 0)
                << "Index " << i << " was never sampled";
        }
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

TYPED_TEST_SUITE(WeightBasedRandomPickerTest, GPU_USAGE_TYPES);

TYPED_TEST(WeightBasedRandomPickerTest, ValidationChecks)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // Invalid start index
    EXPECT_EQ(this->picker.pick_random(weights, -1, 2), -1);

    // End <= start
    EXPECT_EQ(this->picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(this->picker.pick_random(weights, 2, 1), -1);

    // End > size
    EXPECT_EQ(this->picker.pick_random(weights, 0, 5), -1);
}

TYPED_TEST(WeightBasedRandomPickerTest, FullRangeSampling)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    this->verify_sampling_range(weights, 0, 4);
}

TYPED_TEST(WeightBasedRandomPickerTest, SubrangeSampling)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    // Test all subranges with the same weight vector
    this->verify_sampling_range(weights, 1, 3);  // middle range
    this->verify_sampling_range(weights, 0, 2);  // start range
    this->verify_sampling_range(weights, 2, 4);  // end range
}

TYPED_TEST(WeightBasedRandomPickerTest, SingleElementRange)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    constexpr int num_samples = 100;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, TypeParam::value);
    double* random_nums = random_nums_buf.data();

    // When sampling single element, should always return that index
    for (int i = 0; i < num_samples; i++)
    {
        EXPECT_EQ(this->picker.pick_random_with_provided_number(weights, 1, 2, random_nums + i), 1);
    }
}

TYPED_TEST(WeightBasedRandomPickerTest, WeightDistributionTest)
{
    // Create weights with known distribution
    std::vector<double> weights = {0.25, 0.5, 0.75, 1.0}; // Equal increments

    std::map<int, int> sample_counts;
    constexpr int num_samples = 100000;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, TypeParam::value);
    double* random_nums = random_nums_buf.data();

    for (int i = 0; i < num_samples; i++)
    {
        int picked = this->picker.pick_random_with_provided_number(weights, 0, 4, random_nums + i);
        ++sample_counts[picked];
    }

    // Each index should be sampled roughly equally since weights
    // have equal increments
    for (int i = 0; i < 4; i++)
    {
        const double proportion = static_cast<double>(sample_counts[i]) / num_samples;
        EXPECT_NEAR(proportion, 0.25, 0.01)
            << "Proportion for index " << i << " was " << proportion;
    }
}

TYPED_TEST(WeightBasedRandomPickerTest, EdgeCaseWeights)
{
    // Test with very small weight differences
    std::vector<double> small_diffs = {0.1, 0.100001, 0.100002, 0.100003};
    EXPECT_NE(this->picker.pick_random(small_diffs, 0, 4), -1);

    // Test with very large weight differences
    std::vector<double> large_diffs = {0.1, 0.5, 0.9, 1000.0};
    EXPECT_NE(this->picker.pick_random(large_diffs, 0, 4), -1);
}

// Regression test for the slice_start parameter in
// random_pickers::pick_random_exponential_weights.
//
// Per-node cum_weights arrays (e.g. outbound_forward_cumulative_weights_
// exponential) are piecewise: each node's range is independently
// normalized and ends at 1.0. Before the fix, picking a range that
// starts exactly at a non-first node's boundary
// (group_start == node_group_begin > 0) collapsed to "always return the
// last group" — the picker read weights[group_start - 1] as the prefix
// sum, which is actually the previous node's total (= 1.0), making
// (end_sum - start_sum) == 0 and random_val deterministic at 1.0.
//
// With slice_start = node_group_begin, the picker treats
// group_start == slice_start as "prefix is 0" and samples correctly.
//
// Host-only test — the picker is HOST DEVICE, and the sampling math is
// bit-identical on both paths by design of the unified implementation.
TEST(WeightPickerPiecewiseCDFTest, SliceStartFixAtNodeBoundary)
{
    // Two back-to-back piecewise per-node CDFs, each ending at 1.0:
    //   node A (indices 0, 1): cum [0.3, 1.0]   (per-group weights [0.3, 0.7])
    //   node B (indices 2, 3): cum [0.25, 1.0]  (per-group weights [0.25, 0.75])
    const std::vector<double> weights = {0.3, 1.0, 0.25, 1.0};
    constexpr size_t node_b_begin = 2;
    constexpr size_t node_b_end   = 4;

    constexpr int num_samples = 100000;

    // Deterministic seed keeps the test reproducible across runs.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // --- Correct usage: slice_start = node_b_begin -----------------------
    std::map<int, int> sample_counts;
    for (int i = 0; i < num_samples; i++)
    {
        const double r = dist(rng);
        const int picked = random_pickers::pick_random_exponential_weights(
            weights.data(), weights.size(),
            node_b_begin, node_b_end, r,
            /*slice_start=*/node_b_begin);
        ASSERT_GE(picked, static_cast<int>(node_b_begin));
        ASSERT_LT(picked, static_cast<int>(node_b_end));
        ++sample_counts[picked];
    }

    const double p_group_2 = static_cast<double>(sample_counts[2]) / num_samples;
    const double p_group_3 = static_cast<double>(sample_counts[3]) / num_samples;
    EXPECT_NEAR(p_group_2, 0.25, 0.01)
        << "Node B group 0 should be sampled ~25% of the time; got "
        << p_group_2;
    EXPECT_NEAR(p_group_3, 0.75, 0.01)
        << "Node B group 1 should be sampled ~75% of the time; got "
        << p_group_3;

    // --- Documented failure mode: default slice_start=0 on piecewise -----
    // If a caller forgets slice_start on a piecewise CDF, the picker
    // reads weights[node_b_begin - 1] = 1.0 as the "prefix sum," making
    // (end_sum - start_sum) == 0 and random_val deterministic at 1.0.
    // Every sample then lands on the last group. This assertion
    // documents the exact failure mode and keeps the slice_start
    // parameter load-bearing — if someone were to remove it, this
    // assertion would still pass, but the correctness assertion above
    // would catastrophically fail.
    std::mt19937 rng_buggy(42);  // same seed as correct path above
    int degenerate_hits = 0;
    for (int i = 0; i < num_samples; i++)
    {
        const double r = dist(rng_buggy);
        const int picked = random_pickers::pick_random_exponential_weights(
            weights.data(), weights.size(),
            node_b_begin, node_b_end, r);  // slice_start defaults to 0
        if (picked == static_cast<int>(node_b_end - 1)) ++degenerate_hits;
    }
    EXPECT_EQ(degenerate_hits, num_samples)
        << "Piecewise CDF with slice_start=0 should degenerate to always "
           "picking the last group — this is the exact bug the fix "
           "addresses. If this assertion fails, reconsider the semantics "
           "of pick_random_exponential_weights.";
}