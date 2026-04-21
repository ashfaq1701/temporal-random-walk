#include <gtest/gtest.h>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include "test_utils.h"
#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/TemporalRandomWalk.cuh"

namespace {

// Value chosen to be well outside any real node id in sample_data.csv (max
// id is 111). A padding that collides with a real node would conflate
// "unused slot" with "legitimate node", which is a separate design concern.
constexpr int CUSTOM_WALK_PADDING_VALUE = 1000000;
constexpr int MAX_WALK_LEN = 20;
constexpr int NUM_WALKS_PER_NODE = 10;
constexpr RandomPickerType LINEAR_PICKER = RandomPickerType::Linear;

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

// One trw instance per fixture — KernelLaunchType is chosen per
// walk-sampling call, so the same trw drives both FullWalk and StepBased
// invocations. On CPU (use_gpu=false) both dispatch to
// launch_random_walk_cpu_new; on GPU each KernelLaunchType exercises its
// own kernel pipeline. This satisfies CPU_GPU_PAIRING.md: both backends
// covered, both GPU kernel variants covered.
template <typename T>
class WalkPaddingValueTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        trw = std::make_unique<TemporalRandomWalk>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/false,
            /*enable_temporal_node2vec=*/false,
            /*timescale_bound=*/DEFAULT_TIMESCALE_BOUND,
            /*node2vec_p=*/DEFAULT_NODE2VEC_P,
            /*node2vec_q=*/DEFAULT_NODE2VEC_Q,
            /*walk_padding_value=*/CUSTOM_WALK_PADDING_VALUE);
        trw->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<TemporalRandomWalk> trw;
};

TYPED_TEST_SUITE(WalkPaddingValueTest, GPU_USAGE_TYPES);

struct WalkStats {
    size_t num_walks = 0;
    size_t num_walks_longer_than_3 = 0;
};

// Runs the three invariants the wiring fix has to uphold:
//   (a) every slot in [0, walk_len) holds a real node — no -1 sentinel
//       leak, and no accidental hop onto the configured padding value
//       (which would mean the termination check failed to fire).
//   (b) every slot in [walk_len, max_walk_len) equals the configured
//       padding value — the WalkSet tail is consistent with what the user
//       asked for.
//   (c) enough walks extend past length 3 — if wiring is wrong, the
//       symptom we care about is *every* walk getting stuck at 2 or 3
//       (start edge + maybe one hop, then jammed), which this catches.
// ASSERT_* inside a helper can only short-circuit a void function, so the
// caller reads stats via outparam and wraps the call with
// ASSERT_NO_FATAL_FAILURE.
void assert_walks_respect_padding(
    const WalksWithEdgeFeaturesHost& walks,
    const int expected_padding_value,
    const size_t max_walk_len,
    WalkStats* out_stats) {
    *out_stats = WalkStats{};

    const int* nodes_ptr = walks.walk_set.nodes_ptr();
    const size_t* walk_lens_ptr = walks.walk_set.walk_lens_ptr();
    const size_t num_walks_total = walks.walk_set.num_walks();

    for (size_t w = 0; w < num_walks_total; ++w) {
        const size_t walk_len = walk_lens_ptr[w];
        if (walk_len == 0) continue;

        ++out_stats->num_walks;
        if (walk_len > 3) {
            ++out_stats->num_walks_longer_than_3;
        }

        const size_t row_base = w * max_walk_len;

        for (size_t i = 0; i < walk_len; ++i) {
            const int node = nodes_ptr[row_base + i];
            ASSERT_NE(node, -1)
                << "Walk " << w << " hop " << i
                << " leaked the -1 sentinel into the valid prefix";
            ASSERT_NE(node, expected_padding_value)
                << "Walk " << w << " hop " << i
                << " equals the configured padding value; the termination "
                << "check should have fired before this hop was written";
        }

        for (size_t i = walk_len; i < max_walk_len; ++i) {
            const int node = nodes_ptr[row_base + i];
            ASSERT_EQ(node, expected_padding_value)
                << "Walk " << w << " tail slot " << i << " is " << node
                << " but should be the configured padding value "
                << expected_padding_value;
        }
    }
}

void expect_enough_long_walks(const WalkStats& stats) {
    ASSERT_GT(stats.num_walks, 0u);
    const double frac_long =
        static_cast<double>(stats.num_walks_longer_than_3)
        / static_cast<double>(stats.num_walks);
    EXPECT_GE(frac_long, 0.05)
        << "Only " << stats.num_walks_longer_than_3 << " / "
        << stats.num_walks
        << " walks reached length > 3 under non-default padding. "
        << "The walk path is likely truncating because walk_padding_value "
        << "is not threaded into the kernel's termination check.";
}

}  // namespace

TYPED_TEST(WalkPaddingValueTest, FullWalkRespectsConfiguredPaddingValue) {
    const auto walks = this->trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::FULL_WALK);

    WalkStats stats{};
    ASSERT_NO_FATAL_FAILURE(assert_walks_respect_padding(
        walks, CUSTOM_WALK_PADDING_VALUE,
        static_cast<size_t>(MAX_WALK_LEN), &stats));
    expect_enough_long_walks(stats);
}

TYPED_TEST(WalkPaddingValueTest, StepBasedRespectsConfiguredPaddingValue) {
    const auto walks = this->trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::STEP_BASED);

    WalkStats stats{};
    ASSERT_NO_FATAL_FAILURE(assert_walks_respect_padding(
        walks, CUSTOM_WALK_PADDING_VALUE,
        static_cast<size_t>(MAX_WALK_LEN), &stats));
    expect_enough_long_walks(stats);
}
