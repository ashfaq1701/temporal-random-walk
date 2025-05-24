#include <gtest/gtest.h>
#include <cmath>

#include "test_utils.h"
#include "../src/proxies/TemporalRandomWalk.cuh"

constexpr int TEST_NODE_ID = 42;
constexpr int64_t MAX_TIME_CAPACITY = 5;

constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;

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

template<typename T>
class EmptyTemporalRandomWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, T::value, -1, true, -1);
    }

    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
};

TYPED_TEST_SUITE(EmptyTemporalRandomWalkTest, GPU_USAGE_TYPES);

template<typename T>
class EmptyTemporalRandomWalkTestWithMaxCapacity : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, T::value, MAX_TIME_CAPACITY, true, -1);
    }

    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
};

TYPED_TEST_SUITE(EmptyTemporalRandomWalkTestWithMaxCapacity, GPU_USAGE_TYPES);

struct TestParams {
    bool use_gpu;
    int walk_length;

    // Helper for test naming
    [[nodiscard]] std::string ToString() const {
        return std::string(use_gpu ? "GPU" : "CPU") + "_WalkLen" + std::to_string(walk_length);
    }
};

class TestParamGenerator {
public:
    static std::vector<TestParams> GetParams() {
        std::vector<TestParams> params;

        // Walk lengths to test both kernel approaches
        const std::vector<int> walk_lengths = {200, 300}; // Below and above 256 threshold

        #ifdef HAS_CUDA
        // Test both CPU and GPU modes
        const std::vector<bool> gpu_modes = {false, true};
        #else
        // CPU mode only
        const std::vector<bool> gpu_modes = {false};
        #endif

        for (const bool use_gpu : gpu_modes) {
            for (const int walk_len : walk_lengths) {
                params.push_back({use_gpu, walk_len});
            }
        }

        return params;
    }
};

// Parameterized FilledDirectedTemporalRandomWalkTest
class FilledDirectedTemporalRandomWalkTest : public ::testing::TestWithParam<TestParams> {
protected:
    FilledDirectedTemporalRandomWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        const auto& params = GetParam();
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, params.use_gpu, -1, true, -1);
        temporal_random_walk->add_multiple_edges(sample_edges);
        current_walk_length = params.walk_length;
        is_using_gpu = params.use_gpu;
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
    int current_walk_length{};
    bool is_using_gpu{};

    // Helper method to get expected kernel type for logging
    [[nodiscard]] std::string GetExpectedKernel() const {
        return current_walk_length <= RANDOM_WALK_FULL_WALK_SWITCH_THRESHOLD ? "step-by-step" : "single-loop";
    }
};

// Parameterized FilledUndirectedTemporalRandomWalkTest
class FilledUndirectedTemporalRandomWalkTest : public ::testing::TestWithParam<TestParams> {
protected:
    FilledUndirectedTemporalRandomWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        const auto& params = GetParam();
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(false, params.use_gpu, -1, true, -1);
        temporal_random_walk->add_multiple_edges(sample_edges);
        current_walk_length = params.walk_length;
        is_using_gpu = params.use_gpu;
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
    int current_walk_length{};
    bool is_using_gpu{};

    // Helper method to get expected kernel type for logging
    [[nodiscard]] std::string GetExpectedKernel() const {
        return current_walk_length <= RANDOM_WALK_FULL_WALK_SWITCH_THRESHOLD ? "step-by-step" : "single-loop";
    }
};

// Custom test name generator
struct TestNameGenerator {
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const {
        return info.param.ToString();
    }
};

// Parameterized TimescaleBoundedTemporalRandomWalkTest
class TimescaleBoundedTemporalRandomWalkTest : public ::testing::TestWithParam<TestParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, params.use_gpu, -1, true, 10.0);
        temporal_random_walk->add_multiple_edges({
            // Node 1's outgoing edges
            {1, 2, 100},
            {1, 3, 100}, // Same timestamp
            {1, 4, 101}, // Small difference
            {1, 5, 110}, // Larger difference

            // Node 2's outgoing edges
            {2, 3, 130},
            {2, 4, 130}, // Same timestamp
            {2, 5, 160}, // Larger difference

            // Node 3's outgoing edges
            {3, 4, 200},
            {3, 5, 200}, // Same timestamp
            {3, 6, 250}, // Larger difference
        });
        current_walk_length = params.walk_length;
        is_using_gpu = params.use_gpu;
    }

    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
    int current_walk_length{};
    bool is_using_gpu{};

    // Helper method to get expected kernel type for logging
    [[nodiscard]] std::string GetExpectedKernel() const {
        return current_walk_length <= 256 ? "step-by-step" : "single-loop";
    }
};

// Instantiate the parameterized test suites
INSTANTIATE_TEST_SUITE_P(
    DirectedGraphTests,
    FilledDirectedTemporalRandomWalkTest,
    ::testing::ValuesIn(TestParamGenerator::GetParams()),
    TestNameGenerator{}
);

INSTANTIATE_TEST_SUITE_P(
    UndirectedGraphTests,
    FilledUndirectedTemporalRandomWalkTest,
    ::testing::ValuesIn(TestParamGenerator::GetParams()),
    TestNameGenerator{}
);

// Instantiate the parameterized test suite for TimescaleBounded tests
INSTANTIATE_TEST_SUITE_P(
    TimescaleBoundedTests,
    TimescaleBoundedTemporalRandomWalkTest,
    ::testing::ValuesIn(TestParamGenerator::GetParams()),
    TestNameGenerator{}
);

// Test the constructor of TemporalRandomWalk to ensure it initializes correctly.
TYPED_TEST(EmptyTemporalRandomWalkTest, ConstructorTest) {
    EXPECT_NO_THROW(this->temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, TypeParam::value));
    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 0); // Assuming initial node count is 0
}


// Test adding an edge to the TemporalRandomWalk when it's empty.
TYPED_TEST(EmptyTemporalRandomWalkTest, AddEdgeTest) {
    this->temporal_random_walk->add_multiple_edges({
        {1, 2, 100},
        {2, 3, 101},
        {7, 8, 102},
        {1, 7, 103},
        {3, 2, 103},
        {10, 11, 104}
    });

    EXPECT_EQ(this->temporal_random_walk->get_edge_count(), 6);
    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 7);
}

// When later edges are added than the allowed max time capacity, older edges are automatically deleted.
TYPED_TEST(EmptyTemporalRandomWalkTestWithMaxCapacity, WhenMaxTimeCapacityExceedsEdgesAreDeletedAutomatically) {
    this->temporal_random_walk->add_multiple_edges({
        { 0, 2, 1 },
        { 2, 3, 3 },
        { 1, 9, 2 },
        { 2, 4, 3 },
        { 2, 4, 1 },
        { 1, 5, 4 }
    });

    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 7);
    EXPECT_EQ(this->temporal_random_walk->get_edge_count(), 6);

    this->temporal_random_walk->add_multiple_edges({
        { 5, 6, 4 },
        { 2, 5, 4 },
        { 4, 3, 5 },
    });

    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 8);
    EXPECT_EQ(this->temporal_random_walk->get_edge_count(), 9);

    this->temporal_random_walk->add_multiple_edges({
        { 1, 7, 6 }
    });

    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 8);
    EXPECT_EQ(this->temporal_random_walk->get_edge_count(), 8);

    this->temporal_random_walk->add_multiple_edges({
        { 1, 5, 7 },
        { 4, 7, 8 }
    });

    EXPECT_EQ(this->temporal_random_walk->get_node_count(), 7);
    EXPECT_EQ(this->temporal_random_walk->get_edge_count(), 7);
}

// Test to check if a specific node ID is present in the filled TemporalRandomWalk.
TEST_P(FilledDirectedTemporalRandomWalkTest, TestNodeFoundTest) {
    const auto nodes = this->temporal_random_walk->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

// Test that the number of random walks generated matches the expected count and checks that no walk exceeds its length.
// Also test that the system can sample walks of length more than 1.
TEST_P(FilledDirectedTemporalRandomWalkTest, WalkCountAndLensTest) {
    const auto walk_set = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10);

    int total_walk_lens = 0;

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        EXPECT_LE(walk.size(), current_walk_length) << "A walk exceeds the maximum length of " << current_walk_length;
        EXPECT_GT(walk.size(), 0);

        total_walk_lens += static_cast<int>(walk.size());
    }

    auto average_walk_len = static_cast<float>(total_walk_lens) / static_cast<float>(walk_set.size());
    EXPECT_GT(average_walk_len, 1) << "System could not sample any walk of length more than 1";
}

// Test to verify that the timestamps in each walk are strictly increasing in directed graphs.
TEST_P(FilledDirectedTemporalRandomWalkTest, WalkIncreasingTimestampTest) {
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify that the timestamps in each walk are strictly increasing in undirected graphs.
TEST_P(FilledUndirectedTemporalRandomWalkTest, WalkIncreasingTimestampTest) {
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify that each step in walks uses valid edges from the graph
TEST_P(FilledDirectedTemporalRandomWalkTest, WalkValidEdgesTest) {
    // Create a map of valid edges for O(1) lookup
    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->sample_edges) {
        valid_edges[edge] = true;
    }

    // Check forward walks
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i + 1].node;
            int64_t ts = walk[i + 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }

    // Check backward walks
    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }
}

TEST_P(FilledDirectedTemporalRandomWalkTest, WalkTerminalEdgesTest) {
    // For forward walks, track maximum outgoing timestamps
    std::map<int, int64_t> max_outgoing_timestamps;
    // For backward walks, track minimum incoming timestamps
    std::map<int, int64_t> min_incoming_timestamps;

    // Build timestamp maps
    for (const auto& [src, dst, ts] : this->sample_edges) {
        // Track max timestamp of outgoing edges for forward walks
        if (!max_outgoing_timestamps.count(src) || max_outgoing_timestamps[src] < ts) {
            max_outgoing_timestamps[src] = ts;
        }
        // Track min timestamp of incoming edges for backward walks
        if (!min_incoming_timestamps.count(dst) || min_incoming_timestamps[dst] > ts) {
            min_incoming_timestamps[dst] = ts;
        }
    }

    // Check forward walks
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty()) continue;

        // MAX_WALK_LEN approached. No need to check such walks, because they might have finished immaturely.
        if (walk.size() == current_walk_length) continue;

        int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        // Skip if node has no outgoing edges
        if (!max_outgoing_timestamps.count(last_node)) continue;

        int64_t max_ts = max_outgoing_timestamps[last_node];
        if (last_ts < max_ts) {
            // Check for valid edges that we could have walked to
            for (const auto& [src, dst, ts] : this->sample_edges) {
                if (src == last_node && ts > last_ts && ts <= max_ts) {
                    FAIL() << "Forward walk incorrectly terminated:\n"
                          << "  Node: " << last_node << "\n"
                          << "  Current timestamp: " << last_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Max possible timestamp: " << max_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }

    // Check backward walks
    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty()) continue;

        // MAX_WALK_LEN approached. No need to check such walks, because they might have finished immaturely.
        if (walk.size() == current_walk_length) continue;

        int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        // Skip if node has no incoming edges
        if (!min_incoming_timestamps.count(first_node)) continue;

        int64_t min_ts = min_incoming_timestamps[first_node];
        if (first_ts > min_ts) {
            // Check for valid edges that we could have walked to
            for (const auto& [src, dst, ts] : this->sample_edges) {
                if (dst == first_node && ts < first_ts && ts >= min_ts) {
                    FAIL() << "Backward walk incorrectly terminated:\n"
                          << "  Node: " << first_node << "\n"
                          << "  Current timestamp: " << first_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Min possible timestamp: " << min_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }
}

// Test timestamps and valid edges with WeightBasedRandomPicker
TEST_P(FilledDirectedTemporalRandomWalkTest, WalkIncreasingTimestampWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing in backward walk at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

TEST_P(FilledDirectedTemporalRandomWalkTest, WalkValidEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    // Create edge lookup map
    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->sample_edges) {
        valid_edges[edge] = true;
    }

    // Test forward walks
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i+1].node;
            int64_t ts = walk[i+1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }

    // Test backward walks
    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }
}

TEST_P(FilledDirectedTemporalRandomWalkTest, WalkTerminalEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    // Track valid timestamps for each node
    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    for (const auto& [src, dst, ts] : this->temporal_random_walk->get_edges()) {
        next_valid_timestamps[src].push_back(ts);
    }

    // Sort timestamps
    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 100);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == current_walk_length) continue;

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto next_ts = next_valid_timestamps.find(last_node);
        if (next_ts == next_valid_timestamps.end()) continue;

        const auto& next_timestamps = next_ts->second;
        auto next_ts_it = std::upper_bound(next_timestamps.begin(), next_timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, next_timestamps.end())
            << "Timescale bounded walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }

    std::map<int, std::vector<int64_t>> prev_valid_timestamps;
    for (const auto& [src, dst, ts] : this->sample_edges) {
        prev_valid_timestamps[dst].push_back(ts);
    }

    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 100, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == current_walk_length) continue;
        if (walk.back().timestamp == INT64_MAX) continue;  // Skip last sentinel value

        const int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        auto prev_ts = prev_valid_timestamps.find(first_node);
        if (prev_ts == prev_valid_timestamps.end()) continue;

        const auto& prev_timestamps = prev_ts->second;
        auto prev_ts_it = std::lower_bound(prev_timestamps.begin(), prev_timestamps.end(), first_ts);

        EXPECT_GT(prev_ts_it, prev_timestamps.begin())
            << "Backward walk terminated despite having valid edges to node "
            << first_node << " before timestamp " << first_ts;
    }
}

TEST_P(TimescaleBoundedTemporalRandomWalkTest, ValidEdgesWithScaling) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->temporal_random_walk->get_edges()) {
        valid_edges[edge] = true;
    }

    const auto walk_set = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 1000);

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            const auto edge = std::make_tuple(
                walk[i].node,
                walk[i+1].node,
                walk[i+1].timestamp
            );
            EXPECT_TRUE(valid_edges[edge])
                << "Invalid edge in timescale bounded walk: ("
                << std::get<0>(edge) << ","
                << std::get<1>(edge) << ","
                << std::get<2>(edge) << ")";
        }
    }
}

TEST_P(TimescaleBoundedTemporalRandomWalkTest, TerminalEdgeValidation) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    std::map<int, std::vector<int64_t>> prev_valid_timestamps;

    for (const auto& [src, dst, ts] : this->temporal_random_walk->get_edges()) {
        next_valid_timestamps[src].push_back(ts);
        prev_valid_timestamps[dst].push_back(ts);
    }

    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }
    for (auto& [_, timestamps] : prev_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    // Test forward walks
    const auto walk_set_forward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 100, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == current_walk_length) continue;
        if (walk[0].timestamp == INT64_MIN) continue;  // Skip first sentinel value

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto next_ts = next_valid_timestamps.find(last_node);
        if (next_ts == next_valid_timestamps.end()) continue;

        const auto& next_timestamps = next_ts->second;
        auto next_ts_it = std::upper_bound(next_timestamps.begin(), next_timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, next_timestamps.end())
            << "Forward walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }

    // Test backward walks
    const auto walk_set_backward = this->temporal_random_walk->get_random_walks_and_times_for_all_nodes(
        current_walk_length, &exponential_weight_picker, 100, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == current_walk_length) continue;
        if (walk.back().timestamp == INT64_MAX) continue;  // Skip last sentinel value

        const int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        auto prev_ts = prev_valid_timestamps.find(first_node);
        if (prev_ts == prev_valid_timestamps.end()) continue;

        const auto& prev_timestamps = prev_ts->second;
        auto prev_ts_it = std::lower_bound(prev_timestamps.begin(), prev_timestamps.end(), first_ts);

        EXPECT_GT(prev_ts_it, prev_timestamps.begin())
            << "Backward walk terminated despite having valid edges to node "
            << first_node << " before timestamp " << first_ts;
    }
}
