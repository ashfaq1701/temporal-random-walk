#include <gtest/gtest.h>

#include "../src/proxies/TemporalGraph.cuh"
#include "../src/stores/temporal_node2vec_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

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

template<typename T>
class TemporalNode2VecHelpersTest : public ::testing::Test {
protected:
    TemporalGraph graph{true, T::value, -1, true, -1, 2.0, 0.5};

    void SetUp() override {
        graph.add_multiple_edges({
            Edge{0, 1, 10},
            Edge{0, 2, 10},
            Edge{0, 3, 20},
            Edge{1, 2, 5},
            Edge{4, 1, 6}
        });
    }

    static long expected_edge_pick(
        const TemporalGraphStore* graph,
        const int node_id,
        const int prev_node,
        const size_t edge_start,
        const size_t edge_end,
        size_t* node_ts_sorted_indices,
        const double edge_selector_rand_num) {
        double beta_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = temporal_graph::get_node2vec_candidate_node<true, true>(graph, node_id, edge_idx);
            beta_sum += temporal_graph::compute_node2vec_beta_host(graph, prev_node, w);
        }

        const double target = edge_selector_rand_num * beta_sum;
        double running_sum = 0.0;
        for (size_t i = edge_start; i < edge_end; ++i) {
            const size_t edge_idx = node_ts_sorted_indices[i];
            const int w = temporal_graph::get_node2vec_candidate_node<true, true>(graph, node_id, edge_idx);
            running_sum += temporal_graph::compute_node2vec_beta_host(graph, prev_node, w);
            if (running_sum >= target) {
                return static_cast<long>(edge_idx);
            }
        }

        return static_cast<long>(node_ts_sorted_indices[edge_end - 1]);
    }
};

TYPED_TEST_SUITE(TemporalNode2VecHelpersTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalNode2VecHelpersTest, AdjacencyAndBetaRulesWorkForDirectedGraph) {
    const auto* store = this->graph.get_graph();

    EXPECT_TRUE(temporal_graph::is_node_adjacent_to_host_or_device(store, 1, 2));
    EXPECT_TRUE(temporal_graph::is_node_adjacent_to_host_or_device(store, 1, 4));
    EXPECT_FALSE(temporal_graph::is_node_adjacent_to_host_or_device(store, 1, 3));

    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(store, 1, 1), 0.5);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(store, 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(temporal_graph::compute_node2vec_beta_host(store, 1, 3), 2.0);
}

TYPED_TEST(TemporalNode2VecHelpersTest, GroupWeightFromCumulativeHandlesSubranges) {
    std::vector<double> cumulative = {0.10, 0.35, 0.65, 1.00};

    EXPECT_NEAR(temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 1, 1), 0.25, 1e-12);
    EXPECT_NEAR(temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 2, 1), 0.30, 1e-12);
    EXPECT_NEAR(temporal_graph::get_group_exponential_weight_from_cumulative(cumulative.data(), 3, 1), 0.35, 1e-12);
}

TYPED_TEST(TemporalNode2VecHelpersTest, TemporalNode2VecGroupSelectionRespectsCombinedWeightAndBeta) {
    const auto* store = this->graph.get_graph();

    auto* groups_offsets = store->node_edge_index->node_ts_group_outbound_offsets;
    auto* sorted_indices = store->node_edge_index->node_ts_sorted_outbound_indices;
    auto* group_ranges = store->node_edge_index->count_ts_group_per_node_outbound;

    const size_t range_start = group_ranges[0];
    const size_t range_end = group_ranges[1];
    ASSERT_EQ(range_end - range_start, 2);

    std::vector<double> synthetic_cumulative_weights = {0.20, 1.00};

    const int early_pick = temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        store,
        0,
        1,
        range_start,
        range_end,
        range_end,
        groups_offsets,
        sorted_indices,
        synthetic_cumulative_weights.data(),
        0.10);

    const int late_pick = temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        store,
        0,
        1,
        range_start,
        range_end,
        range_end,
        groups_offsets,
        sorted_indices,
        synthetic_cumulative_weights.data(),
        0.90);

    EXPECT_EQ(early_pick, static_cast<int>(range_start));
    EXPECT_EQ(late_pick, static_cast<int>(range_start + 1));
}

TYPED_TEST(TemporalNode2VecHelpersTest, TemporalNode2VecEdgeSelectionMatchesBetaPrefixSampling) {
    const auto* store = this->graph.get_graph();

    auto* groups_offsets = store->node_edge_index->node_ts_group_outbound_offsets;
    auto* sorted_indices = store->node_edge_index->node_ts_sorted_outbound_indices;
    auto* group_ranges = store->node_edge_index->count_ts_group_per_node_outbound;

    const size_t range_start = group_ranges[0];
    const size_t edge_start = groups_offsets[range_start];
    const size_t edge_end = groups_offsets[range_start + 1];
    ASSERT_GT(edge_end, edge_start);

    const long picked_low_rand = temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        store,
        0,
        1,
        edge_start,
        edge_end,
        sorted_indices,
        0.10);

    const long picked_high_rand = temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        store,
        0,
        1,
        edge_start,
        edge_end,
        sorted_indices,
        0.90);

    EXPECT_EQ(picked_low_rand, this->expected_edge_pick(store, 0, 1, edge_start, edge_end, sorted_indices, 0.10));
    EXPECT_EQ(picked_high_rand, this->expected_edge_pick(store, 0, 1, edge_start, edge_end, sorted_indices, 0.90));
}

TYPED_TEST(TemporalNode2VecHelpersTest, InvalidTemporalNode2VecInputsReturnSentinel) {
    const auto* store = this->graph.get_graph();

    auto* groups_offsets = store->node_edge_index->node_ts_group_outbound_offsets;
    auto* sorted_indices = store->node_edge_index->node_ts_sorted_outbound_indices;
    auto* group_ranges = store->node_edge_index->count_ts_group_per_node_outbound;

    const size_t range_start = group_ranges[0];
    const size_t range_end = group_ranges[1];

    std::vector<double> synthetic_cumulative_weights = {0.20, 1.00};

    EXPECT_EQ(temporal_graph::pick_random_temporal_node2vec_host<true, true>(
        store,
        0,
        -1,
        range_start,
        range_end,
        range_end,
        groups_offsets,
        sorted_indices,
        synthetic_cumulative_weights.data(),
        0.5), -1);

    EXPECT_EQ(temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        store,
        0,
        -1,
        groups_offsets[range_start],
        groups_offsets[range_start + 1],
        sorted_indices,
        0.5), -1);

    EXPECT_EQ(temporal_graph::pick_random_temporal_node2vec_edge_host<true, true>(
        store,
        0,
        1,
        groups_offsets[range_start],
        groups_offsets[range_start],
        sorted_indices,
        0.5), -1);
}

#ifdef HAS_CUDA

namespace {

__global__ void adjacency_and_beta_device_kernel(
    const TemporalGraphStore* graph,
    bool* adj_12,
    bool* adj_14,
    bool* adj_13,
    double* beta_return,
    double* beta_neighbor,
    double* beta_distant) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *adj_12 = temporal_graph::is_node_adjacent_to_host_or_device(graph, 1, 2);
        *adj_14 = temporal_graph::is_node_adjacent_to_host_or_device(graph, 1, 4);
        *adj_13 = temporal_graph::is_node_adjacent_to_host_or_device(graph, 1, 3);
        *beta_return = temporal_graph::compute_node2vec_beta_device(graph, 1, 1);
        *beta_neighbor = temporal_graph::compute_node2vec_beta_device(graph, 1, 2);
        *beta_distant = temporal_graph::compute_node2vec_beta_device(graph, 1, 3);
    }
}

__global__ void group_weight_from_cumulative_device_kernel(
    const double* weights,
    double* out_1,
    double* out_2,
    double* out_3) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_1 = temporal_graph::get_group_exponential_weight_from_cumulative(weights, 1, 1);
        *out_2 = temporal_graph::get_group_exponential_weight_from_cumulative(weights, 2, 1);
        *out_3 = temporal_graph::get_group_exponential_weight_from_cumulative(weights, 3, 1);
    }
}

__global__ void temporal_node2vec_group_pick_device_kernel(
    const TemporalGraphStore* graph,
    const int node_id,
    const int prev_node,
    const size_t range_start,
    const size_t range_end,
    const size_t group_end_offset,
    double* weights,
    const double group_selector_rand_num,
    int* out_group) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_group = temporal_graph::pick_random_temporal_node2vec_device<true, true>(
            graph,
            node_id,
            prev_node,
            range_start,
            range_end,
            group_end_offset,
            graph->node_edge_index->node_ts_group_outbound_offsets,
            graph->node_edge_index->node_ts_sorted_outbound_indices,
            weights,
            group_selector_rand_num);
    }
}

__global__ void temporal_node2vec_edge_pick_device_kernel(
    const TemporalGraphStore* graph,
    const int node_id,
    const int prev_node,
    const size_t edge_start,
    const size_t edge_end,
    const double edge_selector_rand_num,
    long* out_edge_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_edge_idx = temporal_graph::pick_random_temporal_node2vec_edge_device<true, true>(
            graph,
            node_id,
            prev_node,
            edge_start,
            edge_end,
            graph->node_edge_index->node_ts_sorted_outbound_indices,
            edge_selector_rand_num);
    }
}

} // namespace

TYPED_TEST(TemporalNode2VecHelpersTest, DeviceAdjacencyAndBetaRulesWorkForDirectedGraph) {
    if constexpr (!TypeParam::value) {
        GTEST_SKIP() << "Device test only runs for GPU-typed variant.";
    }

    const auto* host_store = this->graph.get_graph();
    auto* d_graph = temporal_graph::to_device_ptr(host_store);

    bool *d_adj_12, *d_adj_14, *d_adj_13;
    double *d_beta_return, *d_beta_neighbor, *d_beta_distant;

    ASSERT_EQ(cudaMalloc(&d_adj_12, sizeof(bool)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_adj_14, sizeof(bool)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_adj_13, sizeof(bool)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta_return, sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta_neighbor, sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta_distant, sizeof(double)), cudaSuccess);

    adjacency_and_beta_device_kernel<<<1, 1>>>(d_graph, d_adj_12, d_adj_14, d_adj_13, d_beta_return, d_beta_neighbor, d_beta_distant);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool adj_12 = false, adj_14 = false, adj_13 = true;
    double beta_return = 0.0, beta_neighbor = 0.0, beta_distant = 0.0;

    ASSERT_EQ(cudaMemcpy(&adj_12, d_adj_12, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&adj_14, d_adj_14, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&adj_13, d_adj_13, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&beta_return, d_beta_return, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&beta_neighbor, d_beta_neighbor, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&beta_distant, d_beta_distant, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(adj_12);
    EXPECT_TRUE(adj_14);
    EXPECT_FALSE(adj_13);
    EXPECT_DOUBLE_EQ(beta_return, 0.5);
    EXPECT_DOUBLE_EQ(beta_neighbor, 1.0);
    EXPECT_DOUBLE_EQ(beta_distant, 2.0);

    cudaFree(d_adj_12);
    cudaFree(d_adj_14);
    cudaFree(d_adj_13);
    cudaFree(d_beta_return);
    cudaFree(d_beta_neighbor);
    cudaFree(d_beta_distant);
    temporal_graph::free_device_pointers(d_graph);
}

TYPED_TEST(TemporalNode2VecHelpersTest, DeviceGroupWeightFromCumulativeHandlesSubranges) {
    if constexpr (!TypeParam::value) {
        GTEST_SKIP() << "Device test only runs for GPU-typed variant.";
    }

    std::vector<double> cumulative = {0.10, 0.35, 0.65, 1.00};

    double* d_cumulative;
    double *d_out_1, *d_out_2, *d_out_3;

    ASSERT_EQ(cudaMalloc(&d_cumulative, cumulative.size() * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_1, sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_2, sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_3, sizeof(double)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_cumulative, cumulative.data(), cumulative.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);

    group_weight_from_cumulative_device_kernel<<<1, 1>>>(d_cumulative, d_out_1, d_out_2, d_out_3);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    double out_1 = 0.0, out_2 = 0.0, out_3 = 0.0;
    ASSERT_EQ(cudaMemcpy(&out_1, d_out_1, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&out_2, d_out_2, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&out_3, d_out_3, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_NEAR(out_1, 0.25, 1e-12);
    EXPECT_NEAR(out_2, 0.30, 1e-12);
    EXPECT_NEAR(out_3, 0.35, 1e-12);

    cudaFree(d_cumulative);
    cudaFree(d_out_1);
    cudaFree(d_out_2);
    cudaFree(d_out_3);
}

TYPED_TEST(TemporalNode2VecHelpersTest, DeviceTemporalNode2VecGroupSelectionRespectsCombinedWeightAndBeta) {
    if constexpr (!TypeParam::value) {
        GTEST_SKIP() << "Device test only runs for GPU-typed variant.";
    }

    const auto* host_store = this->graph.get_graph();
    auto* d_graph = temporal_graph::to_device_ptr(host_store);

    auto* group_ranges = host_store->node_edge_index->count_ts_group_per_node_outbound;
    const size_t range_start = group_ranges[0];
    const size_t range_end = group_ranges[1];
    ASSERT_EQ(range_end - range_start, 2);

    std::vector<double> synthetic_cumulative_weights = {0.20, 1.00};
    double* d_weights;
    int* d_out_group;

    ASSERT_EQ(cudaMalloc(&d_weights, synthetic_cumulative_weights.size() * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_weights, synthetic_cumulative_weights.data(), synthetic_cumulative_weights.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_group, sizeof(int)), cudaSuccess);

    temporal_node2vec_group_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        1,
        range_start,
        range_end,
        range_end,
        d_weights,
        0.10,
        d_out_group);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int early_pick = -1;
    ASSERT_EQ(cudaMemcpy(&early_pick, d_out_group, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    temporal_node2vec_group_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        1,
        range_start,
        range_end,
        range_end,
        d_weights,
        0.90,
        d_out_group);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int late_pick = -1;
    ASSERT_EQ(cudaMemcpy(&late_pick, d_out_group, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(early_pick, static_cast<int>(range_start));
    EXPECT_EQ(late_pick, static_cast<int>(range_start + 1));

    cudaFree(d_weights);
    cudaFree(d_out_group);
    temporal_graph::free_device_pointers(d_graph);
}

TYPED_TEST(TemporalNode2VecHelpersTest, DeviceTemporalNode2VecEdgeSelectionMatchesBetaPrefixSampling) {
    if constexpr (!TypeParam::value) {
        GTEST_SKIP() << "Device test only runs for GPU-typed variant.";
    }

    const auto* host_store = this->graph.get_graph();
    auto* d_graph = temporal_graph::to_device_ptr(host_store);

    auto* groups_offsets = host_store->node_edge_index->node_ts_group_outbound_offsets;
    auto* sorted_indices = host_store->node_edge_index->node_ts_sorted_outbound_indices;
    auto* group_ranges = host_store->node_edge_index->count_ts_group_per_node_outbound;

    const size_t range_start = group_ranges[0];
    const size_t edge_start = groups_offsets[range_start];
    const size_t edge_end = groups_offsets[range_start + 1];
    ASSERT_GT(edge_end, edge_start);

    long* d_out_edge_idx;
    ASSERT_EQ(cudaMalloc(&d_out_edge_idx, sizeof(long)), cudaSuccess);

    temporal_node2vec_edge_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        1,
        edge_start,
        edge_end,
        0.10,
        d_out_edge_idx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    long picked_low_rand = -1;
    ASSERT_EQ(cudaMemcpy(&picked_low_rand, d_out_edge_idx, sizeof(long), cudaMemcpyDeviceToHost), cudaSuccess);

    temporal_node2vec_edge_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        1,
        edge_start,
        edge_end,
        0.90,
        d_out_edge_idx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    long picked_high_rand = -1;
    ASSERT_EQ(cudaMemcpy(&picked_high_rand, d_out_edge_idx, sizeof(long), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(picked_low_rand, this->expected_edge_pick(host_store, 0, 1, edge_start, edge_end, sorted_indices, 0.10));
    EXPECT_EQ(picked_high_rand, this->expected_edge_pick(host_store, 0, 1, edge_start, edge_end, sorted_indices, 0.90));

    cudaFree(d_out_edge_idx);
    temporal_graph::free_device_pointers(d_graph);
}

TYPED_TEST(TemporalNode2VecHelpersTest, DeviceInvalidTemporalNode2VecInputsReturnSentinel) {
    if constexpr (!TypeParam::value) {
        GTEST_SKIP() << "Device test only runs for GPU-typed variant.";
    }

    const auto* host_store = this->graph.get_graph();
    auto* d_graph = temporal_graph::to_device_ptr(host_store);

    auto* group_ranges = host_store->node_edge_index->count_ts_group_per_node_outbound;
    const size_t range_start = group_ranges[0];
    const size_t range_end = group_ranges[1];

    std::vector<double> synthetic_cumulative_weights = {0.20, 1.00};
    double* d_weights;
    int* d_out_group;
    long* d_out_edge_idx;

    ASSERT_EQ(cudaMalloc(&d_weights, synthetic_cumulative_weights.size() * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_weights, synthetic_cumulative_weights.data(), synthetic_cumulative_weights.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_group, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out_edge_idx, sizeof(long)), cudaSuccess);

    temporal_node2vec_group_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        -1,
        range_start,
        range_end,
        range_end,
        d_weights,
        0.5,
        d_out_group);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int group_pick = 0;
    ASSERT_EQ(cudaMemcpy(&group_pick, d_out_group, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(group_pick, -1);

    temporal_node2vec_edge_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        -1,
        host_store->node_edge_index->node_ts_group_outbound_offsets[range_start],
        host_store->node_edge_index->node_ts_group_outbound_offsets[range_start + 1],
        0.5,
        d_out_edge_idx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    long edge_pick_prev_missing = 0;
    ASSERT_EQ(cudaMemcpy(&edge_pick_prev_missing, d_out_edge_idx, sizeof(long), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(edge_pick_prev_missing, -1);

    temporal_node2vec_edge_pick_device_kernel<<<1, 1>>>(
        d_graph,
        0,
        1,
        host_store->node_edge_index->node_ts_group_outbound_offsets[range_start],
        host_store->node_edge_index->node_ts_group_outbound_offsets[range_start],
        0.5,
        d_out_edge_idx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    long edge_pick_empty_range = 0;
    ASSERT_EQ(cudaMemcpy(&edge_pick_empty_range, d_out_edge_idx, sizeof(long), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(edge_pick_empty_range, -1);

    cudaFree(d_weights);
    cudaFree(d_out_group);
    cudaFree(d_out_edge_idx);
    temporal_graph::free_device_pointers(d_graph);
}

#endif
