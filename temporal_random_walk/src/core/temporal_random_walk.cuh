#ifndef TEMPORAL_RANDOM_WALK_STORE_H
#define TEMPORAL_RANDOM_WALK_STORE_H

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "../common/macros.cuh"
#include "../common/const.cuh"
#include "../common/cuda_config.cuh"   // BLOCK_DIM default for block_dim params
#include "../data/structs.cuh"
#include "../data/enums.cuh"
#include "../data/buffer.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/walk_set/walks_with_edge_features_host.cuh"
#include "../graph/temporal_graph.cuh"
#include "../graph/node_features.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

// ==================================================================
// Top-level class.
// ==================================================================

namespace core {

class TemporalRandomWalk {
public:
    TemporalRandomWalk(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity = -1,
        bool enable_weight_computation = false,
        bool enable_temporal_node2vec = false,
        double timescale_bound = DEFAULT_TIMESCALE_BOUND,
        double node2vec_p = DEFAULT_NODE2VEC_P,
        double node2vec_q = DEFAULT_NODE2VEC_Q,
        int walk_padding_value = EMPTY_NODE_VALUE,
        uint64_t global_seed = EMPTY_GLOBAL_SEED,
        bool shuffle_walk_order = DEFAULT_SHUFFLE_WALK_ORDER);

    ~TemporalRandomWalk();

    TemporalRandomWalk(const TemporalRandomWalk&) = delete;
    TemporalRandomWalk& operator=(const TemporalRandomWalk&) = delete;
    // Move semantics need to manage stream ownership, so implement manually.
    TemporalRandomWalk(TemporalRandomWalk&&) noexcept;
    TemporalRandomWalk& operator=(TemporalRandomWalk&&) noexcept;

    // Accessors
    TemporalGraphData&       data()       { return data_; }
    const TemporalGraphData& data() const { return data_; }

    int      walk_padding_value() const { return walk_padding_value_; }
    uint64_t global_seed()        const { return global_seed_; }
    bool     shuffle_walk_order() const { return shuffle_walk_order_; }
    bool     is_directed()        const { return data_.is_directed; }

    Buffer<int>&       last_batch_unique_sources()       { return last_batch_unique_sources_; }
    const Buffer<int>& last_batch_unique_sources() const { return last_batch_unique_sources_; }
    Buffer<int>&       last_batch_unique_targets()       { return last_batch_unique_targets_; }
    const Buffer<int>& last_batch_unique_targets() const { return last_batch_unique_targets_; }

#ifdef HAS_CUDA
    const cudaDeviceProp& cuda_device_prop() const { return cuda_device_prop_; }

    // Non-blocking stream owned by this instance. All GPU work inside
    // the walk pipeline (kernel launches, thrust ops, async memcpys)
    // should eventually flow through this stream so that concurrent
    // instances don't serialize on the default legacy stream.
    cudaStream_t stream() const { return stream_; }

    // Block the host until any outstanding async work on stream_ has
    // completed. Called at user-visible boundaries (e.g. before
    // returning results to the caller) so host-observed reads are safe.
    void sync_stream() const {
        if (data_.use_gpu && stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
    }
#endif

    // Public methods (forward to namespace free functions)
    void add_multiple_edges(
        const int* sources, const int* targets,
        const int64_t* timestamps, size_t num_edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    void add_multiple_edges(
        const std::vector<std::tuple<int, int, int64_t>>& edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    WalksWithEdgeFeaturesHost get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    void set_node_features(
        const int* node_ids, size_t num_nodes,
        const float* node_features_src, size_t feature_dim);

    size_t get_node_count() const;
    size_t get_edge_count() const;
    std::vector<int> get_node_ids() const;
    std::vector<Edge> get_edges() const;
    bool get_is_directed() const { return data_.is_directed; }
    void clear();
    size_t get_memory_used() const;

private:
    TemporalGraphData data_;
    int      walk_padding_value_;
    uint64_t global_seed_;
    bool     shuffle_walk_order_;
    Buffer<int> last_batch_unique_sources_{/*use_gpu=*/false};
    Buffer<int> last_batch_unique_targets_{/*use_gpu=*/false};
#ifdef HAS_CUDA
    cudaDeviceProp cuda_device_prop_{};
    cudaStream_t   stream_{nullptr};
#endif
};

} // namespace core

// ==================================================================
// Namespace free-function overloads taking core::TemporalRandomWalk*.
// ==================================================================

namespace temporal_random_walk {

    HOST void add_multiple_edges(
        core::TemporalRandomWalk* trw,
        const int* sources, const int* targets,
        const int64_t* timestamps, size_t num_edges,
        const float* edge_features = nullptr, size_t feature_dim = 0,
        size_t block_dim = BLOCK_DIM);

    HOST size_t get_node_count(const core::TemporalRandomWalk* trw);
    HOST size_t get_edge_count(const core::TemporalRandomWalk* trw);
    HOST std::vector<int>  get_node_ids(const core::TemporalRandomWalk* trw);
    HOST std::vector<Edge> get_edges(const core::TemporalRandomWalk* trw);
    HOST bool              get_is_directed(const core::TemporalRandomWalk* trw);
    HOST void              clear(core::TemporalRandomWalk* trw);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes_std(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch_std(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_std(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time);

#ifdef HAS_CUDA
    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_all_nodes_cuda(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_for_last_batch_cuda(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);

    HOST WalksWithEdgeFeaturesHost get_random_walks_and_times_cuda(
        core::TemporalRandomWalk* trw,
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias = nullptr,
        WalkDirection walk_direction = WalkDirection::Forward_In_Time,
        KernelLaunchType kernel_launch_type = DEFAULT_KERNEL_LAUNCH_TYPE,
        size_t block_dim = BLOCK_DIM,
        int w_threshold_warp = W_THRESHOLD_WARP);
#endif

    HOST size_t get_memory_used(const core::TemporalRandomWalk* trw);
}

#endif // TEMPORAL_RANDOM_WALK_STORE_H
