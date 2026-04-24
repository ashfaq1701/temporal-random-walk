#include "temporal_random_walk.cuh"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

#include "temporal_random_walk_cpu.cuh"
#include "../common/setup.cuh"
#include "../common/random_gen.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"
#include "../common/nvtx.cuh"
#include "../graph/edge_data.cuh"
#include "../graph/node_edge_index.cuh"
#include "../random/pickers.cuh"
#include "../utils/utils.cuh"
#include "../utils/random.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include "temporal_random_walk_kernels_full_walk.cuh"
#include "node_grouped/dispatch.cuh"
#include "../data/walk_set/walk_set_device.cuh"
#endif

#include "../data/walk_set/walk_set_host.cuh"
#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"

// ==================================================================
// core::TemporalRandomWalk constructor
// ==================================================================

core::TemporalRandomWalk::TemporalRandomWalk(
    const bool is_directed, const bool use_gpu,
    const int64_t max_time_capacity,
    const bool enable_weight_computation,
    const bool enable_temporal_node2vec,
    const double timescale_bound,
    const double node2vec_p, const double node2vec_q,
    const int walk_padding_value,
    const uint64_t global_seed,
    const bool shuffle_walk_order)
    : data_(use_gpu),
      walk_padding_value_(walk_padding_value),
      global_seed_(global_seed),
      shuffle_walk_order_(shuffle_walk_order) {

    // last_batch_unique_* track the source/target nodes of the most recent
    // add_multiple_edges batch. They live on the same side as the graph so
    // the cuda walk paths can consume them with no H<->D staging. Default
    // construction in the header pinned them to use_gpu=false; re-bind here
    // to match data_.use_gpu.
    last_batch_unique_sources_ = Buffer<int>(use_gpu);
    last_batch_unique_targets_ = Buffer<int>(use_gpu);

    data_.is_directed              = is_directed;
    data_.max_time_capacity        = max_time_capacity;
    data_.timescale_bound          = timescale_bound;
    data_.node2vec_p               = node2vec_p;
    data_.node2vec_q               = node2vec_q;
    data_.inv_p                    = 1.0 / node2vec_p;
    data_.inv_q                    = 1.0 / node2vec_q;
    data_.enable_temporal_node2vec = enable_temporal_node2vec;
    data_.enable_weight_computation =
        enable_weight_computation || enable_temporal_node2vec;

#ifdef HAS_CUDA
    if (use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaGetDeviceProperties(&cuda_device_prop_, 0));
        // Non-blocking flag ensures this stream does not synchronize
        // against the default legacy stream — two TemporalRandomWalk
        // instances can run GPU work concurrently.
        CUDA_CHECK_AND_CLEAR(cudaStreamCreateWithFlags(
            &stream_, cudaStreamNonBlocking));
    }
#endif
}

// ==================================================================
// core::TemporalRandomWalk destructor + move
// ==================================================================

core::TemporalRandomWalk::~TemporalRandomWalk() {
#ifdef HAS_CUDA
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
#endif
}

core::TemporalRandomWalk::TemporalRandomWalk(TemporalRandomWalk&& other) noexcept
    : data_(std::move(other.data_)),
      walk_padding_value_(other.walk_padding_value_),
      global_seed_(other.global_seed_),
      shuffle_walk_order_(other.shuffle_walk_order_),
      last_batch_unique_sources_(std::move(other.last_batch_unique_sources_)),
      last_batch_unique_targets_(std::move(other.last_batch_unique_targets_))
#ifdef HAS_CUDA
    , cuda_device_prop_(other.cuda_device_prop_),
      stream_(other.stream_)
#endif
{
#ifdef HAS_CUDA
    other.stream_ = nullptr;
#endif
}

core::TemporalRandomWalk& core::TemporalRandomWalk::operator=(
    TemporalRandomWalk&& other) noexcept {
    if (this == &other) return *this;
#ifdef HAS_CUDA
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
#endif
    data_                       = std::move(other.data_);
    walk_padding_value_         = other.walk_padding_value_;
    global_seed_                = other.global_seed_;
    shuffle_walk_order_         = other.shuffle_walk_order_;
    last_batch_unique_sources_  = std::move(other.last_batch_unique_sources_);
    last_batch_unique_targets_  = std::move(other.last_batch_unique_targets_);
#ifdef HAS_CUDA
    cuda_device_prop_ = other.cuda_device_prop_;
    stream_           = other.stream_;
    other.stream_     = nullptr;
#endif
    return *this;
}

// ==================================================================
// namespace temporal_random_walk — private helpers
// ==================================================================

namespace {

// Host path: std::set dedup. values is a host pointer.
void set_last_batch_unique_std(
    const int* values, const size_t n, Buffer<int>& out) {
    std::set<int> s(values, values + n);
    out.shrink_to_fit_empty();
    if (!s.empty()) {
        out.resize(s.size());
        std::copy(s.begin(), s.end(), out.data());
    }
}

#ifdef HAS_CUDA
// GPU path: thrust::sort + thrust::unique on a device scratch buffer
// provided by the caller. Staging H->D is the caller's job so that the
// two copies (sources + targets) issue back-to-back and pipeline in
// parallel before the first sort+unique runs. `out` is
// `last_batch_unique_*_` which is device-resident when data.use_gpu;
// downstream consumers read it directly on device (no D->H copy).
//
// Runs on the default (legacy null) stream so it is naturally ordered
// with the rest of add_multiple_edges_cuda's graph mutations. Walk
// kernels then sync the null stream once at entry before running on
// trw->stream().
void set_last_batch_unique_cuda_device_input(
    int* values_device, const size_t n, Buffer<int>& out) {
    if (n == 0) {
        out.shrink_to_fit_empty();
        return;
    }

    thrust::device_ptr<int> s_begin(values_device);
    thrust::device_ptr<int> s_end(values_device + n);
    thrust::sort(DEVICE_EXECUTION_POLICY, s_begin, s_end);
    auto new_end = thrust::unique(DEVICE_EXECUTION_POLICY, s_begin, s_end);
    const size_t unique_count = static_cast<size_t>(new_end - s_begin);

    out.shrink_to_fit_empty();
    out.resize(unique_count);
    CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
        out.data(), values_device, unique_count * sizeof(int),
        cudaMemcpyDeviceToDevice));
}
#endif

#ifdef HAS_CUDA
// Build (src-only directed / src+tgt deduped undirected) start-node list
// on device, then repeat each by num_walks_per_node. Output is a device
// DataBlock<int> ready for consumption by the _cuda walk kernels.
DataBlock<int> get_last_batch_start_nodes_device(
    const core::TemporalRandomWalk* trw,
    const int num_walks_per_node) {
    const Buffer<int>& src = trw->last_batch_unique_sources();

    Buffer<int> start_device(/*use_gpu=*/true);

    if (trw->is_directed()) {
        start_device.resize(src.size());
        if (src.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data(), src.data(),
                src.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
    } else {
        const Buffer<int>& dst = trw->last_batch_unique_targets();
        const size_t total = src.size() + dst.size();
        start_device.resize(total);
        if (src.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data(), src.data(),
                src.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        if (dst.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data() + src.size(), dst.data(),
                dst.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        if (total > 0) {
            thrust::device_ptr<int> d_begin(start_device.data());
            thrust::device_ptr<int> d_end(start_device.data() + total);
            thrust::sort(DEVICE_EXECUTION_POLICY, d_begin, d_end);
            auto new_end = thrust::unique(DEVICE_EXECUTION_POLICY, d_begin, d_end);
            const size_t unique_count = static_cast<size_t>(new_end - d_begin);
            // Shrink the logical size; keep capacity — scratch is freed on scope exit.
            start_device.resize(unique_count);
        }
    }

    const size_t out_size = start_device.size() * static_cast<size_t>(num_walks_per_node);
    DataBlock<int> repeated(out_size, /*use_gpu=*/true);
    if (start_device.size() > 0 && num_walks_per_node > 0) {
        const int* src_ptr = start_device.data();
        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(out_size),
            thrust::device_pointer_cast(repeated.data),
            [src_ptr, num_walks_per_node] DEVICE (const size_t idx) {
                return src_ptr[idx / static_cast<size_t>(num_walks_per_node)];
            });
        CUDA_KERNEL_CHECK(
            "After thrust transform in get_last_batch_start_nodes_device");
    }
    return repeated;
}
#endif

DataBlock<int> get_last_batch_start_nodes_new(
    const core::TemporalRandomWalk* trw,
    const int num_walks_per_node) {
#ifdef HAS_CUDA
    if (trw->data().use_gpu) {
        return get_last_batch_start_nodes_device(trw, num_walks_per_node);
    }
#endif

    std::vector<int> start_nodes;
    if (trw->is_directed()) {
        const Buffer<int>& src = trw->last_batch_unique_sources();
        start_nodes.assign(src.data(), src.data() + src.size());
    } else {
        const Buffer<int>& src = trw->last_batch_unique_sources();
        const Buffer<int>& dst = trw->last_batch_unique_targets();
        std::set<int> u(src.data(), src.data() + src.size());
        u.insert(dst.data(), dst.data() + dst.size());
        start_nodes.assign(u.begin(), u.end());
    }

    return repeat_elements(
        start_nodes.data(), start_nodes.size(),
        num_walks_per_node, /*use_gpu=*/false);
}

} // namespace

// ==================================================================
// namespace temporal_random_walk — common
// ==================================================================

HOST void temporal_random_walk::add_multiple_edges(
    core::TemporalRandomWalk* trw,
    const int* sources, const int* targets, const int64_t* timestamps,
    const size_t num_edges,
    const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {

    (void)block_dim;  // plumbed for API symmetry; graph-ingestion kernels
                      // still use the compile-time BLOCK_DIM. Tuning
                      // block_dim for walk sampling is where it matters.
    if (num_edges == 0) return;

    NVTX_RANGE_COLORED("add_multiple_edges", nvtx_colors::edge_purple);

#ifdef HAS_CUDA
    if (trw->data().use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            trw->data(), sources, targets, timestamps, num_edges,
            edge_features, feature_dim);
    } else
#endif
    {
        temporal_graph::add_multiple_edges_std(
            trw->data(), sources, targets, timestamps, num_edges,
            edge_features, feature_dim);
    }

    {
        NVTX_RANGE_COLORED("Unique sources/targets", nvtx_colors::edge_purple);
#ifdef HAS_CUDA
        if (trw->data().use_gpu) {
            Buffer<int> src_scratch(/*use_gpu=*/true);
            Buffer<int> tgt_scratch(/*use_gpu=*/true);
            src_scratch.resize(num_edges);
            tgt_scratch.resize(num_edges);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                src_scratch.data(), sources, num_edges * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                tgt_scratch.data(), targets, num_edges * sizeof(int),
                cudaMemcpyHostToDevice));

            set_last_batch_unique_cuda_device_input(
                src_scratch.data(), num_edges,
                trw->last_batch_unique_sources());
            set_last_batch_unique_cuda_device_input(
                tgt_scratch.data(), num_edges,
                trw->last_batch_unique_targets());
            return;
        }
#endif
        set_last_batch_unique_std(sources, num_edges, trw->last_batch_unique_sources());
        set_last_batch_unique_std(targets, num_edges, trw->last_batch_unique_targets());
    }
}

HOST size_t temporal_random_walk::get_node_count(const core::TemporalRandomWalk* trw) {
    return temporal_graph::get_node_count(trw->data());
}

HOST size_t temporal_random_walk::get_edge_count(const core::TemporalRandomWalk* trw) {
    return temporal_graph::get_total_edges(trw->data());
}

HOST std::vector<int> temporal_random_walk::get_node_ids(const core::TemporalRandomWalk* trw) {
    return temporal_graph::get_node_ids(trw->data());
}

HOST std::vector<Edge> temporal_random_walk::get_edges(const core::TemporalRandomWalk* trw) {
    return temporal_graph::get_edges(trw->data());
}

HOST bool temporal_random_walk::get_is_directed(const core::TemporalRandomWalk* trw) {
    return trw->data().is_directed;
}

HOST void temporal_random_walk::clear(core::TemporalRandomWalk* trw) {
    const bool use_gpu = trw->data().use_gpu;
    TemporalGraphData fresh(use_gpu);
    fresh.is_directed               = trw->data().is_directed;
    fresh.max_time_capacity         = trw->data().max_time_capacity;
    fresh.timescale_bound           = trw->data().timescale_bound;
    fresh.node2vec_p                = trw->data().node2vec_p;
    fresh.node2vec_q                = trw->data().node2vec_q;
    fresh.inv_p                     = trw->data().inv_p;
    fresh.inv_q                     = trw->data().inv_q;
    fresh.enable_weight_computation = trw->data().enable_weight_computation;
    fresh.enable_temporal_node2vec  = trw->data().enable_temporal_node2vec;
    trw->data() = std::move(fresh);
    trw->last_batch_unique_sources().shrink_to_fit_empty();
    trw->last_batch_unique_targets().shrink_to_fit_empty();
}

HOST size_t temporal_random_walk::get_memory_used(const core::TemporalRandomWalk* trw) {
    return temporal_graph::get_memory_used(trw->data());
}

// ==================================================================
// Walk sampling — CPU
// ==================================================================

namespace {

WalksWithEdgeFeaturesHost finalize_host_walks(
    core::TemporalRandomWalk* trw, WalkSetHost host_walks) {
    const int fdim = static_cast<int>(trw->data().feature_dim);
    WalksWithEdgeFeaturesHost result(std::move(host_walks), fdim);
    if (fdim > 0) {
        // edge_features is host-resident in TemporalGraphData regardless
        // of use_gpu (Buffer<float> edge_features{false}).
        result.populate_walk_edge_features(trw->data().edge_features.data());
    }
    return result;
}

} // namespace

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (all nodes, std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    std::vector<int> node_ids = temporal_graph::get_node_ids(trw->data());
    const DataBlock<int> repeated_node_ids = repeat_elements(
        node_ids.data(), node_ids.size(),
        num_walks_per_node, trw->data().use_gpu);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);
    }

    WalkSetHost host_walks(repeated_node_ids.size, max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_for_last_batch_std(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (last batch, std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const DataBlock<int> repeated_node_ids =
        get_last_batch_start_nodes_new(trw, num_walks_per_node);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);
    }

    WalkSetHost host_walks(repeated_node_ids.size, max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_std(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    WalkSetHost host_walks(static_cast<size_t>(num_walks_total), max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        num_walks_total + num_walks_total * max_walk_len * 2, false);

    const std::vector<int> start_node_ids(num_walks_total, -1);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        start_node_ids.data(),
        static_cast<size_t>(num_walks_total),
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

// ==================================================================
// Walk sampling — CUDA
// ==================================================================

#ifdef HAS_CUDA

namespace {

uint64_t resolve_base_seed(const core::TemporalRandomWalk* trw) {
    return (trw->global_seed() != EMPTY_GLOBAL_SEED)
        ? trw->global_seed()
        : secure_random_seed();
}

void launch_walk_kernel_dispatch(
    const KernelLaunchType kernel_launch_type,
    const TemporalGraphView& view,
    const bool is_directed,
    const WalkSetView& walk_set_view,
    const int max_walk_len,
    const int* start_node_ids,
    const size_t num_walks,
    const bool all_starts_unconstrained,
    const RandomPickerType walk_bias,
    const RandomPickerType initial_edge_bias,
    const WalkDirection walk_direction,
    const uint64_t base_seed,
    const dim3& grid_dim,
    const dim3& block_dim,
    const cudaStream_t stream) {
    switch (kernel_launch_type) {
        case KernelLaunchType::FULL_WALK: {
            NVTX_RANGE_COLORED("Launch walk kernel (full)", nvtx_colors::walk_green);
            temporal_random_walk::launch_random_walk_kernel_full_walk(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream);
            break;
        }
        case KernelLaunchType::NODE_GROUPED: {
            NVTX_RANGE_COLORED("Launch walk kernel (node-grouped)", nvtx_colors::walk_green);
            temporal_random_walk::dispatch_node_grouped_kernel(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks, all_starts_unconstrained,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream,
                /*force_global_only=*/false);
            break;
        }
        case KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY: {
            NVTX_RANGE_COLORED("Launch walk kernel (node-grouped, global-only)",
                               nvtx_colors::walk_green);
            temporal_random_walk::dispatch_node_grouped_kernel(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks, all_starts_unconstrained,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream,
                /*force_global_only=*/true);
            break;
        }
        default:
            throw std::runtime_error("Unknown KernelLaunchType");
    }
}

} // namespace

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
    NVTX_RANGE_COLORED("Walk Sampling (all nodes)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Drain any pending graph-mutation work issued on the default stream
    // (e.g. from a prior add_multiple_edges) before we enqueue kernels
    // on trw->stream(). trw->stream() is non-blocking, so it does not
    // auto-sync against the legacy null stream.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    std::vector<int> host_node_ids = temporal_graph::get_node_ids(trw->data());
    const DataBlock<int> repeated_node_ids = repeat_elements(
        host_node_ids.data(), host_node_ids.size(),
        num_walks_per_node, trw->data().use_gpu);

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        &trw->cuda_device_prop(),
        block_dim);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
        CUDA_KERNEL_CHECK(
            "After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");
    }

    WalkSetDevice device_walks(repeated_node_ids.size, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    // Drain stream-0 work issued by the prep phase (repeat_elements,
    // shuffle_vector_device, WalkSetDevice::fill) before launching the
    // kernel on the non-blocking trw->stream(); non-blocking streams do
    // not auto-sync with the legacy null stream, so without this the
    // kernel can observe partially written prep buffers.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    // All-nodes variant seeds start_node_ids from the graph's node list, so
    // every entry is a real node id. → constrained path.
    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, repeated_node_ids.data, repeated_node_ids.size,
        /*all_starts_unconstrained=*/false,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_for_all_nodes_cuda");

    // Wait for walk kernels on trw->stream() to complete before the
    // synchronous cudaMemcpy inside download_to_host, which otherwise
    // races (cudaMemcpy only blocks the legacy null stream, not a
    // non-blocking stream).
    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_for_last_batch_cuda(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
    NVTX_RANGE_COLORED("Walk Sampling (last batch)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Drain any pending graph-mutation work on the default stream before
    // trw->stream() kernel launches; see note in the _for_all_nodes_cuda
    // counterpart above.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    const DataBlock<int> repeated_node_ids =
        get_last_batch_start_nodes_new(trw, num_walks_per_node);

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        &trw->cuda_device_prop(),
        block_dim);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
        CUDA_KERNEL_CHECK(
            "After shuffle_vector_device in get_random_walks_and_times_for_last_batch_cuda");
    }

    WalkSetDevice device_walks(repeated_node_ids.size, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    // Drain stream-0 prep work before launching on trw->stream(); see note
    // in the _for_all_nodes_cuda counterpart above.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    // Last-batch variant seeds start_node_ids from real edge endpoints, so
    // every entry is a valid node id. → constrained path.
    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, repeated_node_ids.data, repeated_node_ids.size,
        /*all_starts_unconstrained=*/false,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_for_last_batch_cuda");

    // Wait for walk kernels on trw->stream() before the synchronous
    // cudaMemcpy inside download_to_host; see note in the
    // _for_all_nodes_cuda counterpart above.
    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
temporal_random_walk::get_random_walks_and_times_cuda(
    core::TemporalRandomWalk* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
    NVTX_RANGE_COLORED("Walk Sampling", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Drain any pending graph-mutation work on the default stream before
    // trw->stream() kernel launches; see note in the _for_all_nodes_cuda
    // counterpart above.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        num_walks_total,
        &trw->cuda_device_prop(),
        block_dim);

    // Device-side start_node_ids filled with -1 (random start).
    Buffer<int> start_node_ids(true);
    start_node_ids.resize(num_walks_total);
    start_node_ids.fill(-1);

    WalkSetDevice device_walks(num_walks_total, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    // Drain stream-0 prep work before launching on trw->stream(); see note
    // in the _for_all_nodes_cuda counterpart above.
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    // start_node_ids is filled with -1 above; every entry is a random-start
    // sentinel. → fully unconstrained path.
    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, start_node_ids.data(), static_cast<size_t>(num_walks_total),
        /*all_starts_unconstrained=*/true,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_cuda");

    // Wait for walk kernels on trw->stream() before the synchronous
    // cudaMemcpy inside download_to_host; see note in the
    // _for_all_nodes_cuda counterpart above.
    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

#endif

// ==================================================================
// core::TemporalRandomWalk method bodies (forwarders)
// ==================================================================

void core::TemporalRandomWalk::add_multiple_edges(
    const int* sources, const int* targets, const int64_t* timestamps,
    const size_t n, const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {
    temporal_random_walk::add_multiple_edges(
        this, sources, targets, timestamps, n, edge_features, feature_dim,
        block_dim);
}

void core::TemporalRandomWalk::add_multiple_edges(
    const std::vector<std::tuple<int, int, int64_t>>& edges,
    const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {
    std::vector<int> sources; sources.reserve(edges.size());
    std::vector<int> targets; targets.reserve(edges.size());
    std::vector<int64_t> timestamps; timestamps.reserve(edges.size());
    for (const auto& e : edges) {
        sources.push_back(std::get<0>(e));
        targets.push_back(std::get<1>(e));
        timestamps.push_back(std::get<2>(e));
    }
    add_multiple_edges(sources.data(), targets.data(), timestamps.data(),
                       timestamps.size(), edge_features, feature_dim, block_dim);
}

WalksWithEdgeFeaturesHost
core::TemporalRandomWalk::get_random_walks_and_times_for_all_nodes(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
#ifdef HAS_CUDA
    if (data_.use_gpu) {
        return temporal_random_walk::get_random_walks_and_times_for_all_nodes_cuda(
            this, max_walk_len, walk_bias, num_walks_per_node,
            initial_edge_bias, walk_direction, kernel_launch_type, block_dim);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    return temporal_random_walk::get_random_walks_and_times_for_all_nodes_std(
        this, max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction);
}

WalksWithEdgeFeaturesHost
core::TemporalRandomWalk::get_random_walks_and_times_for_last_batch(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
#ifdef HAS_CUDA
    if (data_.use_gpu) {
        return temporal_random_walk::get_random_walks_and_times_for_last_batch_cuda(
            this, max_walk_len, walk_bias, num_walks_per_node,
            initial_edge_bias, walk_direction, kernel_launch_type, block_dim);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    return temporal_random_walk::get_random_walks_and_times_for_last_batch_std(
        this, max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction);
}

WalksWithEdgeFeaturesHost
core::TemporalRandomWalk::get_random_walks_and_times(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim) {
#ifdef HAS_CUDA
    if (data_.use_gpu) {
        return temporal_random_walk::get_random_walks_and_times_cuda(
            this, max_walk_len, walk_bias, num_walks_total,
            initial_edge_bias, walk_direction, kernel_launch_type, block_dim);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    return temporal_random_walk::get_random_walks_and_times_std(
        this, max_walk_len, walk_bias, num_walks_total,
        initial_edge_bias, walk_direction);
}

void core::TemporalRandomWalk::set_node_features(
    const int* node_ids, const size_t num_nodes,
    const float* node_features_src, const size_t feature_dim) {
    node_features::set_node_features(
        data_, data_.max_node_id, node_ids, num_nodes,
        node_features_src, feature_dim);
}

size_t core::TemporalRandomWalk::get_node_count() const {
    return temporal_random_walk::get_node_count(this);
}
size_t core::TemporalRandomWalk::get_edge_count() const {
    return temporal_random_walk::get_edge_count(this);
}
std::vector<int> core::TemporalRandomWalk::get_node_ids() const {
    return temporal_random_walk::get_node_ids(this);
}
std::vector<Edge> core::TemporalRandomWalk::get_edges() const {
    return temporal_random_walk::get_edges(this);
}
void core::TemporalRandomWalk::clear() {
    temporal_random_walk::clear(this);
}
size_t core::TemporalRandomWalk::get_memory_used() const {
    return temporal_random_walk::get_memory_used(this);
}
