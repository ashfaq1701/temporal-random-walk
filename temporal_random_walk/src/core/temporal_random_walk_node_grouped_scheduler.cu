#include "temporal_random_walk_node_grouped_scheduler.cuh"

#ifdef HAS_CUDA

#include <cstddef>
#include <cstdint>

#include "../common/cuda_sort.cuh"
#include "../common/cuda_scan.cuh"
#include "../common/error_handlers.cuh"
#include "../common/nvtx.cuh"

namespace temporal_random_walk {

namespace {

// ==========================================================================
// Scheduler helper kernels. Non-templated, anonymous-namespace scoped so
// only this TU instantiates them. Kernels that service per-walk work
// (pick_start_edges_kernel, node_grouped_solo_kernel, the four cooperative
// scaffolds, reverse_walks_kernel) live in
// temporal_random_walk_kernels_node_grouped.cuh.
// ==========================================================================

// 0, 1, ..., n-1 — seed the sort/compact payload buffer.
__global__ void iota_int_kernel(int* __restrict__ out, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i;
}

// Flag live walks (last_node != walk_padding_value). Input for compaction.
__global__ void walk_alive_flags_kernel(
    WalkSetView walk_set,
    const int step_number,
    const int max_walk_len,
    const std::size_t num_walks,
    uint8_t* __restrict__ alive_flags) {

    const std::size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    const std::size_t offset = walk_idx * static_cast<std::size_t>(max_walk_len)
                               + static_cast<std::size_t>(step_number);
    alive_flags[walk_idx] =
        (walk_set.nodes[offset] != walk_set.walk_padding_value)
            ? uint8_t{1} : uint8_t{0};
}

// Gather last-node keys for the compacted active-walk set.
__global__ void gather_last_nodes_kernel(
    WalkSetView walk_set,
    const int* __restrict__ active_walk_idx,
    const int* __restrict__ num_active_ptr,
    const int step_number,
    const int max_walk_len,
    int* __restrict__ last_nodes_out) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_active = *num_active_ptr;
    if (i >= num_active) return;

    const int walk_idx = active_walk_idx[i];
    const std::size_t offset = static_cast<std::size_t>(walk_idx)
                               * static_cast<std::size_t>(max_walk_len)
                               + static_cast<std::size_t>(step_number);
    last_nodes_out[i] = walk_set.nodes[offset];
}

// For each sorted slot, binary-search its run in run_starts and write the
// run's length back to walk_to_group_size at the original walk's index.
__global__ void scatter_walk_group_sizes_kernel(
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ run_starts,
    const int* __restrict__ run_lengths,
    const int* __restrict__ num_runs_ptr,
    const int* __restrict__ num_items_ptr,
    int* walk_to_group_size,
    const int num_walks) {

    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_items = *num_items_ptr;
    if (slot >= num_items) return;

    const int num_runs = *num_runs_ptr;
    if (num_runs <= 0) return;

    int lo = 0, hi = num_runs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (run_starts[mid] <= slot) lo = mid;
        else                         hi = mid - 1;
    }

    const int walk_idx = sorted_walk_idx[slot];
    if (walk_idx < 0 || walk_idx >= num_walks) return;
    walk_to_group_size[walk_idx] = run_lengths[lo];
}

// Zero-fill an int buffer of length n.
__global__ void zero_int_buffer_kernel(int* __restrict__ buf, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = 0;
}

}  // namespace

// ==========================================================================
// NodeGroupedScheduler — implementation.
// ==========================================================================

NodeGroupedScheduler::NodeGroupedScheduler(
    const std::size_t num_walks,
    const dim3 block_dim,
    const cudaStream_t stream)
    : num_walks_(num_walks),
      num_walks_int_(static_cast<int>(num_walks)),
      block_dim_(block_dim),
      stream_(stream),
      arena_(/*use_gpu=*/true),
      iota_src_(/*use_gpu=*/true),
      walk_to_group_size_(/*use_gpu=*/true) {

    iota_src_.resize(num_walks);
    walk_to_group_size_.resize(num_walks);

    // iota_src_ is the reusable input to cub_partition_flagged in every
    // intermediate step. Seed once on the caller's stream.
    const std::size_t iota_blocks =
        (num_walks + block_dim_.x - 1) / block_dim_.x;
    iota_int_kernel<<<dim3(static_cast<unsigned>(iota_blocks)),
                      block_dim_, 0, stream_>>>(
        iota_src_.data(), num_walks_int_);

    // Unconstrained step 0 never calls setup_step0_constrained. Zero so
    // downstream consumers (task 5's W-partition) see a defined value.
    CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
        walk_to_group_size_.data(), 0,
        num_walks * sizeof(int), stream_));
}

void NodeGroupedScheduler::setup_step0_constrained(
    const int* start_node_ids) {

    NVTX_RANGE_COLORED("NG step0 setup", nvtx_colors::index_blue);

    arena_.reset();

    int* sorted_start_keys = arena_.acquire<int>(num_walks_);
    int* sorted_walk_idx   = arena_.acquire<int>(num_walks_);
    int* unique_start_keys = arena_.acquire<int>(num_walks_);
    int* run_lengths       = arena_.acquire<int>(num_walks_);
    int* run_starts        = arena_.acquire<int>(num_walks_);
    int* d_num_runs        = arena_.acquire<int>(1);
    int* d_num_walks_const = arena_.acquire<int>(1);

    // RLE writes only the first num_runs entries; zero the rest so the
    // exclusive-scan extent of num_walks is well-defined.
    CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
        run_lengths, 0, num_walks_ * sizeof(int), stream_));
    CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
        d_num_walks_const, &num_walks_int_, sizeof(int),
        cudaMemcpyHostToDevice, stream_));

    {
        NVTX_RANGE_COLORED("NG step0 sort", nvtx_colors::index_blue);
        cub_sort_pairs(
            start_node_ids, sorted_start_keys,
            iota_src_.data(), sorted_walk_idx,
            num_walks_, stream_);
    }

    {
        NVTX_RANGE_COLORED("NG step0 RLE", nvtx_colors::index_blue);
        cub_run_length_encode(
            sorted_start_keys, unique_start_keys,
            run_lengths, d_num_runs,
            num_walks_, stream_);
        cub_exclusive_sum(run_lengths, run_starts, num_walks_, stream_);
    }

    {
        NVTX_RANGE_COLORED("NG step0 scatter", nvtx_colors::weight_orange);
        const std::size_t zero_blocks =
            (num_walks_ + block_dim_.x - 1) / block_dim_.x;

        zero_int_buffer_kernel
            <<<dim3(static_cast<unsigned>(zero_blocks)),
               block_dim_, 0, stream_>>>(
                walk_to_group_size_.data(), num_walks_int_);

        scatter_walk_group_sizes_kernel
            <<<dim3(static_cast<unsigned>(zero_blocks)),
               block_dim_, 0, stream_>>>(
                sorted_walk_idx,
                run_starts, run_lengths, d_num_runs,
                d_num_walks_const,
                walk_to_group_size_.data(),
                num_walks_int_);
    }
}

NodeGroupedScheduler::StepOutputs NodeGroupedScheduler::run_step(
    WalkSetView walk_set_view,
    const int step_number,
    const int max_walk_len) {

    NVTX_RANGE_COLORED("NG step", nvtx_colors::walk_green);

    arena_.reset();

    uint8_t* alive_flags       = arena_.acquire<uint8_t>(num_walks_);
    int*     active_walk_idx   = arena_.acquire<int>(num_walks_);
    int*     step_num_active   = arena_.acquire<int>(1);
    int*     last_nodes_active = arena_.acquire<int>(num_walks_);
    int*     sorted_last_nodes = arena_.acquire<int>(num_walks_);
    int*     sorted_active_idx = arena_.acquire<int>(num_walks_);
    int*     unique_last_nodes = arena_.acquire<int>(num_walks_);
    int*     step_run_lengths  = arena_.acquire<int>(num_walks_);
    int*     step_run_starts   = arena_.acquire<int>(num_walks_);
    int*     step_num_runs     = arena_.acquire<int>(1);
    int*     step_group_size   = arena_.acquire<int>(num_walks_);

    const std::size_t flag_blocks =
        (num_walks_ + block_dim_.x - 1) / block_dim_.x;

    // (a) filter
    {
        NVTX_RANGE_COLORED("NG filter alive", nvtx_colors::io_grey);
        walk_alive_flags_kernel
            <<<dim3(static_cast<unsigned>(flag_blocks)),
               block_dim_, 0, stream_>>>(
                walk_set_view, step_number, max_walk_len, num_walks_,
                alive_flags);
    }

    // (b) compact
    {
        NVTX_RANGE_COLORED("NG compact", nvtx_colors::io_grey);
        cub_partition_flagged(
            iota_src_.data(), alive_flags,
            active_walk_idx, step_num_active,
            num_walks_, stream_);
    }

    // D2H readback of num_active. CUB sort/RLE/scan need host-side
    // num_items; this is the only way to drive them by num_active.
    int host_num_active = 0;
    {
        NVTX_RANGE_COLORED("NG num_active readback", nvtx_colors::io_grey);
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            &host_num_active, step_num_active, sizeof(int),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream_));
    }

    if (host_num_active <= 0) {
        return StepOutputs{
            /*sorted_walk_idx=*/nullptr,
            /*num_active_device=*/step_num_active,
            /*step_group_size=*/step_group_size,
            /*num_active_host=*/0};
    }

    const std::size_t active_items =
        static_cast<std::size_t>(host_num_active);
    const std::size_t active_blocks =
        (active_items + block_dim_.x - 1) / block_dim_.x;
    const dim3        active_grid(static_cast<unsigned>(active_blocks));

    // (c) gather
    {
        NVTX_RANGE_COLORED("NG gather", nvtx_colors::io_grey);
        gather_last_nodes_kernel
            <<<active_grid, block_dim_, 0, stream_>>>(
                walk_set_view,
                active_walk_idx, step_num_active,
                step_number, max_walk_len,
                last_nodes_active);
    }

    // (d) sort
    {
        NVTX_RANGE_COLORED("NG sort", nvtx_colors::index_blue);
        cub_sort_pairs(
            last_nodes_active, sorted_last_nodes,
            active_walk_idx, sorted_active_idx,
            active_items, stream_);
    }

    // (e) RLE + (f) exclusive-scan
    {
        NVTX_RANGE_COLORED("NG RLE+scan", nvtx_colors::index_blue);
        CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
            step_run_lengths, 0, active_items * sizeof(int), stream_));
        cub_run_length_encode(
            sorted_last_nodes, unique_last_nodes,
            step_run_lengths, step_num_runs,
            active_items, stream_);
        cub_exclusive_sum(
            step_run_lengths, step_run_starts,
            active_items, stream_);
    }

    // (g) scatter
    {
        NVTX_RANGE_COLORED("NG scatter", nvtx_colors::weight_orange);
        // step_group_size is indexed by original walk_idx, so zero the full
        // num_walks extent. Only the active_items scatter writes populate
        // real values this step.
        const std::size_t zero_blocks =
            (num_walks_ + block_dim_.x - 1) / block_dim_.x;
        zero_int_buffer_kernel
            <<<dim3(static_cast<unsigned>(zero_blocks)),
               block_dim_, 0, stream_>>>(
                step_group_size, num_walks_int_);

        scatter_walk_group_sizes_kernel
            <<<active_grid, block_dim_, 0, stream_>>>(
                sorted_active_idx,
                step_run_starts, step_run_lengths, step_num_runs,
                step_num_active,
                step_group_size,
                num_walks_int_);
    }

    return StepOutputs{
        /*sorted_walk_idx=*/sorted_active_idx,
        /*num_active_device=*/step_num_active,
        /*step_group_size=*/step_group_size,
        /*num_active_host=*/host_num_active};
}

}  // namespace temporal_random_walk

#endif  // HAS_CUDA
