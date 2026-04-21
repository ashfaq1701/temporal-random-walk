#include "temporal_random_walk_node_grouped_scheduler.cuh"

#ifdef HAS_CUDA

#include <cstddef>
#include <cstdint>

#include "../common/cuda_sort.cuh"
#include "../common/cuda_scan.cuh"
#include "../common/error_handlers.cuh"
#include "../common/nvtx.cuh"
#include "../common/warp_coop_config.cuh"
#include "../random/pickers.cuh"

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

// W-partition: classify each unique node (RLE run) by its walk count into
// solo / warp / block tier and append it to the appropriate output list
// via atomicAdd on a shared counters[3] array.
//
// Atomic contention is per-bucket, not cross-bucket; on typical workloads
// each tier's atomic stream is shallow. Output order within each tier is
// non-deterministic — that's fine because the cooperative kernels iterate
// their task lists independently.
__global__ void partition_by_w_kernel(
    const int* __restrict__ unique_nodes,
    const int* __restrict__ run_starts,
    const int* __restrict__ run_lengths,
    const int* __restrict__ num_runs_ptr,
    const int* __restrict__ sorted_walk_idx,
    const int t_warp,                           // solo / warp boundary (W<=t_warp is solo)
    const int t_block,                          // warp / block boundary (W<=t_block is warp)
    int* __restrict__ solo_walks,
    int* __restrict__ warp_nodes,
    int* __restrict__ warp_walk_starts,
    int* __restrict__ warp_walk_counts,
    int* __restrict__ block_nodes,
    int* __restrict__ block_walk_starts,
    int* __restrict__ block_walk_counts,
    int* __restrict__ counters) {   // int[3]: {num_solo, num_warp, num_block}

    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= *num_runs_ptr) return;

    const int W = run_lengths[r];
    const int start = run_starts[r];
    const int node = unique_nodes[r];

    if (W <= t_warp) {
        // W == 1 (solo). Single walk, copied directly.
        const int idx = atomicAdd(&counters[0], 1);
        solo_walks[idx] = sorted_walk_idx[start];
    } else if (W <= t_block) {
        const int idx = atomicAdd(&counters[1], 1);
        warp_nodes[idx]       = node;
        warp_walk_starts[idx] = start;
        warp_walk_counts[idx] = W;
    } else {
        const int idx = atomicAdd(&counters[2], 1);
        block_nodes[idx]       = node;
        block_walk_starts[idx] = start;
        block_walk_counts[idx] = W;
    }
}

// G-partition: splits one tier's task list into (smem, global) variants
// based on per-node G (distinct-timestamp-group count). G <= g_cap fits
// the smem panel; G > g_cap routes to the global-fallback kernel.
//
// Called twice per step — once for the warp tier, once for the block tier,
// each with its own g_cap (index vs weighted picker class resolved by
// run_step's caller).
__global__ void partition_by_g_kernel(
    const int* __restrict__ tier_nodes,
    const int* __restrict__ tier_walk_starts,
    const int* __restrict__ tier_walk_counts,
    const int* __restrict__ num_tier_tasks_ptr,
    const std::size_t* __restrict__ count_ts_group_per_node,
    const int g_cap,
    int* __restrict__ smem_nodes,
    int* __restrict__ smem_walk_starts,
    int* __restrict__ smem_walk_counts,
    int* __restrict__ global_nodes,
    int* __restrict__ global_walk_starts,
    int* __restrict__ global_walk_counts,
    int* __restrict__ g_counters) {     // int[2]: {num_smem, num_global}

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tier_tasks_ptr) return;

    const int node_id    = tier_nodes[t];
    const int walk_start = tier_walk_starts[t];
    const int walk_count = tier_walk_counts[t];

    // G = number of distinct timestamp groups at node_id (directional —
    // count_ts_group_per_node is the caller's already-resolved array).
    const std::size_t group_begin = count_ts_group_per_node[node_id];
    const std::size_t group_end   = count_ts_group_per_node[node_id + 1];
    const int G = static_cast<int>(group_end - group_begin);

    if (G <= g_cap) {
        const int idx = atomicAdd(&g_counters[0], 1);
        smem_nodes[idx]        = node_id;
        smem_walk_starts[idx]  = walk_start;
        smem_walk_counts[idx]  = walk_count;
    } else {
        const int idx = atomicAdd(&g_counters[1], 1);
        global_nodes[idx]        = node_id;
        global_walk_starts[idx]  = walk_start;
        global_walk_counts[idx]  = walk_count;
    }
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
      iota_src_(/*use_gpu=*/true) {

    iota_src_.resize(num_walks);

    // iota_src_ is the reusable input to cub_partition_flagged in every
    // intermediate step. Seed once on the caller's stream.
    const std::size_t iota_blocks =
        (num_walks + block_dim_.x - 1) / block_dim_.x;
    iota_int_kernel<<<dim3(static_cast<unsigned>(iota_blocks)),
                      block_dim_, 0, stream_>>>(
        iota_src_.data(), num_walks_int_);
}

NodeGroupedScheduler::StepOutputs NodeGroupedScheduler::run_step(
    WalkSetView walk_set_view,
    const int step_number,
    const int max_walk_len,
    const std::size_t* count_ts_group_per_node,
    const RandomPickerType edge_picker_type) {

    NVTX_RANGE_COLORED("NG step", nvtx_colors::walk_green);

    arena_.reset();

    // Per-stage scratch.
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

    // W-partition outputs. The warp/block lists here are INTERMEDIATE —
    // they feed the two G-partition passes below. Only solo_walks and the
    // four (smem|global) tier lists from the G-partition are exposed in
    // StepOutputs. All upper-bound-sized to num_walks.
    int*     solo_walks         = arena_.acquire<int>(num_walks_);
    int*     warp_nodes_w       = arena_.acquire<int>(num_walks_);
    int*     warp_walk_starts_w = arena_.acquire<int>(num_walks_);
    int*     warp_walk_counts_w = arena_.acquire<int>(num_walks_);
    int*     block_nodes_w      = arena_.acquire<int>(num_walks_);
    int*     block_walk_starts_w= arena_.acquire<int>(num_walks_);
    int*     block_walk_counts_w= arena_.acquire<int>(num_walks_);

    // G-partition outputs for the warp tier.
    int*     warp_smem_nodes    = arena_.acquire<int>(num_walks_);
    int*     warp_smem_starts   = arena_.acquire<int>(num_walks_);
    int*     warp_smem_counts   = arena_.acquire<int>(num_walks_);
    int*     warp_global_nodes  = arena_.acquire<int>(num_walks_);
    int*     warp_global_starts = arena_.acquire<int>(num_walks_);
    int*     warp_global_counts = arena_.acquire<int>(num_walks_);

    // G-partition outputs for the block tier.
    int*     block_smem_nodes    = arena_.acquire<int>(num_walks_);
    int*     block_smem_starts   = arena_.acquire<int>(num_walks_);
    int*     block_smem_counts   = arena_.acquire<int>(num_walks_);
    int*     block_global_nodes  = arena_.acquire<int>(num_walks_);
    int*     block_global_starts = arena_.acquire<int>(num_walks_);
    int*     block_global_counts = arena_.acquire<int>(num_walks_);

    // Counter layout (int[7]) — single contiguous block for one D2H:
    //   [0] num_solo        (from W-partition)
    //   [1] num_warp_w      (W-partition intermediate, consumed by G-partition)
    //   [2] num_block_w     (W-partition intermediate)
    //   [3] num_warp_smem   (warp G-partition output)
    //   [4] num_warp_global (warp G-partition output)
    //   [5] num_block_smem  (block G-partition output)
    //   [6] num_block_global (block G-partition output)
    constexpr int kCounterSlots = 7;
    int* tier_counters = arena_.acquire<int>(kCounterSlots);

    // Pick G caps by picker class (runtime — switch lifts to compile time
    // via template specialization in the real coop bodies, tasks 8–11).
    const bool is_index_picker =
        random_pickers::is_index_based_picker(edge_picker_type);
    const int g_cap_warp = is_index_picker
        ? TRW_NODE_GROUPED_G_CAP_WARP_INDEX
        : TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED;
    const int g_cap_block = is_index_picker
        ? TRW_NODE_GROUPED_G_CAP_BLOCK_INDEX
        : TRW_NODE_GROUPED_G_CAP_BLOCK_WEIGHTED;

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

    // D2H readback of num_active (drives CUB sort/RLE/scan extents).
    int host_num_active = 0;
    {
        NVTX_RANGE_COLORED("NG num_active readback", nvtx_colors::io_grey);
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            &host_num_active, step_num_active, sizeof(int),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream_));
    }

    // Zero all tier counters up front; partition kernels atomic-add.
    CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
        tier_counters, 0, kCounterSlots * sizeof(int), stream_));

    auto empty_tier = [&](const int device_counter_slot) {
        // Empty TierTaskList (num_tasks_host=0). Nodes/starts/counts
        // pointers can be nullptr — caller gates on num_tasks_host.
        return TierTaskList{
            /*nodes=*/nullptr,
            /*walk_starts=*/nullptr,
            /*walk_counts=*/nullptr,
            /*num_tasks_device=*/&tier_counters[device_counter_slot],
            /*num_tasks_host=*/0};
    };

    if (host_num_active <= 0) {
        return StepOutputs{
            /*sorted_walk_idx=*/nullptr,
            /*num_active_host=*/0,
            /*solo_walks=*/solo_walks,
            /*num_solo_walks_device=*/&tier_counters[0],
            /*num_solo_walks_host=*/0,
            /*warp_smem=*/  empty_tier(3),
            /*warp_global=*/empty_tier(4),
            /*block_smem=*/ empty_tier(5),
            /*block_global=*/empty_tier(6)};
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

    // (g) W-partition: classify each run into solo / warp / block.
    // Counters into tier_counters[0..2].
    {
        NVTX_RANGE_COLORED("NG W-partition", nvtx_colors::weight_orange);
        partition_by_w_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            unique_last_nodes, step_run_starts, step_run_lengths,
            step_num_runs, sorted_active_idx,
            TRW_NODE_GROUPED_T_WARP,
            TRW_NODE_GROUPED_T_BLOCK,
            solo_walks,
            warp_nodes_w,  warp_walk_starts_w,  warp_walk_counts_w,
            block_nodes_w, block_walk_starts_w, block_walk_counts_w,
            tier_counters);   // int[3] slots [0..2]
    }

    // (h) G-partition: warp tier -> (warp_smem, warp_global).
    // Grid upper-bound is active_grid; kernel threads with
    // t >= *tier_counters[1] early-exit. Counters into tier_counters[3..4].
    {
        NVTX_RANGE_COLORED("NG G-partition (warp)", nvtx_colors::weight_orange);
        partition_by_g_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            warp_nodes_w, warp_walk_starts_w, warp_walk_counts_w,
            &tier_counters[1],   // num_warp_w
            count_ts_group_per_node,
            g_cap_warp,
            warp_smem_nodes,  warp_smem_starts,  warp_smem_counts,
            warp_global_nodes, warp_global_starts, warp_global_counts,
            &tier_counters[3]);  // int[2] slots [3..4]
    }

    // (i) G-partition: block tier -> (block_smem, block_global).
    {
        NVTX_RANGE_COLORED("NG G-partition (block)", nvtx_colors::weight_orange);
        partition_by_g_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            block_nodes_w, block_walk_starts_w, block_walk_counts_w,
            &tier_counters[2],   // num_block_w
            count_ts_group_per_node,
            g_cap_block,
            block_smem_nodes,  block_smem_starts,  block_smem_counts,
            block_global_nodes, block_global_starts, block_global_counts,
            &tier_counters[5]);  // int[2] slots [5..6]
    }

    // Final per-step D2H: read all seven counters in one copy.
    int counts_host[kCounterSlots] = {};
    {
        NVTX_RANGE_COLORED("NG tier-count readback", nvtx_colors::io_grey);
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            counts_host, tier_counters, kCounterSlots * sizeof(int),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream_));
    }

    return StepOutputs{
        /*sorted_walk_idx=*/sorted_active_idx,
        /*num_active_host=*/host_num_active,
        /*solo_walks=*/solo_walks,
        /*num_solo_walks_device=*/&tier_counters[0],
        /*num_solo_walks_host=*/counts_host[0],
        /*warp_smem=*/TierTaskList{
            warp_smem_nodes, warp_smem_starts, warp_smem_counts,
            &tier_counters[3], counts_host[3]},
        /*warp_global=*/TierTaskList{
            warp_global_nodes, warp_global_starts, warp_global_counts,
            &tier_counters[4], counts_host[4]},
        /*block_smem=*/TierTaskList{
            block_smem_nodes, block_smem_starts, block_smem_counts,
            &tier_counters[5], counts_host[5]},
        /*block_global=*/TierTaskList{
            block_global_nodes, block_global_starts, block_global_counts,
            &tier_counters[6], counts_host[6]}};
}

}  // namespace temporal_random_walk

#endif  // HAS_CUDA
