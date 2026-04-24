#ifndef NODE_GROUPED_KERNELS_COOP_WARP_CUH
#define NODE_GROUPED_KERNELS_COOP_WARP_CUH

// Warp-tier cooperative kernels:
//   node_grouped_warp_smem_kernel    - G <= warp cap, per-warp panel preload
//   node_grouped_warp_global_kernel  - G >  warp cap, global fallback
//
// Both service non-Node2Vec pickers only; the dispatcher gates Node2Vec
// out of the cooperative pipeline entirely (prev_node-dependent sampling
// can't share a panel).

#include "common.cuh"    // NodeDirPtrs, resolve_node_dir_ptrs,
                         // sample_edge_and_add_hop, coop_warp_smem_g_cap
#include "per_walk.cuh"  // step_kernel_philox_offset (+ transitive philox/utils)
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/cuda_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Warp-smem cooperative kernel.
//
// blockDim.x / 32 warps per block (derived at runtime from the caller's
// block_dim). Each warp services one warp-task (one unique node's walks
// for this step): task_id = blockIdx.x * (blockDim.x/32) + warp_id. Warps
// in a block run independent tasks with no cross-warp coordination —
// their panels are tiled side-by-side in one flat dynamic-smem allocation.
//
// Per-warp panel (size-adjusted by picker-class G cap, 8 × per warp):
//   [0   .. 64)           alignment pad
//   [64  .. 64 + G*8)     s_group_offsets (size_t)
//   [.. + G*8)            s_first_ts      (int64_t)
//
// Task-level scalars are NOT broadcast via smem — task_id is warp-
// uniform, so lane-level broadcast loads cover it in a single transaction.
//
// Sync discipline: only __syncwarp() — never __syncthreads(). Warps in
// the same block run different tasks and may diverge at the task-id
// guard (partial last block when num_tasks % 8 != 0). __syncthreads()
// would deadlock against the idle warps. Each __syncwarp() sees only
// this warp's 32 lanes; all lanes are in lockstep at the sync points.
//
// Weighted pickers still read cum_weights from global (see block-smem).
// ==========================================================================
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_smem_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_nodes,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (step_number >= max_walk_len - 1) return;

    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int lane_id = static_cast<int>(threadIdx.x & 31u);
    const int task_id =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5)) + warp_id;

    // All 32 lanes of an out-of-range warp return together — task_id is
    // warp-uniform, so the guard is a 32-lane-wide return.
    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    // Task-level scalars. Broadcast loads — task_id is warp-uniform.
    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    // Per-warp panel layout.
    constexpr int kGCap = coop_warp_smem_g_cap<EdgePickerType>();
    constexpr size_t kHeaderBytes        = 64;
    constexpr size_t kGroupOffsetsOffset = kHeaderBytes;
    constexpr size_t kFirstTsOffset      =
        kGroupOffsetsOffset + sizeof(size_t) * kGCap;
    constexpr size_t kPerWarpBytes =
        kFirstTsOffset + sizeof(int64_t) * kGCap;

    extern __shared__ unsigned char s_pool[];
    unsigned char* const s_my =
        s_pool + static_cast<size_t>(warp_id) * kPerWarpBytes;
    size_t* const  s_group_offsets =
        reinterpret_cast<size_t*>(s_my + kGroupOffsetsOffset);
    int64_t* const s_first_ts      =
        reinterpret_cast<int64_t*>(s_my + kFirstTsOffset);

    // Stage 1: warp-cooperative preload of the G-sized panel. Lane l
    // handles indices l, l+32, l+64, ... At G=340 that's 11 rounds/lane.
    for (int p = lane_id; p < G; p += 32) {
        const size_t ts_group_offset =
            ptrs.node_ts_groups_offsets[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        s_first_ts[p] =
            view.timestamps[ptrs.node_ts_sorted_indices[ts_group_offset]];
    }
    __syncwarp();

    // Stage 2: intra-warp stride over this task's walks. At W=T_BLOCK
    // (255), ⌈255/32⌉ = 8 rounds/lane.
    for (int k = lane_id; k < walk_count; k += 32) {
        const size_t walk_idx =
            static_cast<size_t>(sorted_walk_idx[walk_start + k]);
        const size_t offset   =
            walk_idx * static_cast<size_t>(max_walk_len)
            + static_cast<size_t>(step_number);
        const int64_t last_ts = walk_set.timestamps[offset];

        PhiloxState rng;
        init_philox_state(rng, base_seed, walk_idx,
                          step_kernel_philox_offset(step_number));
        const double r_group = draw_u01_philox(rng);
        const double r_edge  = draw_u01_philox(rng);

        // Stage 2a: fast-path binary search against the warp's smem panel.
        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            s_group_offsets, s_first_ts,
            /*node_ts_sorted_indices=*/nullptr,
            /*view_timestamps=*/nullptr,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, r_group);
        if (local_pos == -1) continue;

        // Stage 2b+c: edge range via smem offsets + pick + hop.
        sample_edge_and_add_hop<IsDirected, Forward>(
            view, walk_set, ptrs,
            s_group_offsets, local_pos, G, node_edge_end, node_id,
            walk_idx, r_edge);
    }
}

// Dynamic smem size for warp-smem: per-warp panel × warps_per_block. The
// per-warp footprint is compile-time (picker class); warps_per_block comes
// from the launcher's block_dim so this scales with the caller's block size.
template <RandomPickerType PickerType>
HOST inline size_t warp_smem_dynamic_smem_bytes(const dim3& block_dim) {
    constexpr size_t kGCap = static_cast<size_t>(
        coop_warp_smem_g_cap<PickerType>());
    constexpr size_t kPerWarp =
        size_t{64} + (sizeof(size_t) + sizeof(int64_t)) * kGCap;
    return kPerWarp * static_cast<size_t>(block_dim.x >> 5);
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_smem_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_nodes,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        const size_t smem_bytes = warp_smem_dynamic_smem_bytes<kEdge>(block_dim);
        node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, smem_bytes, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// ==========================================================================
// Warp-global cooperative kernel.
//
// Same launch topology as warp-smem (8 warps/block, one task per warp),
// but no panel preload. Services nodes with G > G_THRESHOLD_WARP_*
// whose metadata wouldn't fit in the per-warp slice. Binary search runs
// against the GLOBAL node_ts_groups_offsets slice via find_group_pos_slice's
// double-indirect comparator. Cooperative win: 32 lanes share L1/L2 cache
// residency of the per-node arrays.
//
// No smem, no sync: task_id is warp-uniform so every lane reading
// node_walk_nodes[task_id] etc. coalesces into broadcast loads. There's
// no preload stage, so there's nothing to sync on either.
// ==========================================================================
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_global_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_nodes,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (step_number >= max_walk_len - 1) return;

    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int lane_id = static_cast<int>(threadIdx.x & 31u);
    const int task_id =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5)) + warp_id;

    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    // Task-level scalars. Broadcast loads — task_id is warp-uniform.
    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    // Slice pointer into the GLOBAL ts-group-offsets array.
    const size_t* group_offsets_slice =
        ptrs.node_ts_groups_offsets + node_group_begin;

    // Intra-warp stride over this task's walks.
    for (int k = lane_id; k < walk_count; k += 32) {
        const size_t walk_idx =
            static_cast<size_t>(sorted_walk_idx[walk_start + k]);
        const size_t offset   =
            walk_idx * static_cast<size_t>(max_walk_len)
            + static_cast<size_t>(step_number);
        const int64_t last_ts = walk_set.timestamps[offset];

        PhiloxState rng;
        init_philox_state(rng, base_seed, walk_idx,
                          step_kernel_philox_offset(step_number));
        const double r_group = draw_u01_philox(rng);
        const double r_edge  = draw_u01_philox(rng);

        // Stage 1: group pick via the global double-indirect comparator.
        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            group_offsets_slice,
            /*first_ts=*/nullptr,
            ptrs.node_ts_sorted_indices, view.timestamps,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, r_group);
        if (local_pos == -1) continue;

        // Stage 2b+c: edge range (from the same global slice) + pick + hop.
        sample_edge_and_add_hop<IsDirected, Forward>(
            view, walk_set, ptrs,
            group_offsets_slice, local_pos, G, node_edge_end, node_id,
            walk_idx, r_edge);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_global_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_nodes,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_warp_global_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // NODE_GROUPED_KERNELS_COOP_WARP_CUH
