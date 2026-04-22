#ifndef NODE_GROUPED_KERNELS_COOP_WARP_CUH
#define NODE_GROUPED_KERNELS_COOP_WARP_CUH

// Warp-tier cooperative kernels: smem (G <= warp cap, per-warp panel
// preload) and global fallback.
//
//   task 10 done -> node_grouped_warp_smem_kernel   (real cooperative body)
//   task 11 TODO -> node_grouped_warp_global_kernel (currently scaffold)

#include "common.cuh"    // NodeDirPtrs, resolve_node_dir_ptrs
#include "per_walk.cuh"  // advance_one_walk, philox offset helper
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/warp_coop_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Warp-smem cooperative kernel (task 10 — real body).
//
// TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK (8) warps per block,
// TRW_NODE_GROUPED_COOP_BLOCK_THREADS  (256) threads total. Each warp
// services one warp-task (one unique node's walks for this step);
// task_id = blockIdx.x * 8 + warp_id. Warps inside a block run
// independent tasks with no cross-warp coordination — their panels
// are tiled side-by-side in one flat dynamic-smem allocation.
//
// Smem layout (dynamic `extern __shared__`, size = 8 × per_warp_bytes):
//   warp w's slice lives at  s_pool + w * kPerWarpBytes
//   inside each slice (same shape as block-smem, just smaller G cap):
//     [ 0 ..  16)  s_header[4]: node_id, walk_start, walk_count, G
//     [16 ..  24)  s_node_edge_end (= node_edge_offsets[node+1])
//     [24 ..  64)  alignment padding
//     [64 .. 64 + G*8)   s_group_offsets (size_t)
//     [.. + G*8)         s_first_ts     (int64_t)
//
// G cap for this tier:
//   TRW_NODE_GROUPED_G_CAP_WARP_INDEX    (340) for index pickers
//   TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED (220) for weighted pickers
//   Scheduler's G-partition guarantees G <= cap.
//
// Per-warp byte budget at worst (index): 64 + 340*(8+8) = 5504 B/warp.
// 8 warps: 44 032 B/block — fits the static 48 KB envelope, no
// cudaFuncSetAttribute opt-in required.
//
// Weighted pickers still read cum_weights from global (see block-smem).
// Node2Vec short-circuits to per-walk processing inside the warp (its
// prev_node-dependent picker breaks panel sharing).
//
// Sync discipline: only __syncwarp() — never __syncthreads(). Warps in
// the same block run different tasks and may diverge at the task-id
// guard (partial last block when num_tasks % 8 != 0). A __syncthreads()
// would deadlock against the idle warps. Each __syncwarp() sees only
// the 32 lanes of one warp; those lanes are always in lockstep at the
// sync points because all 32 either return at the guard or all 32
// continue.
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
        static_cast<int>(blockIdx.x) * TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK
        + warp_id;

    // All 32 lanes of an out-of-range warp return together — task_id is
    // warp-uniform, so divergence is 32-lane wide and safe.
    if (task_id >= *num_tasks_ptr) return;

    // Node2Vec: per-walk prev_node bias breaks panel sharing; distribute
    // walks across this warp's 32 lanes and fall through advance_one_walk.
    if constexpr (EdgePickerType == RandomPickerType::TemporalNode2Vec) {
        const int walk_start = node_walk_starts[task_id];
        const int walk_count = node_walk_counts[task_id];
        for (int k = lane_id; k < walk_count; k += 32) {
            advance_one_walk<IsDirected, Forward, EdgePickerType>(
                view, walk_set, sorted_walk_idx[walk_start + k],
                step_number, max_walk_len, base_seed);
        }
        return;
    }

    // Resolve direction-dependent per-graph pointers once per task.
    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    // Per-warp panel layout, sized from the picker-class G cap.
    constexpr bool kIsIndexPicker =
        random_pickers::is_index_based_picker_v<EdgePickerType>;
    constexpr int kGCap = kIsIndexPicker
        ? TRW_NODE_GROUPED_G_CAP_WARP_INDEX
        : TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED;
    constexpr size_t kHeaderBytes        = 64;
    constexpr size_t kGroupOffsetsOffset = kHeaderBytes;
    constexpr size_t kFirstTsOffset      =
        kGroupOffsetsOffset + sizeof(size_t) * kGCap;
    constexpr size_t kPerWarpBytes =
        kFirstTsOffset + sizeof(int64_t) * kGCap;

    extern __shared__ unsigned char s_pool[];
    unsigned char* const s_my =
        s_pool + static_cast<size_t>(warp_id) * kPerWarpBytes;
    int* const      s_header        = reinterpret_cast<int*>(s_my);
    size_t* const   s_node_edge_end = reinterpret_cast<size_t*>(s_my + 16);
    size_t* const   s_group_offsets =
        reinterpret_cast<size_t*>(s_my + kGroupOffsetsOffset);
    int64_t* const  s_first_ts      =
        reinterpret_cast<int64_t*>(s_my + kFirstTsOffset);

    // Stage 0: lane 0 of THIS warp broadcasts the task header into the
    // warp's private smem slice.
    if (lane_id == 0) {
        const int node_id    = node_walk_nodes[task_id];
        const int walk_start = node_walk_starts[task_id];
        const int walk_count = node_walk_counts[task_id];
        const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
        const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
        s_header[0] = node_id;
        s_header[1] = walk_start;
        s_header[2] = walk_count;
        s_header[3] = static_cast<int>(node_group_end - node_group_begin);
        *s_node_edge_end = ptrs.node_edge_offsets[node_id + 1];
    }
    __syncwarp();

    const int    node_id    = s_header[0];
    const int    walk_start = s_header[1];
    const int    walk_count = s_header[2];
    const int    G          = s_header[3];
    const size_t node_edge_end_global = *s_node_edge_end;
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];

    // Stage 1: warp-cooperative preload of this task's G-sized panel.
    // Lane l handles indices l, l+32, l+64, ... At G=340 that's
    // ⌈340/32⌉ = 11 rounds per lane in the worst case.
    for (int p = lane_id; p < G; p += 32) {
        const size_t ts_group_offset =
            ptrs.node_ts_groups_offsets[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        // Kill the double-indirect load chain on every binary-search
        // probe in stage 2's find_group_pos_slice.
        s_first_ts[p] =
            view.timestamps[ptrs.node_ts_sorted_indices[ts_group_offset]];
    }
    __syncwarp();

    // Stage 2: intra-warp stride over the walks in this task's slice.
    // Up to ⌈255/32⌉ = 8 rounds per lane at W=T_BLOCK.
    for (int k = lane_id; k < walk_count; k += 32) {
        const int walk_idx_int = sorted_walk_idx[walk_start + k];
        if (walk_idx_int < 0) continue;

        const size_t walk_idx = static_cast<size_t>(walk_idx_int);
        const size_t offset   =
            walk_idx * static_cast<size_t>(max_walk_len)
            + static_cast<size_t>(step_number);
        const int64_t last_ts = walk_set.timestamps[offset];
        const int prev_node   =
            step_number > 0 ? walk_set.nodes[offset - 1] : -1;

        PhiloxState rng;
        init_philox_state(rng, base_seed, walk_idx,
                          step_kernel_philox_offset(step_number));
        const double r_group = draw_u01_philox(rng);
        const double r_edge  = draw_u01_philox(rng);

        // Stage 2a: fast-path binary search against this warp's smem
        // panel (first_ts non-null -> single-load comparator).
        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            s_group_offsets,
            s_first_ts,
            /*node_ts_sorted_indices=*/nullptr,
            /*view_timestamps=*/nullptr,
            ptrs.weights, ptrs.weights_size,
            node_group_begin,
            G, last_ts, r_group);
        (void)prev_node;  // Node2Vec short-circuited above; non-Node2Vec
                          // pickers don't use prev_node.
        if (local_pos == -1) continue;

        // Stage 2b: group -> edge range via the warp's smem offsets.
        const size_t valid_edge_start = s_group_offsets[local_pos];
        const size_t valid_edge_end =
            (local_pos + 1 < G)
                ? s_group_offsets[local_pos + 1]
                : node_edge_end_global;
        if (valid_edge_start >= valid_edge_end) continue;

        // Stage 2c: uniform random edge in the group; fetch endpoint +
        // timestamp from the global view.
        const long edge_idx = static_cast<long>(ptrs.node_ts_sorted_indices[
            valid_edge_start +
            generate_random_number_bounded_by(
                static_cast<int>(valid_edge_end - valid_edge_start),
                r_edge)]);

        if constexpr (IsDirected) {
            walk_set.add_hop(walk_idx,
                             Forward ? view.targets[edge_idx]
                                     : view.sources[edge_idx],
                             view.timestamps[edge_idx],
                             edge_idx);
        } else {
            const int next_node = pick_other_number(
                view.sources[edge_idx], view.targets[edge_idx], node_id);
            walk_set.add_hop(walk_idx, next_node,
                             view.timestamps[edge_idx], edge_idx);
        }
    }
}

// Dynamic smem size for warp-smem: per-warp (64-byte header + G_CAP
// entries of (size_t + int64_t)) × TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK.
// Chosen at compile time from the picker class so the launch matches
// the kernel's layout.
template <RandomPickerType PickerType>
HOST constexpr inline size_t warp_smem_dynamic_smem_bytes() {
    constexpr bool kIsIndexPicker =
        random_pickers::is_index_based_picker_v<PickerType>;
    constexpr size_t kGCap = kIsIndexPicker
        ? static_cast<size_t>(TRW_NODE_GROUPED_G_CAP_WARP_INDEX)
        : static_cast<size_t>(TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED);
    constexpr size_t kPerWarp =
        size_t{64} + sizeof(size_t) * kGCap + sizeof(int64_t) * kGCap;
    return kPerWarp
        * static_cast<size_t>(TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK);
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
        constexpr size_t kSmemBytes = warp_smem_dynamic_smem_bytes<kEdge>();
        node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, kSmemBytes, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// ==========================================================================
// Warp-global cooperative kernel (task 11 — scaffold).
//
// Services warp tasks whose G exceeds TRW_NODE_GROUPED_G_CAP_WARP_*. The
// scaffold runs one thread per task looping advance_one_walk. Real
// cooperative body lands with task 11 — warp equivalent of block-global:
// 8 warps/block, one task per warp, no panel preload, double-indirect
// binary search against global arrays via find_group_pos_slice's
// fallback path.
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
    // SCAFFOLD (task 3): per-node sequential loop. Real warp-global body in task 11.
    (void)node_walk_nodes;  // unused in the scaffold; task 11 consumes it.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tasks_ptr) return;

    const int walk_start = node_walk_starts[t];
    const int walk_count = node_walk_counts[t];
    for (int k = 0; k < walk_count; ++k) {
        advance_one_walk<IsDirected, Forward, EdgePickerType>(
            view, walk_set, sorted_walk_idx[walk_start + k],
            step_number, max_walk_len, base_seed);
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
