#ifndef NODE_GROUPED_KERNELS_COOP_BLOCK_CUH
#define NODE_GROUPED_KERNELS_COOP_BLOCK_CUH

// Block-tier cooperative kernels:
//   node_grouped_block_smem_kernel    - G <= block cap, smem panel preload
//   node_grouped_block_global_kernel  - G >  block cap, global fallback
//
// Both service non-Node2Vec pickers only; the dispatcher gates Node2Vec
// out of the cooperative pipeline entirely (prev_node-dependent sampling
// can't share a panel).

#include "common.cuh"    // NodeDirPtrs, resolve_node_dir_ptrs,
                         // sample_edge_and_add_hop
#include "per_walk.cuh"  // step_kernel_philox_offset (+ transitive philox/utils)
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/cuda_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Block-smem cooperative kernel.
//
// One block per block-task, blockDim.x threads
// per block. The block cooperatively preloads the per-node G-sized panel
// (s_group_offsets[G] and s_first_ts[G]) into shared memory once, then all
// 256 threads stride through the task's walks sampling edges against it.
//
// Smem layout (dynamic `extern __shared__`, size = 64B pad + G*(size_t + int64_t)):
//   [ 0 ..  64)              alignment pad (keeps smem panel 16B-aligned)
//   [64 ..  64 + G*8)        s_group_offsets (size_t)
//   [.. + G*8)               s_first_ts      (int64_t)
// Sizing uses G_THRESHOLD_BLOCK_{INDEX,WEIGHT} by picker class;
// the scheduler's G-partition guarantees G <= cap.
//
// Task-level scalars (node_id, walk_start, walk_count, G, node_edge_end,
// node_group_begin) are NOT broadcast through smem — task_id is block-
// uniform so every thread reading them from global coalesces into a single
// broadcast load. Saves one __syncthreads and one smem header.
//
// Weighted pickers still read the cumulative-weight array from global;
// preloading cum_weights would require reworking the CDF offset math and
// is deferred. The main smem win is killing the double-indirect load
// chain on every binary-search probe — driven by first_ts.
// ==========================================================================
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_block_smem_kernel(
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
    const int task_id = blockIdx.x;
    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    // Task-level scalars: task_id is block-uniform -> each global load
    // broadcasts to all 256 threads via one memory transaction.
    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    // Smem panel layout.
    constexpr int kGCap = coop_block_smem_g_cap<EdgePickerType>();
    constexpr size_t kHeaderBytes        = 64;
    constexpr size_t kGroupOffsetsOffset = kHeaderBytes;
    constexpr size_t kFirstTsOffset      =
        kGroupOffsetsOffset + sizeof(size_t) * kGCap;

    extern __shared__ unsigned char s_panel[];
    size_t* const  s_group_offsets =
        reinterpret_cast<size_t*>(s_panel + kGroupOffsetsOffset);
    int64_t* const s_first_ts      =
        reinterpret_cast<int64_t*>(s_panel + kFirstTsOffset);

    // Stage 1: cooperative preload of the G-sized panel. Stride by
    // blockDim.x — each thread handles ⌈G/blockDim.x⌉ entries.
    for (int p = threadIdx.x; p < G; p += blockDim.x) {
        const size_t ts_group_offset =
            ptrs.node_ts_groups_offsets[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        // Preload first_ts to kill the double-indirect load chain on
        // every binary-search probe in stage 2a.
        s_first_ts[p] =
            view.timestamps[ptrs.node_ts_sorted_indices[ts_group_offset]];
    }
    __syncthreads();

    // Stage 2: stride loop over this task's walks.
    for (int k = threadIdx.x; k < walk_count; k += blockDim.x) {
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

        // Stage 2a: group pick against the smem panel (first_ts non-null
        // -> fast-path single-load comparator).
        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            s_group_offsets, s_first_ts,
            /*node_ts_sorted_indices=*/nullptr,
            /*view_timestamps=*/nullptr,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, r_group);
        if (local_pos == -1) continue;

        // Stage 2b+c: resolve edge range via smem offsets, pick edge,
        // write the hop. Shared helper in common.cuh.
        sample_edge_and_add_hop<IsDirected, Forward>(
            view, walk_set, ptrs,
            s_group_offsets, local_pos, G, node_edge_end, node_id,
            walk_idx, r_edge);
    }
}

// Dynamic smem size for block-smem: 64-byte alignment pad + G_CAP entries
// each of (size_t + int64_t) = 16 bytes. Chosen at compile time from the
// picker class so the launch matches the kernel's panel layout.
template <RandomPickerType PickerType>
HOST constexpr inline size_t block_smem_dynamic_smem_bytes() {
    constexpr size_t kGCap = static_cast<size_t>(
        coop_block_smem_g_cap<PickerType>());
    return size_t{64} + (sizeof(size_t) + sizeof(int64_t)) * kGCap;
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_block_smem_kernel(
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
        constexpr size_t kSmemBytes = block_smem_dynamic_smem_bytes<kEdge>();
        node_grouped_block_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, kSmemBytes, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// ==========================================================================
// Block-global cooperative kernel.
//
// Same launch topology as block-smem (one block per block-task, 256 threads
// per block); services nodes with G > G_THRESHOLD_BLOCK_* whose
// panel wouldn't fit in smem. Binary search runs against the GLOBAL
// node_ts_groups_offsets slice via find_group_pos_slice's double-indirect
// comparator. The cooperative win is L1/L2 cache residency of the per-node
// arrays shared across the 256 threads — each walk does NOT pay its own
// scatter read on a different node, as it would in solo.
//
// No broadcast stage: task_id is block-uniform, so every thread's read of
// node_walk_nodes[task_id] etc. coalesces into a broadcast load. No smem,
// no __syncthreads. The per-walk work is entirely independent.
// ==========================================================================
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_block_global_kernel(
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
    const int task_id = blockIdx.x;
    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    // Task-level scalars. Broadcast loads — task_id is block-uniform.
    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    // Slice pointer into the GLOBAL ts-group-offsets array. Each binary-
    // search probe reads via the double-indirect comparator inside
    // find_group_pos_slice (first_ts == nullptr triggers the fallback).
    const size_t* group_offsets_slice =
        ptrs.node_ts_groups_offsets + node_group_begin;

    for (int k = threadIdx.x; k < walk_count; k += blockDim.x) {
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
inline void dispatch_node_grouped_block_global_kernel(
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
        node_grouped_block_global_kernel<IsDirected, Forward, kEdge>
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

#endif // NODE_GROUPED_KERNELS_COOP_BLOCK_CUH
