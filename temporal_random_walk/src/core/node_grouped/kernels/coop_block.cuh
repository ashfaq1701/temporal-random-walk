#ifndef NODE_GROUPED_KERNELS_COOP_BLOCK_CUH
#define NODE_GROUPED_KERNELS_COOP_BLOCK_CUH

// Block-tier cooperative kernels: smem (G <= block cap, cooperative preload)
// and global fallback (still scaffold as of task 8).
//
//   task 8 done -> node_grouped_block_smem_kernel   (real cooperative body)
//   task 9 TODO -> node_grouped_block_global_kernel (currently scaffold)

#include "per_walk.cuh"  // advance_one_walk, philox offset helper
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/warp_coop_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Block-smem cooperative kernel (task 8 — real body).
//
// One block per block-task, TRW_NODE_GROUPED_COOP_BLOCK_THREADS (256) threads
// per block. The block cooperatively preloads the per-node G-sized panel
// (s_group_offsets[G] and s_first_ts[G]) into shared memory once, then all
// 256 threads stride through the task's walk_count walks sampling edges
// against the smem panel.
//
// Smem layout (dynamic `extern __shared__`, size chosen at launch):
//   [ 0 ..  16)                    s_header[4]: node_id, walk_start,
//                                                walk_count, G
//   [16 ..  24)                    s_node_edge_end (= node_edge_offsets[node+1])
//   [24 ..  64)                    alignment padding
//   [64 .. 64 + G*8)               s_group_offsets (size_t)
//   [.. + G*8)                     s_first_ts     (int64_t)
// Sizing uses TRW_NODE_GROUPED_G_CAP_BLOCK_{INDEX,WEIGHTED} depending on
// picker class; the scheduler's G-partition guarantees G <= cap.
//
// Weighted pickers still read the cumulative-weight array from global —
// preloading it into smem would require reworking the CDF offset math;
// left as a follow-up optimization. The main smem win is the binary-search
// dependent-load chain, which is driven by first_ts.
//
// Node2Vec is picker-type-ineligible for cooperative preload (per-walk
// prev_node breaks CDF sharing), so Node2Vec tasks short-circuit to
// scaffold-style per-walk processing distributed across the block.
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

    // Node2Vec: fall back to scaffold-style per-walk processing.
    if constexpr (EdgePickerType == RandomPickerType::TemporalNode2Vec) {
        const int walk_start = node_walk_starts[task_id];
        const int walk_count = node_walk_counts[task_id];
        for (int k = threadIdx.x; k < walk_count; k += blockDim.x) {
            advance_one_walk<IsDirected, Forward, EdgePickerType>(
                view, walk_set, sorted_walk_idx[walk_start + k],
                step_number, max_walk_len, base_seed);
        }
        return;
    }

    // Resolve direction-dependent per-graph pointers.
    const size_t* count_ts_group_per_node =
        Forward ? view.count_ts_group_per_node_outbound
                : (IsDirected ? view.count_ts_group_per_node_inbound
                              : view.count_ts_group_per_node_outbound);
    const size_t* node_ts_groups_offsets_g =
        Forward ? view.node_ts_group_outbound_offsets
                : (IsDirected ? view.node_ts_group_inbound_offsets
                              : view.node_ts_group_outbound_offsets);
    const size_t* node_ts_sorted_indices =
        Forward ? view.node_ts_sorted_outbound_indices
                : (IsDirected ? view.node_ts_sorted_inbound_indices
                              : view.node_ts_sorted_outbound_indices);
    const size_t* node_edge_offsets =
        Forward ? view.node_group_outbound_offsets
                : (IsDirected ? view.node_group_inbound_offsets
                              : view.node_group_outbound_offsets);
    const double* weights =
        Forward ? view.outbound_forward_cumulative_weights_exponential
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential
                              : view.outbound_backward_cumulative_weights_exponential);
    const size_t weights_size =
        Forward ? view.outbound_forward_cumulative_weights_exponential_size
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential_size
                              : view.outbound_backward_cumulative_weights_exponential_size);

    // Smem layout.
    constexpr bool kIsIndexPicker =
        random_pickers::is_index_based_picker_v<EdgePickerType>;
    constexpr int kGCap = kIsIndexPicker
        ? TRW_NODE_GROUPED_G_CAP_BLOCK_INDEX
        : TRW_NODE_GROUPED_G_CAP_BLOCK_WEIGHTED;
    constexpr size_t kHeaderBytes        = 64;
    constexpr size_t kGroupOffsetsOffset = kHeaderBytes;
    constexpr size_t kFirstTsOffset      =
        kGroupOffsetsOffset + sizeof(size_t) * kGCap;

    extern __shared__ unsigned char s_panel[];
    int* const      s_header        = reinterpret_cast<int*>(s_panel);
    size_t* const   s_node_edge_end = reinterpret_cast<size_t*>(s_panel + 16);
    size_t* const   s_group_offsets =
        reinterpret_cast<size_t*>(s_panel + kGroupOffsetsOffset);
    int64_t* const  s_first_ts      =
        reinterpret_cast<int64_t*>(s_panel + kFirstTsOffset);

    // Stage 0: thread-0 broadcast of task header.
    if (threadIdx.x == 0) {
        const int node_id    = node_walk_nodes[task_id];
        const int walk_start = node_walk_starts[task_id];
        const int walk_count = node_walk_counts[task_id];
        const size_t node_group_begin = count_ts_group_per_node[node_id];
        const size_t node_group_end   = count_ts_group_per_node[node_id + 1];
        s_header[0] = node_id;
        s_header[1] = walk_start;
        s_header[2] = walk_count;
        s_header[3] = static_cast<int>(node_group_end - node_group_begin);
        *s_node_edge_end = node_edge_offsets[node_id + 1];
    }
    __syncthreads();

    const int node_id    = s_header[0];
    const int walk_start = s_header[1];
    const int walk_count = s_header[2];
    const int G          = s_header[3];
    const size_t node_edge_end_global = *s_node_edge_end;
    const size_t node_group_begin = count_ts_group_per_node[node_id];

    // Stage 1: cooperative preload of the G-sized panel. Stride by
    // blockDim.x. Each thread handles ⌈G/blockDim.x⌉ entries.
    for (int p = threadIdx.x; p < G; p += blockDim.x) {
        const size_t ts_group_offset =
            node_ts_groups_offsets_g[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        // Preload first_ts to kill the double-indirect load on every
        // binary-search probe in stage 2's find_group_pos_slice.
        s_first_ts[p] =
            view.timestamps[node_ts_sorted_indices[ts_group_offset]];
    }
    __syncthreads();

    // Stage 2: stride loop over walks in this task's slice.
    for (int k = threadIdx.x; k < walk_count; k += blockDim.x) {
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

        // Stage 2a: find group position using the smem panel. first_ts
        // is non-null -> single-load comparator (vs the global path's
        // double indirect).
        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            s_group_offsets,
            s_first_ts,
            /*node_ts_sorted_indices=*/nullptr,
            /*view_timestamps=*/nullptr,
            weights, weights_size,
            node_group_begin,
            G, last_ts, r_group);
        (void)prev_node;  // Node2Vec short-circuited above; non-Node2Vec
                          // pickers don't use prev_node.
        if (local_pos == -1) continue;

        // Stage 2b: group -> edge range via smem offsets.
        const size_t valid_edge_start = s_group_offsets[local_pos];
        const size_t valid_edge_end =
            (local_pos + 1 < G)
                ? s_group_offsets[local_pos + 1]
                : node_edge_end_global;
        if (valid_edge_start >= valid_edge_end) continue;

        // Stage 2c: uniform random edge in the group; fetch endpoint +
        // timestamp from the global view.
        const long edge_idx = static_cast<long>(node_ts_sorted_indices[
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

// Dynamic smem size for block-smem: 64-byte header + G_CAP entries each of
// (size_t + int64_t) = 16 bytes. Chosen at compile time from the picker
// class so the launch matches the kernel's layout.
template <RandomPickerType PickerType>
HOST constexpr inline size_t block_smem_dynamic_smem_bytes() {
    constexpr bool kIsIndexPicker =
        random_pickers::is_index_based_picker_v<PickerType>;
    constexpr size_t kGCap = kIsIndexPicker
        ? static_cast<size_t>(TRW_NODE_GROUPED_G_CAP_BLOCK_INDEX)
        : static_cast<size_t>(TRW_NODE_GROUPED_G_CAP_BLOCK_WEIGHTED);
    return size_t{64} + sizeof(size_t) * kGCap + sizeof(int64_t) * kGCap;
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
    // SCAFFOLD (task 3): per-node sequential loop. Real block-global body in task 9.
    (void)node_walk_nodes;  // unused in the scaffold; task 9 consumes it.
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
