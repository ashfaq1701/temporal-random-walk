#ifndef NODE_GROUPED_KERNELS_COOP_WARP_CUH
#define NODE_GROUPED_KERNELS_COOP_WARP_CUH

// Warp-tier cooperative kernels: smem (G <= warp cap) and global fallback.
//
// Both are still scaffolds as of task 8 — they route through per-node
// sequential loops calling advance_one_walk. Real cooperative bodies land
// with:
//   task 10 -> node_grouped_warp_smem_kernel    (warp preload)
//   task 11 -> node_grouped_warp_global_kernel  (warp fallback)

#include "per_walk.cuh"  // advance_one_walk, philox offset helper
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/warp_coop_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

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
    // SCAFFOLD (task 3): per-node sequential loop. Real warp-smem body in task 10.
    (void)node_walk_nodes;  // unused in the scaffold; task 10 consumes it.
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
        node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx,
                node_walk_nodes, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

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
