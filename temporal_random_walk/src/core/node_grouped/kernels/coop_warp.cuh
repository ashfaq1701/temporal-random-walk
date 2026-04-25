#ifndef NODE_GROUPED_KERNELS_COOP_WARP_CUH
#define NODE_GROUPED_KERNELS_COOP_WARP_CUH

// Warp-tier coop kernels: warp_smem (G <= cap, preloaded panel) and
// warp_global (G > cap, no preload). Non-Node2Vec only.

#include "common.cuh"
#include "per_walk.cuh"
#include "../../../common/picker_dispatch.cuh"
#include "../../../common/cuda_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// blockDim.x/32 warps per block, one task per warp. Panels tiled
// side-by-side in dynamic smem. __syncwarp() only — block-wide syncs
// would deadlock the partial last block's idle warps.
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

    if (task_id >= *num_tasks_ptr) return;

    const auto ptrs = resolve_node_dir_ptrs<IsDirected, Forward>(view);

    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

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

    // Cooperative preload (stride 32) kills the 3-deep dependent load
    // chain at search time.
    for (int p = lane_id; p < G; p += 32) {
        const size_t ts_group_offset =
            ptrs.node_ts_groups_offsets[node_group_begin + p];
        s_group_offsets[p] = ts_group_offset;
        s_first_ts[p] =
            view.timestamps[ptrs.node_ts_sorted_indices[ts_group_offset]];
    }
    __syncwarp();

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

        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            s_group_offsets, s_first_ts,
            /*node_ts_sorted_indices=*/nullptr,
            /*view_timestamps=*/nullptr,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, r_group);
        if (local_pos == -1) continue;

        sample_edge_and_add_hop<IsDirected, Forward>(
            view, walk_set, ptrs,
            s_group_offsets, local_pos, G, node_edge_end, node_id,
            walk_idx, r_edge);
    }
}

// Per-warp panel × (block_dim/32). Per-warp footprint is compile-time.
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
        auto* kernel = node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>;
        // Opt into the arch's dynamic-smem ceiling so block_dim > 256 (warps
        // per block > 8) can allocate the larger tiled per-warp panel past
        // the 48 KB static envelope.
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        kernel<<<grid, block_dim, smem_bytes, stream>>>(
            view, walk_set_view,
            sorted_walk_idx,
            node_walk_nodes, node_walk_starts, node_walk_counts,
            num_tasks_ptr,
            step_number, max_walk_len, base_seed);
    });
}

// Warp-global: same launch shape as warp-smem, no panel. Cooperation
// buys L1/L2 residency across lanes sharing one node's arrays.
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

    const int    node_id          = node_walk_nodes[task_id];
    const int    walk_start       = node_walk_starts[task_id];
    const int    walk_count       = node_walk_counts[task_id];
    const size_t node_group_begin = ptrs.count_ts_group_per_node[node_id];
    const size_t node_group_end   = ptrs.count_ts_group_per_node[node_id + 1];
    const int    G                = static_cast<int>(node_group_end - node_group_begin);
    const size_t node_edge_end    = ptrs.node_edge_offsets[node_id + 1];

    const size_t* group_offsets_slice =
        ptrs.node_ts_groups_offsets + node_group_begin;

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

        const long local_pos = temporal_graph::find_group_pos_slice<Forward, EdgePickerType>(
            group_offsets_slice,
            /*first_ts=*/nullptr,
            ptrs.node_ts_sorted_indices, view.timestamps,
            ptrs.weights, ptrs.weights_size,
            node_group_begin, G, last_ts, r_group);
        if (local_pos == -1) continue;

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
