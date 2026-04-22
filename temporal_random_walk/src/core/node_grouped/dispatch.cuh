#ifndef NODE_GROUPED_DISPATCH_CUH
#define NODE_GROUPED_DISPATCH_CUH

#include "kernels.cuh"
#include "scheduler.cuh"
#include "../../common/picker_dispatch.cuh"
#include "../../common/nvtx.cuh"
#include "../../common/warp_coop_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Top-level router for the node-grouped walk path.
//
//   Step 0 — start edges. pick_start_edges_kernel handles both
//   unconstrained and constrained start today. Task 5 leaves step 0
//   outside the W-partition; a future task moves constrained step 0
//   into the solo_walks list if profiling justifies it.
//
//   Steps 1..max_walk_len-1 — per-step pipeline in
//   NodeGroupedScheduler::run_step: filter -> compact -> num_active D2H
//   readback -> gather -> sort -> RLE -> exclusive-scan -> W-partition
//   (solo / warp / block tiers). The scheduler returns three disjoint
//   task lists; the dispatcher launches up to three kernels per step:
//   solo on solo_walks, warp-smem on warp tasks, block-smem on block tasks.
//   Each launch is skipped if its tier count is 0.
//
//   Reverse — backward-in-time walks get reverse_walks_kernel flipped in
//   place after the step loop.
//
// is_directed and should_walk_forward are lifted to compile-time tags via
// dispatch_bool so the pipeline specializes once per (kDir, kFwd).
// ==========================================================================

inline void dispatch_node_grouped_kernel(
    const TemporalGraphView& view,
    const bool is_directed,
    WalkSetView walk_set_view,
    const int max_walk_len,
    const int* start_node_ids,
    const size_t num_walks,
    const bool all_starts_unconstrained,
    const RandomPickerType edge_picker_type,
    const RandomPickerType start_picker_type,
    const WalkDirection walk_direction,
    const uint64_t base_seed,
    const dim3& grid_dim,
    const dim3& block_dim,
    const cudaStream_t stream = 0) {

    if (num_walks == 0) return;

    dim3 grid = grid_dim;
    if (grid.x == 0) {
        grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
    }

    const bool should_walk_forward = get_should_walk_forward(walk_direction);

    NodeGroupedScheduler scheduler(num_walks, block_dim, stream);

    dispatch_bool(is_directed, [&](auto dir_tag) {
        constexpr bool kDir = decltype(dir_tag)::value;
        dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
            constexpr bool kFwd = decltype(fwd_tag)::value;

            // ---- 1. Step 0 — start edges ------------------------------
            {
                NVTX_RANGE_COLORED("NG step0 pick", nvtx_colors::walk_green);
                dispatch_start_edges_kernel<kDir, kFwd>(
                    view, walk_set_view, start_node_ids,
                    /*constrained=*/!all_starts_unconstrained,
                    max_walk_len, num_walks, start_picker_type,
                    base_seed, grid, block_dim, stream);
            }

            // Per-direction, per-directedness pick of the timestamp-group
            // offsets array. Forward -> outbound; Backward directed -> inbound;
            // Backward undirected -> outbound. Matches get_node_edge_at_device.
            const std::size_t* count_ts_group_per_node =
                kFwd
                    ? view.count_ts_group_per_node_outbound
                    : (kDir ? view.count_ts_group_per_node_inbound
                            : view.count_ts_group_per_node_outbound);

            // ---- 2. Intermediate steps --------------------------------
            for (int step_number = 1; step_number < max_walk_len; ++step_number) {
                auto step_outs = scheduler.run_step(
                    walk_set_view, step_number, max_walk_len,
                    count_ts_group_per_node, edge_picker_type);

                if (step_outs.num_active_host <= 0) continue;

                NVTX_RANGE_COLORED("NG pick", nvtx_colors::walk_green);

                // Solo tier: one thread per walk in solo_walks.
                if (step_outs.num_solo_walks_host > 0) {
                    const size_t solo_blocks =
                        (static_cast<size_t>(step_outs.num_solo_walks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 solo_grid(static_cast<unsigned>(solo_blocks));
                    dispatch_node_grouped_solo_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.solo_walks,
                        step_outs.num_solo_walks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        solo_grid, block_dim, stream);
                }

                // Warp-smem tier: W in [2, T_BLOCK] and G <= warp cap.
                // Scaffold body (per-thread-per-task) until task 10 lands.
                if (step_outs.warp_smem.num_tasks_host > 0) {
                    const size_t blocks =
                        (static_cast<size_t>(step_outs.warp_smem.num_tasks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 warp_smem_grid(static_cast<unsigned>(blocks));
                    dispatch_node_grouped_warp_smem_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.warp_smem.nodes,
                        step_outs.warp_smem.walk_starts,
                        step_outs.warp_smem.walk_counts,
                        step_outs.warp_smem.num_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        warp_smem_grid, block_dim, stream);
                }

                // Warp-global tier: W in [2, T_BLOCK] and G > warp cap.
                // Scaffold body (per-thread-per-task) until task 11 lands.
                if (step_outs.warp_global.num_tasks_host > 0) {
                    const size_t blocks =
                        (static_cast<size_t>(step_outs.warp_global.num_tasks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 warp_global_grid(static_cast<unsigned>(blocks));
                    dispatch_node_grouped_warp_global_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.warp_global.nodes,
                        step_outs.warp_global.walk_starts,
                        step_outs.warp_global.walk_counts,
                        step_outs.warp_global.num_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        warp_global_grid, block_dim, stream);
                }

                // Block-smem tier: W > T_BLOCK and G <= block cap.
                // Real cooperative body (task 8): one block per task, 256
                // threads per block. Dynamic smem is sized inside the
                // dispatch wrapper based on picker class.
                if (step_outs.block_smem.num_tasks_host > 0) {
                    const dim3 block_smem_grid(
                        static_cast<unsigned>(step_outs.block_smem.num_tasks_host));
                    const dim3 block_smem_block(
                        static_cast<unsigned>(TRW_NODE_GROUPED_COOP_BLOCK_THREADS));
                    dispatch_node_grouped_block_smem_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.block_smem.nodes,
                        step_outs.block_smem.walk_starts,
                        step_outs.block_smem.walk_counts,
                        step_outs.block_smem.num_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        block_smem_grid, block_smem_block, stream);
                }

                // Block-global tier: W > T_BLOCK and G > block cap.
                // Real cooperative body (task 9): one block per task, 256
                // threads per block. No smem panel (G doesn't fit); the
                // in-kernel binary search runs against global arrays via
                // find_group_pos_slice's double-indirect fallback.
                if (step_outs.block_global.num_tasks_host > 0) {
                    const dim3 block_global_grid(
                        static_cast<unsigned>(step_outs.block_global.num_tasks_host));
                    const dim3 block_global_block(
                        static_cast<unsigned>(TRW_NODE_GROUPED_COOP_BLOCK_THREADS));
                    dispatch_node_grouped_block_global_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.block_global.nodes,
                        step_outs.block_global.walk_starts,
                        step_outs.block_global.walk_counts,
                        step_outs.block_global.num_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        block_global_grid, block_global_block, stream);
                }
            }

            // ---- 3. Reverse if walking backward -----------------------
            if constexpr (!kFwd) {
                NVTX_RANGE_COLORED("NG reverse", nvtx_colors::edge_purple);
                reverse_walks_kernel<<<grid, block_dim, 0, stream>>>(
                    walk_set_view, num_walks);
            }
        });
    });
}

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // NODE_GROUPED_DISPATCH_CUH
