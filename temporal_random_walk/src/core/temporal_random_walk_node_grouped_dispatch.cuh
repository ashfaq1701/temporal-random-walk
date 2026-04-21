#ifndef TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH
#define TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH

#include "temporal_random_walk_kernels_node_grouped.cuh"
#include "temporal_random_walk_node_grouped_scheduler.cuh"
#include "../common/picker_dispatch.cuh"
#include "../common/nvtx.cuh"

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

            // ---- 2. Intermediate steps --------------------------------
            for (int step_number = 1; step_number < max_walk_len; ++step_number) {
                auto step_outs = scheduler.run_step(
                    walk_set_view, step_number, max_walk_len);

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

                // Warp-smem tier: one task per node (W in [2, T_BLOCK]).
                if (step_outs.num_warp_tasks_host > 0) {
                    const size_t warp_blocks =
                        (static_cast<size_t>(step_outs.num_warp_tasks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 warp_grid(static_cast<unsigned>(warp_blocks));
                    dispatch_node_grouped_warp_smem_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.warp_walk_starts,
                        step_outs.warp_walk_counts,
                        step_outs.num_warp_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        warp_grid, block_dim, stream);
                }

                // Block-smem tier: one task per node (W > T_BLOCK).
                if (step_outs.num_block_tasks_host > 0) {
                    const size_t block_blocks =
                        (static_cast<size_t>(step_outs.num_block_tasks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 block_grid(static_cast<unsigned>(block_blocks));
                    dispatch_node_grouped_block_smem_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.sorted_walk_idx,
                        step_outs.block_walk_starts,
                        step_outs.block_walk_counts,
                        step_outs.num_block_tasks_device,
                        step_number, max_walk_len,
                        edge_picker_type, base_seed,
                        block_grid, block_dim, stream);
                }

                // warp_global and block_global scaffolds are declared but
                // unused at task-5 state — all G-fitting decisions deferred
                // to task 6. Until then every coop walk goes through the
                // smem variant (distribution identical either way since
                // both bodies are still solo-copies).
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

#endif // TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH
