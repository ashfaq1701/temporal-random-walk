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
//   Step 0 — start edges.
//     Unconstrained (all start_node_ids == -1): short-circuit to
//     pick_start_edges_kernel<..., Constrained=false>, no scheduler.
//     Constrained: NodeGroupedScheduler::setup_step0_constrained populates
//     walk_to_group_size by start_node_id (bookkeeping for task 5's
//     W-partition); pick runs through pick_start_edges_kernel<..., true>.
//     Both branches collapse into a single dispatch_start_edges_kernel call.
//
//   Steps 1..max_walk_len-1 — per-step pipeline in
//   NodeGroupedScheduler::run_step: filter (by walk_padding_value) ->
//   compact -> num_active D2H readback -> gather -> sort -> RLE ->
//   exclusive-scan -> scatter. Output is a compacted/sorted walk_idx list
//   that the pick kernel iterates. Grids sized by host_num_active.
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

    if (!all_starts_unconstrained) {
        scheduler.setup_step0_constrained(start_node_ids);
    }

    dispatch_bool(is_directed, [&](auto dir_tag) {
        constexpr bool kDir = decltype(dir_tag)::value;
        dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
            constexpr bool kFwd = decltype(fwd_tag)::value;

            // ---- 1. Step 0 — start edges ------------------------------
            {
                NVTX_RANGE_COLORED("NG step0 pick", nvtx_colors::walk_green);
                // Unconstrained and constrained step 0 both go through
                // pick_start_edges_kernel today; task 5 moves constrained
                // into the solo_walks list.
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

                const size_t active_blocks =
                    (static_cast<size_t>(step_outs.num_active_host)
                     + block_dim.x - 1) / block_dim.x;
                const dim3 active_grid(static_cast<unsigned>(active_blocks));

                NVTX_RANGE_COLORED("NG pick", nvtx_colors::walk_green);
                dispatch_node_grouped_solo_kernel<kDir, kFwd>(
                    view, walk_set_view,
                    step_outs.sorted_walk_idx,
                    step_outs.num_active_device,
                    step_number, max_walk_len,
                    edge_picker_type, base_seed,
                    active_grid, block_dim, stream);
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
