#ifndef NODE_GROUPED_DISPATCH_CUH
#define NODE_GROUPED_DISPATCH_CUH

#include "kernels.cuh"
#include "scheduler.cuh"
#include "../../common/picker_dispatch.cuh"
#include "../../common/nvtx.cuh"
#include "../../common/cuda_config.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// Routes NODE_GROUPED to the scheduler+coop path or to per_walk_step_kernel
// for Node2Vec (whose prev_node-dependent CDF can't share a coop panel).
// force_global_only: ablation — route every coop task to *_global (no smem).
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
    const cudaStream_t stream = 0,
    const bool force_global_only = false) {

    if (num_walks == 0) return;

    dim3 grid = grid_dim;
    if (grid.x == 0) {
        grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
    }

    const bool should_walk_forward = get_should_walk_forward(walk_direction);

    // Node2Vec's per-walk CDF breaks coop panel sharing.
    const bool use_per_walk_path =
        (edge_picker_type == RandomPickerType::TemporalNode2Vec);

    dispatch_bool(is_directed, [&](auto dir_tag) {
        constexpr bool kDir = decltype(dir_tag)::value;
        dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
            constexpr bool kFwd = decltype(fwd_tag)::value;

            {
                NVTX_RANGE_COLORED("NG step0 pick", nvtx_colors::walk_green);
                dispatch_start_edges_kernel<kDir, kFwd>(
                    view, walk_set_view, start_node_ids,
                    /*constrained=*/!all_starts_unconstrained,
                    max_walk_len, num_walks, start_picker_type,
                    base_seed, grid, block_dim, stream);
            }

            if (use_per_walk_path) {
                for (int step_number = 1; step_number < max_walk_len; ++step_number) {
                    NVTX_RANGE_COLORED("NG per-walk step", nvtx_colors::walk_green);
                    dispatch_per_walk_step_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_number, max_walk_len, num_walks,
                        edge_picker_type, base_seed,
                        grid, block_dim, stream);
                }
            } else {
                NodeGroupedScheduler scheduler(num_walks, block_dim, stream);

                // Direction-dependent ts-group offsets; matches get_node_edge_at_device.
                const std::size_t* count_ts_group_per_node =
                    kFwd
                        ? view.count_ts_group_per_node_outbound
                        : (kDir ? view.count_ts_group_per_node_inbound
                                : view.count_ts_group_per_node_outbound);

                for (int step_number = 1; step_number < max_walk_len; ++step_number) {
                    auto step_outs = scheduler.run_step(
                        walk_set_view, step_number, max_walk_len,
                        count_ts_group_per_node, edge_picker_type,
                        force_global_only);

                    // Walk padding is absorbing — once all walks are dead, none revive.
                    if (step_outs.num_active_host <= 0) break;

                    NVTX_RANGE_COLORED("NG pick", nvtx_colors::walk_green);

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

                    const unsigned warps_per_block = block_dim.x / 32u;

                    if (step_outs.warp_smem.num_tasks_host > 0) {
                        const size_t warp_smem_blocks =
                            (static_cast<size_t>(step_outs.warp_smem.num_tasks_host)
                             + warps_per_block - 1) / warps_per_block;
                        const dim3 warp_smem_grid(static_cast<unsigned>(warp_smem_blocks));
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

                    if (step_outs.warp_global.num_tasks_host > 0) {
                        const size_t warp_global_blocks =
                            (static_cast<size_t>(step_outs.warp_global.num_tasks_host)
                             + warps_per_block - 1) / warps_per_block;
                        const dim3 warp_global_grid(static_cast<unsigned>(warp_global_blocks));
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

                    if (step_outs.block_smem.num_tasks_host > 0) {
                        const dim3 block_smem_grid(
                            static_cast<unsigned>(step_outs.block_smem.num_tasks_host));
                        dispatch_node_grouped_block_smem_kernel<kDir, kFwd>(
                            view, walk_set_view,
                            step_outs.sorted_walk_idx,
                            step_outs.block_smem.nodes,
                            step_outs.block_smem.walk_starts,
                            step_outs.block_smem.walk_counts,
                            step_outs.block_smem.num_tasks_device,
                            step_number, max_walk_len,
                            edge_picker_type, base_seed,
                            block_smem_grid, block_dim, stream);
                    }

                    if (step_outs.block_global.num_tasks_host > 0) {
                        const dim3 block_global_grid(
                            static_cast<unsigned>(step_outs.block_global.num_tasks_host));
                        dispatch_node_grouped_block_global_kernel<kDir, kFwd>(
                            view, walk_set_view,
                            step_outs.sorted_walk_idx,
                            step_outs.block_global.nodes,
                            step_outs.block_global.walk_starts,
                            step_outs.block_global.walk_counts,
                            step_outs.block_global.num_tasks_device,
                            step_number, max_walk_len,
                            edge_picker_type, base_seed,
                            block_global_grid, block_dim, stream);
                    }
                }
            }

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
