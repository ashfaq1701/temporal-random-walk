#ifndef TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH
#define TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH

#include "temporal_random_walk_kernels_node_grouped.cuh"
#include "../common/picker_dispatch.cuh"
#include "../common/cuda_sort.cuh"
#include "../common/cuda_scan.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"
#include "../common/nvtx.cuh"
#include "../data/buffer.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Top-level router for the node-grouped walk path.
//
// Phases:
//   1. Step 0 — start edges.
//      - all_starts_unconstrained == true : every walk has start_node_id == -1,
//        no grouping possible. Solo unconstrained kernel handles everything.
//      - all_starts_unconstrained == false: every walk is pinned to a real
//        start_node_id. Sort-by-start_node, RLE, scatter group sizes, then:
//          - group_size == 1 walks → solo kernel.
//          - group_size >= 2 walks → warp-cooperative kernel (TODO body).
//
//   2. Steps 1..max_walk_len-1 — walk progression, same two-tier structure
//      keyed on last_node at the step. The pipeline per step is:
//        a. alive_flags[walk_idx] = (last_node != walk_padding_value)
//        b. cub_partition_flagged compacts original walk indices into
//           active_walk_idx[0..num_active). Preserves original indices.
//        c. gather_last_nodes into a keys buffer.
//        d. cub_sort_pairs (keys=last_node, values=active_walk_idx) →
//           sorted_keys, sorted_walk_idx.
//        e. cub_run_length_encode → unique_keys, run_lengths, *d_num_runs.
//        f. cub_exclusive_sum(run_lengths) → run_starts.
//        g. zero walk_to_group_size, then scatter runs into it using the
//           compacted count as the scatter extent (num_items_ptr).
//        h. Dispatch solo kernel (runs of 1) and TODO coop kernel (runs >= 2).
//
//   3. Reverse — for backward-in-time walks, flip each walk in place.
//
// is_directed and should_walk_forward are lifted to compile-time tags via
// dispatch_bool so the pipeline specializes once per (kDir, kFwd).
//
// Device-side num_runs / num_active counters keep every launch async — no
// host syncs in the fast path.
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

    dim3 grid = grid_dim;
    if (grid.x == 0) {
        grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
    }

    const bool should_walk_forward = get_should_walk_forward(walk_direction);
    const int  num_walks_int       = static_cast<int>(num_walks);

    if (num_walks == 0) return;

    // ----- Step-0 sort-and-group infra (only materialized when constrained) --
    Buffer<int> sorted_walk_idx    {/*use_gpu=*/true};
    Buffer<int> sorted_start_keys  {/*use_gpu=*/true};
    Buffer<int> iota_values        {/*use_gpu=*/true};
    Buffer<int> unique_start_keys  {/*use_gpu=*/true};
    Buffer<int> run_lengths        {/*use_gpu=*/true};
    Buffer<int> run_starts         {/*use_gpu=*/true};
    Buffer<int> d_num_runs         {/*use_gpu=*/true};
    Buffer<int> walk_to_group_size {/*use_gpu=*/true};
    Buffer<int> d_num_walks_const  {/*use_gpu=*/true};  // sorted-slot count for step 0

    if (!all_starts_unconstrained) {
        NVTX_RANGE_COLORED("NG step0 setup", nvtx_colors::index_blue);

        sorted_walk_idx   .resize(num_walks);
        sorted_start_keys .resize(num_walks);
        iota_values       .resize(num_walks);
        unique_start_keys .resize(num_walks);
        run_lengths       .resize(num_walks);
        run_starts        .resize(num_walks);
        d_num_runs        .resize(1);
        walk_to_group_size.resize(num_walks);
        d_num_walks_const .resize(1);

        // run_lengths past *d_num_runs are never written by RLE; zero so the
        // exclusive-scan extent of num_walks is well-defined.
        CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
            run_lengths.data(), 0, num_walks * sizeof(int), stream));

        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            d_num_walks_const.data(), &num_walks_int, sizeof(int),
            cudaMemcpyHostToDevice, stream));

        const size_t iota_blocks = (num_walks + block_dim.x - 1) / block_dim.x;
        iota_int_kernel<<<dim3(static_cast<unsigned>(iota_blocks)),
                          block_dim, 0, stream>>>(
            iota_values.data(), num_walks_int);

        {
            NVTX_RANGE_COLORED("NG step0 sort", nvtx_colors::index_blue);
            cub_sort_pairs(start_node_ids, sorted_start_keys.data(),
                           iota_values.data(), sorted_walk_idx.data(),
                           num_walks, stream);
        }

        {
            NVTX_RANGE_COLORED("NG step0 RLE", nvtx_colors::index_blue);
            cub_run_length_encode(sorted_start_keys.data(),
                                  unique_start_keys.data(),
                                  run_lengths.data(),
                                  d_num_runs.data(),
                                  num_walks, stream);

            cub_exclusive_sum(run_lengths.data(), run_starts.data(),
                              num_walks, stream);
        }

        {
            NVTX_RANGE_COLORED("NG step0 scatter", nvtx_colors::weight_orange);

            // Zero walk_to_group_size; every walk_idx in sorted_walk_idx gets
            // a scatter entry so zero-init is mostly a defensive default (any
            // future caller that re-uses the buffer across batches is safe).
            const size_t zero_blocks = (num_walks + block_dim.x - 1) / block_dim.x;
            zero_int_buffer_kernel<<<dim3(static_cast<unsigned>(zero_blocks)),
                                      block_dim, 0, stream>>>(
                walk_to_group_size.data(), num_walks_int);

            scatter_walk_group_sizes_kernel
                <<<dim3(static_cast<unsigned>(zero_blocks)),
                   block_dim, 0, stream>>>(
                    sorted_walk_idx.data(),
                    run_starts.data(),
                    run_lengths.data(),
                    d_num_runs.data(),
                    d_num_walks_const.data(),  // sorted-slot count == num_walks
                    walk_to_group_size.data(),
                    num_walks_int);
        }
    }

    // ----- Per-step scratch for intermediate-step grouping -----------------
    // Reused across steps; each step rewrites everything it needs.
    Buffer<uint8_t> alive_flags       {/*use_gpu=*/true};
    Buffer<int>     active_walk_idx   {/*use_gpu=*/true};  // original walk indices
    Buffer<int>     last_nodes_active {/*use_gpu=*/true};  // gathered keys
    Buffer<int>     sorted_last_nodes {/*use_gpu=*/true};
    Buffer<int>     sorted_active_idx {/*use_gpu=*/true};
    Buffer<int>     unique_last_nodes {/*use_gpu=*/true};
    Buffer<int>     step_run_lengths  {/*use_gpu=*/true};
    Buffer<int>     step_run_starts   {/*use_gpu=*/true};
    Buffer<int>     step_num_runs     {/*use_gpu=*/true};
    Buffer<int>     step_num_active   {/*use_gpu=*/true};
    Buffer<int>     step_group_size   {/*use_gpu=*/true};

    alive_flags      .resize(num_walks);
    active_walk_idx  .resize(num_walks);
    last_nodes_active.resize(num_walks);
    sorted_last_nodes.resize(num_walks);
    sorted_active_idx.resize(num_walks);
    unique_last_nodes.resize(num_walks);
    step_run_lengths .resize(num_walks);
    step_run_starts  .resize(num_walks);
    step_num_runs    .resize(1);
    step_num_active  .resize(1);
    step_group_size  .resize(num_walks);

    // active-idx source: iota 0..N-1. cub_partition_flagged compacts it
    // against alive_flags into active_walk_idx, preserving original indices.
    Buffer<int> step_iota_src{/*use_gpu=*/true};
    step_iota_src.resize(num_walks);
    {
        NVTX_RANGE_COLORED("NG iota (step-loop)", nvtx_colors::io_grey);
        const size_t iota_blocks = (num_walks + block_dim.x - 1) / block_dim.x;
        iota_int_kernel<<<dim3(static_cast<unsigned>(iota_blocks)),
                          block_dim, 0, stream>>>(
            step_iota_src.data(), num_walks_int);
    }

    dispatch_bool(is_directed, [&](auto dir_tag) {
        constexpr bool kDir = decltype(dir_tag)::value;
        dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
            constexpr bool kFwd = decltype(fwd_tag)::value;

            // ---- 1. Step 0 — start edges ------------------------------
            {
                NVTX_RANGE_COLORED("NG step0 pick", nvtx_colors::walk_green);
                if (all_starts_unconstrained) {
                    dispatch_start_edges_kernel<kDir, kFwd>(
                        view, walk_set_view, start_node_ids,
                        /*walk_to_group_size=*/nullptr,
                        /*constrained=*/false,
                        max_walk_len, num_walks, start_picker_type,
                        base_seed, grid, block_dim, stream);
                } else {
                    // Solo services group_size == 1; coop (TODO) services >= 2.
                    dispatch_start_edges_kernel<kDir, kFwd>(
                        view, walk_set_view, start_node_ids,
                        walk_to_group_size.data(),
                        /*constrained=*/true,
                        max_walk_len, num_walks, start_picker_type,
                        base_seed, grid, block_dim, stream);

                    dispatch_start_edges_cooperative_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        unique_start_keys.data(), run_starts.data(),
                        run_lengths.data(),
                        d_num_runs.data(), sorted_walk_idx.data(),
                        max_walk_len, num_walks, start_picker_type,
                        base_seed, block_dim, stream);
                }
            }

            // ---- 2. Intermediate steps --------------------------------
            for (int step_number = 1; step_number < max_walk_len; ++step_number) {
                NVTX_RANGE_COLORED("NG step", nvtx_colors::walk_green);

                // (a) flag alive walks (last_node != walk_padding_value).
                const size_t flag_blocks =
                    (num_walks + block_dim.x - 1) / block_dim.x;
                {
                    NVTX_RANGE_COLORED("NG filter alive", nvtx_colors::io_grey);
                    walk_alive_flags_kernel
                        <<<dim3(static_cast<unsigned>(flag_blocks)),
                           block_dim, 0, stream>>>(
                            walk_set_view, step_number, max_walk_len, num_walks,
                            alive_flags.data());
                }

                // (b) compact original walk indices into active_walk_idx.
                {
                    NVTX_RANGE_COLORED("NG compact", nvtx_colors::io_grey);
                    cub_partition_flagged(
                        step_iota_src.data(),
                        alive_flags.data(),
                        active_walk_idx.data(),
                        step_num_active.data(),
                        num_walks, stream);
                }

                // (c) gather last_nodes for the compacted active walks.
                {
                    NVTX_RANGE_COLORED("NG gather", nvtx_colors::io_grey);
                    gather_last_nodes_kernel
                        <<<dim3(static_cast<unsigned>(flag_blocks)),
                           block_dim, 0, stream>>>(
                            walk_set_view,
                            active_walk_idx.data(),
                            step_num_active.data(),
                            step_number, max_walk_len,
                            last_nodes_active.data());
                }

                // (d) sort by last_node; values carry original walk indices.
                // NOTE: cub_sort_pairs reads num_items as a host-side size
                // today. Sorting the full num_walks is safe: the suffix
                // [num_active, num_walks) holds stale values but never gets
                // read downstream (RLE/scatter gate on *step_num_active).
                // TODO(perf): thread a device-side item count through the
                // sort pass so we don't move dead tail entries.
                {
                    NVTX_RANGE_COLORED("NG sort", nvtx_colors::index_blue);
                    cub_sort_pairs(
                        last_nodes_active.data(), sorted_last_nodes.data(),
                        active_walk_idx.data(),  sorted_active_idx.data(),
                        num_walks, stream);
                }

                // (e) RLE sorted keys → unique_last_nodes, run_lengths.
                // (f) exclusive-scan run lengths → run_starts.
                // Same "full-extent" caveat as (d): RLE runs over num_walks
                // but the scatter step (g) uses *step_num_active as its
                // slot count, so phantom runs in the tail never leak into
                // walk_to_group_size.
                {
                    NVTX_RANGE_COLORED("NG RLE+scan", nvtx_colors::index_blue);
                    CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
                        step_run_lengths.data(), 0, num_walks * sizeof(int),
                        stream));
                    cub_run_length_encode(
                        sorted_last_nodes.data(),
                        unique_last_nodes.data(),
                        step_run_lengths.data(),
                        step_num_runs.data(),
                        num_walks, stream);

                    cub_exclusive_sum(
                        step_run_lengths.data(), step_run_starts.data(),
                        num_walks, stream);
                }

                // (g) zero group sizes, then scatter. Terminated walks
                // (not in sorted_active_idx) stay 0 and fall through the
                // solo kernel's group-size gate harmlessly.
                {
                    NVTX_RANGE_COLORED("NG scatter", nvtx_colors::weight_orange);
                    const size_t zero_blocks =
                        (num_walks + block_dim.x - 1) / block_dim.x;
                    zero_int_buffer_kernel
                        <<<dim3(static_cast<unsigned>(zero_blocks)),
                           block_dim, 0, stream>>>(
                            step_group_size.data(), num_walks_int);

                    scatter_walk_group_sizes_kernel
                        <<<dim3(static_cast<unsigned>(zero_blocks)),
                           block_dim, 0, stream>>>(
                            sorted_active_idx.data(),
                            step_run_starts.data(),
                            step_run_lengths.data(),
                            step_num_runs.data(),
                            step_num_active.data(),   // stop at compacted len
                            step_group_size.data(),
                            num_walks_int);
                }

                // (h) solo + coop launches.
                {
                    NVTX_RANGE_COLORED("NG pick", nvtx_colors::walk_green);
                    dispatch_intermediate_edges_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        sorted_active_idx.data(),
                        step_num_active.data(),
                        step_group_size.data(),
                        step_number, max_walk_len, num_walks,
                        edge_picker_type, base_seed,
                        grid, block_dim, stream);

                    dispatch_intermediate_edges_cooperative_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        unique_last_nodes.data(),
                        step_run_starts.data(),
                        step_run_lengths.data(),
                        step_num_runs.data(),
                        sorted_active_idx.data(),
                        step_number, max_walk_len, num_walks,
                        edge_picker_type, base_seed, block_dim, stream);
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

#endif // TEMPORAL_RANDOM_WALK_NODE_GROUPED_DISPATCH_CUH
