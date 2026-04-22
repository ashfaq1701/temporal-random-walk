#ifndef NODE_GROUPED_SCHEDULER_CUH
#define NODE_GROUPED_SCHEDULER_CUH

#include <cstddef>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../../data/buffer.cuh"
#include "../../data/device_arena.cuh"
#include "../../data/walk_set/walk_set_view.cuh"
#include "../../data/enums.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// Orchestrates the NODE_GROUPED per-step pipeline: filter -> compact ->
// num_active readback -> gather -> sort -> RLE -> exclusive-scan -> scatter.
//
// Produces the compacted/sorted walk_idx list (in current-node order) and
// the host-side num_active count that downstream kernel launches use to
// size their grids and CUB extents.
//
// Scratch lifetime:
//   - iota_src_ is batch-persistent (Buffer<int>) — reused as the input
//     to cub_partition_flagged every step.
//   - All per-step scratch lives in a DeviceArena (chunked, pointer-
//     stable). run_step resets the arena at the top, acquires scratch,
//     launches kernels. Pointers returned in StepOutputs are valid only
//     until the next run_step call — after reset the arena may hand out
//     the same addresses to new slots.
class NodeGroupedScheduler {
public:
    // Node-level task list shared by the four cooperative tiers. Every
    // task identifies a unique node (for panel preload in the real coop
    // body) plus the walks to service (offset + count into sorted_walk_idx).
    struct TierTaskList {
        int* nodes;              // node_id per task
        int* walk_starts;        // offset into sorted_walk_idx
        int* walk_counts;        // W per task (walks sharing this node)
        int* num_tasks_device;   // device counter (kernel gate)
        int  num_tasks_host;     // post-partition D2H readback
    };

    // One run_step call produces five disjoint task lists:
    //   solo_walks        — walk_idx list for W=1 (single-thread path).
    //   warp_smem         — warp tasks that fit in per-warp smem panel.
    //   warp_global       — warp tasks with G > cap (global fallback).
    //   block_smem        — block tasks that fit in per-block smem panel.
    //   block_global      — block tasks with G > cap.
    // Pointers are valid only until the next run_step call (arena resets
    // at the top of each run).
    struct StepOutputs {
        // Compacted active walks sorted by current-node. Coop kernels
        // index into this via (walk_start, walk_count) per node task.
        int* sorted_walk_idx;

        // Host-side num_active from the first D2H readback. 0 means every
        // walk terminated before this step; caller skips all pick launches.
        int num_active_host;

        // ---- Solo tier (W == 1) --------------------------------------
        int* solo_walks;               // walk_idx list
        int* num_solo_walks_device;
        int  num_solo_walks_host;

        // ---- Cooperative tiers (four variants after W+G partition) ---
        TierTaskList warp_smem;        // W in [2, T_BLOCK] AND G ≤ warp cap
        TierTaskList warp_global;      // W in [2, T_BLOCK] AND G > warp cap
        TierTaskList block_smem;       // W > T_BLOCK        AND G ≤ block cap
        TierTaskList block_global;     // W > T_BLOCK        AND G > block cap
    };

    NodeGroupedScheduler(std::size_t num_walks,
                         dim3 block_dim,
                         cudaStream_t stream);

    // Run one intermediate step's pipeline. Blocks twice on D2H stream
    // syncs per step: first for num_active (drives CUB extents), second
    // for the tier counts (drives kernel grids).
    //
    // count_ts_group_per_node: per-node offset into the global timestamp-
    //   group arrays. Caller resolves this from the graph view based on
    //   walk direction (outbound for forward; inbound for backward-directed;
    //   outbound for backward-undirected).
    // edge_picker_type: determines which G cap applies to each tier:
    //   TRW_NODE_GROUPED_G_CAP_{WARP,BLOCK}_INDEX for index-based pickers
    //   (Uniform / Linear / ExponentialIndex), _WEIGHTED for the rest.
    StepOutputs run_step(WalkSetView walk_set_view,
                         int step_number,
                         int max_walk_len,
                         const std::size_t* count_ts_group_per_node,
                         RandomPickerType edge_picker_type);

private:
    std::size_t  num_walks_;
    int          num_walks_int_;
    dim3         block_dim_;
    cudaStream_t stream_;

    DeviceArena arena_;
    Buffer<int> iota_src_;
};

#endif  // HAS_CUDA

}  // namespace temporal_random_walk

#endif  // NODE_GROUPED_SCHEDULER_CUH
