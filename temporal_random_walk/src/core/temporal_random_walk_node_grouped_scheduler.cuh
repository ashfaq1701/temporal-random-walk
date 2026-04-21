#ifndef TEMPORAL_RANDOM_WALK_NODE_GROUPED_SCHEDULER_CUH
#define TEMPORAL_RANDOM_WALK_NODE_GROUPED_SCHEDULER_CUH

#include <cstddef>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../data/buffer.cuh"
#include "../data/device_arena.cuh"
#include "../data/walk_set/walk_set_view.cuh"

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
//   - All per-step scratch lives in a DeviceArena, reset at the top of
//     every run_step call. Pointers returned in StepOutputs are valid
//     only until the next run_step call.
class NodeGroupedScheduler {
public:
    // One run_step call produces three disjoint task lists (W-partition):
    //   solo_walks:  walk_idx list for W=1 groups (consumed by solo kernel).
    //   warp_tasks:  node-level tasks for W in [2, TRW_NODE_GROUPED_T_BLOCK].
    //   block_tasks: node-level tasks for W > TRW_NODE_GROUPED_T_BLOCK.
    // Pointers are valid only until the next run_step call (arena is reset
    // at the top of each run).
    struct StepOutputs {
        // Compacted active walks sorted by current-node. Coop scaffolds
        // index into this via (walk_start, walk_count) per node task.
        int* sorted_walk_idx;

        // Host-side num_active from the first D2H readback. 0 means every
        // walk terminated before this step; caller skips pick launches.
        int num_active_host;

        // ---- Solo tier (W == 1) --------------------------------------
        int* solo_walks;               // walk_idx list, length num_solo_host
        int* num_solo_walks_device;    // device counter (kernel gate)
        int  num_solo_walks_host;      // post-partition D2H readback

        // ---- Warp tier (W in [2, T_BLOCK]) ---------------------------
        int* warp_nodes;               // node_id per task (for panel preload
                                       // in the real body, task 10)
        int* warp_walk_starts;         // offset into sorted_walk_idx
        int* warp_walk_counts;         // walks per task (W value)
        int* num_warp_tasks_device;
        int  num_warp_tasks_host;

        // ---- Block tier (W > T_BLOCK) --------------------------------
        int* block_nodes;              // node_id per task
        int* block_walk_starts;
        int* block_walk_counts;
        int* num_block_tasks_device;
        int  num_block_tasks_host;
    };

    NodeGroupedScheduler(std::size_t num_walks,
                         dim3 block_dim,
                         cudaStream_t stream);

    // Run one intermediate step's pipeline. Blocks twice on D2H stream
    // syncs per step: first for num_active (drives CUB extents), second
    // for the three tier counts (drives kernel grids). Both are one-int
    // (or three-int) readbacks; cost is trivial next to the sort/RLE
    // work they save.
    StepOutputs run_step(WalkSetView walk_set_view,
                         int step_number,
                         int max_walk_len);

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

#endif  // TEMPORAL_RANDOM_WALK_NODE_GROUPED_SCHEDULER_CUH
