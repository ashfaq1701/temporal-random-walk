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
//   - iota_src_ and walk_to_group_size_ are batch-persistent (Buffer<int>).
//   - All per-step scratch lives in a DeviceArena, reset at the top of
//     every run_step call. Pointers returned in StepOutputs are valid
//     only until the next run_step call.
class NodeGroupedScheduler {
public:
    struct StepOutputs {
        // Compacted active walks sorted by current-node. Payload preserves
        // original walk_idx so downstream kernels address walk_set directly.
        int* sorted_walk_idx;

        // Device-side num_active counter. Kernels gate on *ptr in addition
        // to the host-sized grid — belt-and-suspenders for the last block.
        int* num_active_device;

        // Per-walk count of walks sharing this walk's current node this
        // step. Bookkeeping today — task 5's W-partition is the first real
        // consumer.
        int* step_group_size;

        // Host-side num_active from the D2H readback. 0 means every walk
        // terminated before this step; caller should skip the pick launch.
        int num_active_host;
    };

    NodeGroupedScheduler(std::size_t num_walks,
                         dim3 block_dim,
                         cudaStream_t stream);

    // Populate walk_to_group_size from start_node_ids (sort -> RLE ->
    // scatter by start_node_id). Run once per batch for constrained step
    // 0; skipped for unconstrained. The unconstrained case leaves
    // walk_to_group_size zero-initialized (by the constructor).
    void setup_step0_constrained(const int* start_node_ids);

    // Run one intermediate step's pipeline. Blocks once on a D2H stream
    // sync to read num_active into host so CUB sort/RLE/scan operate on
    // the tight compacted extent. Cost: one sync per step, negligible next
    // to the sort/RLE work it saves.
    StepOutputs run_step(WalkSetView walk_set_view,
                         int step_number,
                         int max_walk_len);

    // Batch-persistent buffer populated by setup_step0_constrained. Task 5
    // is the first real consumer (W-partition key for step 0).
    const int* walk_to_group_size() const { return walk_to_group_size_.data(); }

private:
    std::size_t  num_walks_;
    int          num_walks_int_;
    dim3         block_dim_;
    cudaStream_t stream_;

    DeviceArena arena_;
    Buffer<int> iota_src_;
    Buffer<int> walk_to_group_size_;
};

#endif  // HAS_CUDA

}  // namespace temporal_random_walk

#endif  // TEMPORAL_RANDOM_WALK_NODE_GROUPED_SCHEDULER_CUH
