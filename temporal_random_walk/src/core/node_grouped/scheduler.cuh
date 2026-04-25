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

// Runs the NODE_GROUPED per-step pipeline (filter, sort-by-node, RLE,
// W-partition, G-partition, block-task expansion) and emits five disjoint
// task lists. Per-step scratch lives in a DeviceArena; StepOutputs pointers
// are valid only until the next run_step call.
class NodeGroupedScheduler {
public:
    struct TierTaskList {
        int* nodes;
        int* walk_starts;
        int* walk_counts;
        int* num_tasks_device;
        int  num_tasks_host;
    };

    struct StepOutputs {
        int* sorted_walk_idx;
        int  num_active_host;

        int* solo_walks;
        int* num_solo_walks_device;
        int  num_solo_walks_host;

        TierTaskList warp_smem;
        TierTaskList warp_global;
        TierTaskList block_smem;
        TierTaskList block_global;
    };

    NodeGroupedScheduler(std::size_t num_walks,
                         dim3 block_dim,
                         int w_threshold_warp,
                         cudaStream_t stream);

    // Two blocking D2Hs per call: num_active (drives CUB extents) and
    // tier counts (drives kernel grids). force_global_only sets both G
    // caps to -1 so every coop task routes to *_global (ablation knob).
    StepOutputs run_step(WalkSetView walk_set_view,
                         int step_number,
                         int max_walk_len,
                         const std::size_t* count_ts_group_per_node,
                         RandomPickerType edge_picker_type,
                         bool force_global_only = false);

private:
    std::size_t  num_walks_;
    int          num_walks_int_;
    dim3         block_dim_;
    int          w_threshold_warp_;
    cudaStream_t stream_;

    DeviceArena arena_;
    Buffer<int> iota_src_;
};

#endif  // HAS_CUDA

}  // namespace temporal_random_walk

#endif  // NODE_GROUPED_SCHEDULER_CUH
