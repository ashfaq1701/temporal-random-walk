#ifndef CUDA_CONST_H
#define CUDA_CONST_H

#ifdef HAS_CUDA

#include <thrust/execution_policy.h>

constexpr auto DEVICE_EXECUTION_POLICY = thrust::device;
constexpr size_t BLOCK_DIM = 256;
constexpr size_t BLOCK_DIM_GENERATING_RANDOM_WALKS = 256;

// Node-grouped walk-sampling tier boundaries. See CLAUDE.md §3.
// Tunable — adjust after threshold-calibration sweep; no algorithmic
// assumptions depend on the specific values.
//   group_size <= NODE_GROUPED_T_WARP        -> solo kernel (1 thread / walk)
//   T_WARP <  group_size <= NODE_GROUPED_T_BLOCK -> warp-coop (1 warp / group)
//   group_size >  NODE_GROUPED_T_BLOCK       -> block-coop (1 block / group)
constexpr int NODE_GROUPED_T_WARP  = 1;
constexpr int NODE_GROUPED_T_BLOCK = 32;

// Maximum adjacency-list degree that fits in shared memory for the
// coop tiers (edge_idx + edge_ts + cdf + scratch). See CLAUDE.md §4:
// 4·deg + 8·deg + 4·deg + ~1KB ≤ ~48KB -> deg ≲ 2800 clean fit.
// Groups whose current node has degree > this demote to a global-memory
// coop variant (or solo) at dispatch time.
constexpr int NODE_GROUPED_SMEM_DEG_LIMIT = 2800;

#endif

#endif // CUDA_CONST_H
