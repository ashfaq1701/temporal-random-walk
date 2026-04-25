#ifndef CUDA_CONST_H
#define CUDA_CONST_H

#include <cstddef>

// Library-wide default CUDA block dim. Overridable per public-API call via
// block_dim. Defined outside HAS_CUDA because public headers use it as a
// default-parameter value — it must be visible on CPU-only builds too.
constexpr size_t BLOCK_DIM = 256;

#ifdef HAS_CUDA

#include <thrust/execution_policy.h>

constexpr auto DEVICE_EXECUTION_POLICY = thrust::device;

// =========================================================================
// NODE_GROUPED warp-coop configuration. Rationale in CLAUDE.md §3, §4.
// W_THRESHOLD_BLOCK is naturally BLOCK_DIM.
// =========================================================================

// W <= W_THRESHOLD_WARP -> solo; W <= BLOCK_DIM -> warp; W > BLOCK_DIM -> block.
// Throughput is flat across W in [1, 32] on coin once the W-partition correctly
// handles W>1 in the solo tier — earlier sweeps that suggested W=32 was a big
// win were measuring a bug where walks at hubs with W>1 routed to solo got
// silently dropped (see scheduler.cu partition_by_w_kernel). Default kept at
// 1 (safest semantics: solo only when there is exactly one walk at this node).
// Override per call via the w_threshold_warp parameter if a workload differs.
constexpr int W_THRESHOLD_WARP = 1;

// Above this, a block task splits into ceil(W/cap) disjoint block-tasks.
constexpr int W_THRESHOLD_MULTI_BLOCK = 8192;

// G-fit thresholds per tier × picker class. Per-group bytes: index=16, weight=24.
constexpr int G_THRESHOLD_BLOCK_INDEX  = 2800;  // 2800 * 16 = 44800 B  (≤ 44 KB panel)
constexpr int G_THRESHOLD_BLOCK_WEIGHT = 1800;  // 1800 * 24 = 43200 B
constexpr int G_THRESHOLD_WARP_INDEX   = 340;   // 340 * 16 = 5440 B per warp
constexpr int G_THRESHOLD_WARP_WEIGHT  = 220;   // 220 * 24 = 5280 B per warp

#endif  // HAS_CUDA

#endif  // CUDA_CONST_H
