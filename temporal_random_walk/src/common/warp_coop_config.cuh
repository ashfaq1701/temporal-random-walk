#ifndef WARP_COOP_CONFIG_CUH
#define WARP_COOP_CONFIG_CUH

// Single source of truth for the NODE_GROUPED cooperative-sampling
// configuration. Values and rationale live in CLAUDE.md §3 and §4; this
// header exists so kernels and the scheduler agree on the same constants
// without round-tripping through the documentation.
//
// These are tunable. Algorithmic correctness does not depend on the
// specific values — only on their relative ordering and on the smem-fit
// test using the same cap as the kernel launch expects. Changing a G cap
// requires re-checking that G_CAP * per-group bytes ≤ the panel byte
// budget for that tier.

#ifdef HAS_CUDA

#include <cstddef>
#include <cstdint>

// -----------------------------------------------------------------------
// Tier boundaries on W (walks-per-node at the current step).
// -----------------------------------------------------------------------
//   W <= TRW_NODE_GROUPED_T_WARP                       -> solo        (1 thread per walk)
//   TRW_NODE_GROUPED_T_WARP  < W <= TRW_NODE_GROUPED_T_BLOCK  -> warp-coop   (1 warp per node)
//   W >  TRW_NODE_GROUPED_T_BLOCK                      -> block-coop  (1 block per task)
//
// Warp upper bound matches 8 intra-warp stride rounds (⌈255/32⌉ = 8).
// Block threshold marks the point where an entire 256-thread block
// amortizes the panel preload better than 8 warps spread across blocks.
constexpr int TRW_NODE_GROUPED_T_WARP  = 1;
constexpr int TRW_NODE_GROUPED_T_BLOCK = 255;

// -----------------------------------------------------------------------
// Launch shape for cooperative tiers.
// -----------------------------------------------------------------------
// Warp tier: 8 warps per block, one warp per unique node.
constexpr int TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK = 8;
// Block tier: 256 threads per block, one block per block-task.
constexpr int TRW_NODE_GROUPED_COOP_BLOCK_THREADS   = 256;

// -----------------------------------------------------------------------
// Per-block walk cap for the block tier. Nodes with W > cap split into
// ⌈W/cap⌉ disjoint block-tasks so no single block monopolizes an SM.
// See CLAUDE.md §3 "Per-block walk cap".
// -----------------------------------------------------------------------
constexpr int TRW_NODE_GROUPED_BLOCK_WALK_CAP = 8192;

// -----------------------------------------------------------------------
// Shared-memory budget.
// -----------------------------------------------------------------------
// 48 KB static envelope per block on sm_70+ without cudaFuncSetAttribute
// opt-in, minus ~4 KB for Philox state / control fields / alignment pad
// → 44 KB usable panel per block.
constexpr std::size_t TRW_NODE_GROUPED_SMEM_PANEL_BYTES = 45056u;  // 44 KB

// Per-warp slice (warp tier packs 8 warps into one block's panel).
constexpr std::size_t TRW_NODE_GROUPED_SMEM_PANEL_BYTES_PER_WARP =
    TRW_NODE_GROUPED_SMEM_PANEL_BYTES / TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK;

// -----------------------------------------------------------------------
// Per-group smem footprint (bytes/group). See CLAUDE.md §4.
// -----------------------------------------------------------------------
// Index pickers: s_group_offsets[G] (8 B) + s_first_ts[G] (8 B).
constexpr std::size_t TRW_NODE_GROUPED_PANEL_BYTES_PER_GROUP_INDEX    = 16u;
// Weighted pickers: adds s_cum_weights[G] (8 B).
constexpr std::size_t TRW_NODE_GROUPED_PANEL_BYTES_PER_GROUP_WEIGHTED = 24u;

// -----------------------------------------------------------------------
// Derived G caps. Keying is tier × picker class. The smem-fit test
// (scheduler's G-partition step) compares G to the appropriate cap.
// -----------------------------------------------------------------------
// Block tier: full-block panel ÷ bytes-per-group.
//   2800 * 16 = 44800 B ≤ 45056 ✓
//   1800 * 24 = 43200 B ≤ 45056 ✓
constexpr int TRW_NODE_GROUPED_G_CAP_BLOCK_INDEX    = 2800;
constexpr int TRW_NODE_GROUPED_G_CAP_BLOCK_WEIGHTED = 1800;
// Warp tier: per-warp slice ÷ bytes-per-group.
//   340 * 16 = 5440 B ≤ 5632 ✓
//   220 * 24 = 5280 B ≤ 5632 ✓
constexpr int TRW_NODE_GROUPED_G_CAP_WARP_INDEX     = 340;
constexpr int TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED  = 220;

#endif  // HAS_CUDA

#endif  // WARP_COOP_CONFIG_CUH
