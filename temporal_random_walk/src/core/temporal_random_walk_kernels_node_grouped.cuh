#ifndef TEMPORAL_RANDOM_WALK_KERNELS_NODE_GROUPED_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_NODE_GROUPED_CUH

#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../graph/temporal_graph.cuh"
#include "../graph/edge_selectors.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "../common/picker_dispatch.cuh"
#include "helpers.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// ==========================================================================
// Philox offset scheme
//
// Each walk has its own Philox substream keyed on walk_idx. Within that
// substream the start kernel draws from the head; every intermediate step
// kernel jumps past the start kernel's budget (3 positions — 2 directed /
// 3 undirected, always rounded up) and then consumes exactly 2 positions
// per step. Bit-exactness is a function of (base_seed, walk_idx, step) so
// it is independent of which tier (solo/warp/block) actually ran a walk.
// ==========================================================================

DEVICE __forceinline__ uint64_t step_kernel_philox_offset(const int step_number) {
    constexpr uint64_t START_KERNEL_BUDGET = 3ULL;
    return START_KERNEL_BUDGET + static_cast<uint64_t>(step_number) * 2ULL;
}

// ==========================================================================
// Per-walk kernels (step 0 start edges, solo, cooperative scaffolds, reverse).
//
// Scheduler-internal helpers (iota, alive-flags, gather, scatter, zero) live
// in temporal_random_walk_node_grouped_scheduler.cu's anonymous namespace —
// they are not part of the per-walk kernel interface.
// ==========================================================================

// ==========================================================================
// Step 0 — start edges
//
// Constrained: all walks start from a real node id. Walks sharing a start
// node amortize per-node work in the cooperative tier (TODO). Solo services
// group_size == 1 only; the template tag lifts the "constrained" invariant
// to compile time so there is no per-walk -1 branching.
//
// Unconstrained: all walks start from -1 (pick any edge). No grouping; solo
// handles every walk.
//
// Uniformity is the caller's responsibility — see dispatch_node_grouped_kernel.
// ==========================================================================

template <bool IsDirected, bool Forward, RandomPickerType StartPickerType, bool Constrained>
__global__ void pick_start_edges_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ start_node_ids,
    const int max_walk_len,
    const size_t num_walks,
    const uint64_t base_seed) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;
    if (max_walk_len == 0) return;

    // Temporary: this kernel handles both unconstrained step 0 (§5 short-circuit)
    // and constrained step 0 while the scheduler's W-partition is not yet in
    // place. Task 5 will move the constrained case into solo_walks and this
    // kernel will shrink to the unconstrained path only.

    PhiloxState rng;
    init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx));
    const double r_start_a = draw_u01_philox(rng);
    const double r_start_b = draw_u01_philox(rng);

    InternalEdge start_edge;
    if constexpr (Constrained) {
        start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
            view, start_node_ids[walk_idx], /*timestamp=*/-1, /*prev=*/-1,
            r_start_a, r_start_b);
    } else {
        start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
            view, /*timestamp=*/-1, r_start_a, r_start_b);
    }

    // Dead-end: no edge satisfies the constraint. add_hop was never called,
    // so walk_lens stays 0 and slot 0 keeps walk_padding_value. The
    // intermediate-step filter (walk_alive_flags_kernel) drops the walk.
    if (start_edge.i == -1) return;

    const int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
    const int     start_src = start_edge.u;
    const int     start_dst = start_edge.i;
    const int64_t start_ts  = start_edge.ts;

    if constexpr (IsDirected) {
        if constexpr (Forward) {
            walk_set.add_hop(walk_idx, start_src, sentinel_timestamp);
            walk_set.add_hop(walk_idx, start_dst, start_ts, start_edge.edge_id);
        } else {
            walk_set.add_hop(walk_idx, start_dst, sentinel_timestamp);
            walk_set.add_hop(walk_idx, start_src, start_ts, start_edge.edge_id);
        }
    } else {
        // Unconstrained undirected draws a 3rd Philox value so
        // START_KERNEL_BUDGET == 3 always lands past the start kernel's
        // draws. Constrained consumes 2 in both solo and coop paths, so
        // all tiers stay bit-exact.
        int picked_node;
        if constexpr (Constrained) {
            picked_node = start_node_ids[walk_idx];
        } else {
            picked_node = pick_random_number(start_src, start_dst, draw_u01_philox(rng));
        }
        const int other_node = pick_other_number(start_src, start_dst, picked_node);
        walk_set.add_hop(walk_idx, picked_node, sentinel_timestamp);
        walk_set.add_hop(walk_idx, other_node, start_ts, start_edge.edge_id);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_start_edges_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* start_node_ids,
    const bool constrained,
    const int max_walk_len,
    const size_t num_walks,
    const RandomPickerType start_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(start_picker_type, [&](auto start_tag) {
        constexpr auto kStart = decltype(start_tag)::value;
        dispatch_bool(constrained, [&](auto c_tag) {
            constexpr bool kC = decltype(c_tag)::value;
            pick_start_edges_kernel<IsDirected, Forward, kStart, kC>
                <<<grid, block_dim, 0, stream>>>(
                    view, walk_set_view, start_node_ids,
                    max_walk_len, num_walks, base_seed);
        });
    });
}

// ==========================================================================
// Single-walk advance helper — the sole primitive that computes one walk's
// hop at step_number. Doesn't index any list; caller supplies walk_idx.
//
// Shared by solo and the four cooperative scaffolds. Each walk's outcome is
// a function of (walk_idx, step_number, base_seed) alone — Philox is keyed
// on walk_idx, last_node/last_ts/prev_node are read from walk_set at fixed
// offsets. So any thread can call advance_one_walk for any walk_idx and get
// the same result as solo would.
//
// Scaffold cooperative kernels (see below) exploit this: they consume
// node-level task lists (start/count into sorted_walk_idx) and loop
// advance_one_walk sequentially for the walks in their node's range —
// distribution matches solo, but the launch topology matches the eventual
// cooperative design that tasks 8–11 will fill in.
// ==========================================================================

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
DEVICE __forceinline__ void advance_one_walk(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int walk_idx_int,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    if (walk_idx_int < 0) return;
    if (step_number >= max_walk_len - 1) return;

    const size_t walk_idx = static_cast<size_t>(walk_idx_int);
    const size_t offset = walk_idx * static_cast<size_t>(max_walk_len)
                          + static_cast<size_t>(step_number);

    const int     last_node = walk_set.nodes[offset];
    const int64_t last_ts   = walk_set.timestamps[offset];
    const int     prev_node = step_number > 0 ? walk_set.nodes[offset - 1] : -1;

    PhiloxState rng;
    init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx),
                      step_kernel_philox_offset(step_number));
    const double r_a = draw_u01_philox(rng);
    const double r_b = draw_u01_philox(rng);

    const InternalEdge next_edge =
        temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
            view, last_node, last_ts, prev_node, r_a, r_b);

    if (next_edge.ts == -1) return;

    if constexpr (IsDirected) {
        walk_set.add_hop(walk_idx,
                         Forward ? next_edge.i : next_edge.u,
                         next_edge.ts, next_edge.edge_id);
    } else {
        const int next_node = pick_other_number(next_edge.u, next_edge.i, last_node);
        walk_set.add_hop(walk_idx, next_node, next_edge.ts, next_edge.edge_id);
    }
}

// ==========================================================================
// Solo kernel — one thread per walk. Services the solo_walks list (W=1
// groups after the scheduler's W-partition).
// ==========================================================================
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_solo_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ solo_walks,
    const int* __restrict__ num_solo_walks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *num_solo_walks_ptr) return;

    advance_one_walk<IsDirected, Forward, EdgePickerType>(
        view, walk_set, solo_walks[i],
        step_number, max_walk_len, base_seed);
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_solo_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* solo_walks,
    const int* num_solo_walks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_solo_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                solo_walks, num_solo_walks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// ==========================================================================
// Cooperative-tier scaffolds.
//
// Each kernel consumes a node-level task list (walks belonging to each
// unique node, identified by offsets into sorted_walk_idx). Task 5 wires
// these into the dispatcher; tasks 8–11 replace each scaffold body with
// the real tier-specific implementation (cooperative preload + stride
// loop).
//
// Scaffold semantics: one thread per node task, sequentially iterating
// every walk in its node's range via advance_one_walk. Inefficient — each
// walk runs on one thread instead of sharing work within a warp/block —
// but produces identical per-walk output to solo. Real cooperation lands
// with the bodies:
//   task 8  -> node_grouped_block_smem_kernel   (block preload)
//   task 9  -> node_grouped_block_global_kernel (block fallback)
//   task 10 -> node_grouped_warp_smem_kernel    (warp preload)
//   task 11 -> node_grouped_warp_global_kernel  (warp fallback)
// ==========================================================================

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_smem_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {
    // SCAFFOLD (task 3): per-node sequential loop. Real warp-smem body in task 10.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tasks_ptr) return;

    const int walk_start = node_walk_starts[t];
    const int walk_count = node_walk_counts[t];
    for (int k = 0; k < walk_count; ++k) {
        advance_one_walk<IsDirected, Forward, EdgePickerType>(
            view, walk_set, sorted_walk_idx[walk_start + k],
            step_number, max_walk_len, base_seed);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_smem_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_warp_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_warp_global_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {
    // SCAFFOLD (task 3): per-node sequential loop. Real warp-global body in task 11.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tasks_ptr) return;

    const int walk_start = node_walk_starts[t];
    const int walk_count = node_walk_counts[t];
    for (int k = 0; k < walk_count; ++k) {
        advance_one_walk<IsDirected, Forward, EdgePickerType>(
            view, walk_set, sorted_walk_idx[walk_start + k],
            step_number, max_walk_len, base_seed);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_warp_global_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_warp_global_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_block_smem_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {
    // SCAFFOLD (task 3): per-node sequential loop. Real block-smem body in task 8.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tasks_ptr) return;

    const int walk_start = node_walk_starts[t];
    const int walk_count = node_walk_counts[t];
    for (int k = 0; k < walk_count; ++k) {
        advance_one_walk<IsDirected, Forward, EdgePickerType>(
            view, walk_set, sorted_walk_idx[walk_start + k],
            step_number, max_walk_len, base_seed);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_block_smem_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_block_smem_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_block_global_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ node_walk_starts,
    const int* __restrict__ node_walk_counts,
    const int* __restrict__ num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {
    // SCAFFOLD (task 3): per-node sequential loop. Real block-global body in task 9.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tasks_ptr) return;

    const int walk_start = node_walk_starts[t];
    const int walk_count = node_walk_counts[t];
    for (int k = 0; k < walk_count; ++k) {
        advance_one_walk<IsDirected, Forward, EdgePickerType>(
            view, walk_set, sorted_walk_idx[walk_start + k],
            step_number, max_walk_len, base_seed);
    }
}

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_block_global_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* sorted_walk_idx,
    const int* node_walk_starts,
    const int* node_walk_counts,
    const int* num_tasks_ptr,
    const int step_number,
    const int max_walk_len,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        node_grouped_block_global_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                sorted_walk_idx, node_walk_starts, node_walk_counts,
                num_tasks_ptr,
                step_number, max_walk_len, base_seed);
    });
}

// ==========================================================================
// Backward-walk reversal
// ==========================================================================

__global__ static void reverse_walks_kernel(
    WalkSetView walk_set, const size_t num_walks) {
    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;
    walk_set.reverse_walk(walk_idx);
}

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // TEMPORAL_RANDOM_WALK_KERNELS_NODE_GROUPED_CUH
