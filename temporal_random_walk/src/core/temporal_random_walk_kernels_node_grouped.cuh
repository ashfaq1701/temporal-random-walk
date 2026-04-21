#ifndef TEMPORAL_RANDOM_WALK_KERNELS_NODE_GROUPED_CUH
#define TEMPORAL_RANDOM_WALK_KERNELS_NODE_GROUPED_CUH

#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../graph/temporal_graph.cuh"
#include "../graph/edge_selectors.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "../common/picker_dispatch.cuh"
#include "../common/cuda_config.cuh"
#include "../common/warp_coop_config.cuh"
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
// Sort-and-group infrastructure kernels
//
// These are shared by both the step-0 grouping (by start_node_id) and the
// intermediate-step grouping (by last_node at step S). They stay here rather
// than in src/common because they are intimately tied to walk-set semantics.
// ==========================================================================

// 0, 1, ..., n-1 — seeds the values buffer that rides a sort-by-key pass so
// the sorted permutation is recoverable.
__global__ static void iota_int_kernel(int* __restrict__ out, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i;
}

// For each walk, flag whether the slot at step_number holds a live last-node
// (anything other than the configured walk_padding_value). Terminated walks
// fail this test and get compacted out by cub::DevicePartition::Flagged
// before sort-and-group runs for the step.
__global__ static void walk_alive_flags_kernel(
    WalkSetView walk_set,
    const int step_number,
    const int max_walk_len,
    const size_t num_walks,
    uint8_t* __restrict__ alive_flags) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    const size_t offset = walk_idx * static_cast<size_t>(max_walk_len)
                          + static_cast<size_t>(step_number);
    alive_flags[walk_idx] =
        (walk_set.nodes[offset] != walk_set.walk_padding_value) ? uint8_t{1} : uint8_t{0};
}

// Gather last-node keys for the already-compacted active walks; the
// subsequent sort keys on last_node. active_walk_idx[i] is an *original*
// walk index preserved end-to-end so every downstream kernel addresses the
// correct per-walk slot.
__global__ static void gather_last_nodes_kernel(
    WalkSetView walk_set,
    const int* __restrict__ active_walk_idx,
    const int* __restrict__ num_active_ptr,
    const int step_number,
    const int max_walk_len,
    int* __restrict__ last_nodes_out) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_active = *num_active_ptr;
    if (i >= num_active) return;

    const int walk_idx = active_walk_idx[i];
    const size_t offset = static_cast<size_t>(walk_idx) * static_cast<size_t>(max_walk_len)
                          + static_cast<size_t>(step_number);
    last_nodes_out[i] = walk_set.nodes[offset];
}

// Binary-search-based scatter: for each sorted slot, find its run via
// run_starts and write run_length back into walk_to_group_size at the
// original walk's index. The num_items_ptr parameter is the count of sorted
// slots, which for step 0 equals num_walks and for intermediate steps
// equals *num_active_ptr — pass the right device-side count so the scatter
// stops at the compacted length.
__global__ static void scatter_walk_group_sizes_kernel(
    const int* __restrict__ sorted_walk_idx,
    const int* __restrict__ run_starts,
    const int* __restrict__ run_lengths,
    const int* __restrict__ num_runs_ptr,
    const int* __restrict__ num_items_ptr,
    int* walk_to_group_size,
    const int num_walks) {

    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_items = *num_items_ptr;
    if (slot >= num_items) return;

    const int num_runs = *num_runs_ptr;
    if (num_runs <= 0) return;

    // Largest r with run_starts[r] <= slot.
    int lo = 0, hi = num_runs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (run_starts[mid] <= slot) lo = mid;
        else                         hi = mid - 1;
    }

    const int walk_idx = sorted_walk_idx[slot];
    if (walk_idx < 0 || walk_idx >= num_walks) return;
    walk_to_group_size[walk_idx] = run_lengths[lo];
}

// Zero-fill an int buffer. Used on walk_to_group_size before every step so
// walks that don't appear in the scatter (terminated walks dropped by the
// compaction pass) read 0 — solo kernel's group-size guard treats 0 and 1
// the same way.
__global__ static void zero_int_buffer_kernel(int* __restrict__ buf, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = 0;
}

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
// Solo kernel — services every active walk in the per-step pipeline.
//
// One thread per active walk. Receives compacted/sorted walk indices from
// the scheduler. In the current (pre-scheduler-extraction) tree, "solo"
// is the only kernel — the coop tiers are landing in tasks 3+. At that
// point this kernel will service the solo_walks list only (W=1). For now
// it services every active walk.
// ==========================================================================

template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void node_grouped_solo_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int* __restrict__ walk_idx_list,
    const int* __restrict__ num_active_ptr,
    const int step_number,
    const int max_walk_len,
    const uint64_t base_seed) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_active = *num_active_ptr;
    if (i >= num_active) return;

    if (step_number >= max_walk_len - 1) return;

    const int walk_idx_int = walk_idx_list[i];
    if (walk_idx_int < 0) return;

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

template <bool IsDirected, bool Forward>
inline void dispatch_node_grouped_solo_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int* walk_idx_list,
    const int* num_active_ptr,
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
                walk_idx_list, num_active_ptr,
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
