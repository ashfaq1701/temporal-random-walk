#ifndef NODE_GROUPED_KERNELS_PER_WALK_CUH
#define NODE_GROUPED_KERNELS_PER_WALK_CUH

// Single-walk-per-thread kernels: start edges, advance_one_walk,
// solo wrapper, Node2Vec per-walk step, backward-walk reversal.

#include "../../../data/walk_set/walk_set_view.cuh"
#include "../../../data/temporal_graph_view.cuh"
#include "../../../graph/temporal_graph.cuh"
#include "../../../graph/edge_selectors.cuh"
#include "../../../utils/random.cuh"
#include "../../../utils/utils.cuh"
#include "../../../common/picker_dispatch.cuh"
#include "../../helpers.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// Step 0 is the start step: pick_start_edges_kernel (unconstrained) or the
// step-0 coop pass (constrained walks-per-node) keys Philox at offset 0.
// pick_start_edges_kernel uses up to 3 draws (2 directed / 3 undirected, the
// 3rd picks an endpoint); the constrained coop start pass uses 2. Either way
// the start step's substream is contained in [0, START_KERNEL_BUDGET=3).
// Step k>=1 starts at START_KERNEL_BUDGET + k*2, leaving offsets 3,4 unused
// so intermediate steps stay on the same substream regardless of tier or
// constrained/unconstrained mode.
DEVICE __forceinline__ uint64_t step_kernel_philox_offset(const int step_number) {
    if (step_number == 0) return 0ULL;
    constexpr uint64_t START_KERNEL_BUDGET = 3ULL;
    return START_KERNEL_BUDGET + static_cast<uint64_t>(step_number) * 2ULL;
}

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

    // No edge satisfies the constraint → walk stays at len 0, filter drops it.
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
        // 3rd Philox draw here matches START_KERNEL_BUDGET=3 so all tiers stay bit-exact.
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

// Constrained walks-per-node start: write slot 0 = (start_node_ids[walk_idx],
// sentinel_ts) and set walk_lens[walk_idx] = 1. The step-0 coop pass then
// reads slot 0, picks the first edge under start_picker_type, and the
// resulting hop is appended to slot 1 via add_hop.
//
// Sentinel timestamp matches pick_start_edges_kernel (Forward: INT64_MIN,
// Backward: INT64_MAX) so user-visible slot-0 timestamps stay identical to
// the per-walk start path. filter_valid_groups_by_timestamp_slice's -1 fast
// path doesn't fire here, but the binary-search path returns [0, G) for
// either sentinel — semantically identical, log G probes of overhead at
// step 0 only.
template <bool Forward>
__global__ void prepopulate_start_slot_kernel(
    WalkSetView walk_set,
    const int* __restrict__ start_node_ids,
    const size_t num_walks) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    constexpr int64_t sentinel_timestamp = Forward ? INT64_MIN : INT64_MAX;
    walk_set.add_hop(walk_idx,
                     start_node_ids[walk_idx],
                     sentinel_timestamp);
}

template <bool Forward>
inline void dispatch_prepopulate_start_slot_kernel(
    WalkSetView walk_set_view,
    const int* start_node_ids,
    const size_t num_walks,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    prepopulate_start_slot_kernel<Forward>
        <<<grid, block_dim, 0, stream>>>(
            walk_set_view, start_node_ids, num_walks);
}

// Advance one walk by one step. Outcome depends only on
// (walk_idx, step_number, base_seed) so any tier produces the same walk.
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

// Solo tier: one thread per walk in solo_walks.
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

// Node2Vec path: one thread per walk, no scheduler. prev_node-dependent
// CDF makes coop panels useless. Dead walks no-op inside is_node_active.
template <bool IsDirected, bool Forward, RandomPickerType EdgePickerType>
__global__ void per_walk_step_kernel(
    TemporalGraphView view,
    WalkSetView walk_set,
    const int step_number,
    const int max_walk_len,
    const size_t num_walks,
    const uint64_t base_seed) {

    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    advance_one_walk<IsDirected, Forward, EdgePickerType>(
        view, walk_set, static_cast<int>(walk_idx),
        step_number, max_walk_len, base_seed);
}

template <bool IsDirected, bool Forward>
inline void dispatch_per_walk_step_kernel(
    const TemporalGraphView& view,
    WalkSetView walk_set_view,
    const int step_number,
    const int max_walk_len,
    const size_t num_walks,
    const RandomPickerType edge_picker_type,
    const uint64_t base_seed,
    const dim3& grid,
    const dim3& block_dim,
    const cudaStream_t stream) {

    dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
        constexpr auto kEdge = decltype(edge_tag)::value;
        per_walk_step_kernel<IsDirected, Forward, kEdge>
            <<<grid, block_dim, 0, stream>>>(
                view, walk_set_view,
                step_number, max_walk_len, num_walks, base_seed);
    });
}

__global__ static void reverse_walks_kernel(
    WalkSetView walk_set, const size_t num_walks) {
    const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;
    walk_set.reverse_walk(walk_idx);
}

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // NODE_GROUPED_KERNELS_PER_WALK_CUH
