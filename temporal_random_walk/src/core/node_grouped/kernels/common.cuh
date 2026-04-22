#ifndef NODE_GROUPED_KERNELS_COMMON_CUH
#define NODE_GROUPED_KERNELS_COMMON_CUH

// Shared helpers for the node-grouped cooperative kernels. Anything that
// the block and warp tiers both need (but that isn't per-walk sampling
// logic) lives here.

#include "../../../common/macros.cuh"
#include "../../../common/warp_coop_config.cuh"
#include "../../../data/temporal_graph_view.cuh"
#include "../../../data/walk_set/walk_set_view.cuh"
#include "../../../random/pickers.cuh"
#include "../../../utils/random.cuh"
#include "../../../utils/utils.cuh"

namespace temporal_random_walk {

#ifdef HAS_CUDA

// Direction-dependent per-node pointers the cooperative kernels need to
// sample one hop out of the current node. Resolved once at task entry
// (using compile-time tags) so the stride loop doesn't re-do the
// Forward ? outbound : (IsDirected ? inbound : outbound) ternary chain
// on every walk.
//
// Matches the pointer set get_node_edge_at_device pulls off the view.
struct NodeDirPtrs {
    const size_t* count_ts_group_per_node;
    const size_t* node_ts_groups_offsets;
    const size_t* node_ts_sorted_indices;
    const size_t* node_edge_offsets;
    const double* weights;
    size_t        weights_size;
};

template <bool IsDirected, bool Forward>
DEVICE __forceinline__ NodeDirPtrs
resolve_node_dir_ptrs(const TemporalGraphView& view) {
    NodeDirPtrs p;

    p.count_ts_group_per_node =
        Forward ? view.count_ts_group_per_node_outbound
                : (IsDirected ? view.count_ts_group_per_node_inbound
                              : view.count_ts_group_per_node_outbound);

    p.node_ts_groups_offsets =
        Forward ? view.node_ts_group_outbound_offsets
                : (IsDirected ? view.node_ts_group_inbound_offsets
                              : view.node_ts_group_outbound_offsets);

    p.node_ts_sorted_indices =
        Forward ? view.node_ts_sorted_outbound_indices
                : (IsDirected ? view.node_ts_sorted_inbound_indices
                              : view.node_ts_sorted_outbound_indices);

    p.node_edge_offsets =
        Forward ? view.node_group_outbound_offsets
                : (IsDirected ? view.node_group_inbound_offsets
                              : view.node_group_outbound_offsets);

    p.weights =
        Forward ? view.outbound_forward_cumulative_weights_exponential
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential
                              : view.outbound_backward_cumulative_weights_exponential);

    p.weights_size =
        Forward ? view.outbound_forward_cumulative_weights_exponential_size
                : (IsDirected ? view.inbound_backward_cumulative_weights_exponential_size
                              : view.outbound_backward_cumulative_weights_exponential_size);

    return p;
}

// ==========================================================================
// Picker-class -> G cap for the smem panel tiers. The scheduler's
// G-partition guarantees each cooperative-smem task's G is <= its tier cap.
// ==========================================================================
template <RandomPickerType PickerType>
HOST DEVICE constexpr inline int coop_block_smem_g_cap() {
    return random_pickers::is_index_based_picker_v<PickerType>
               ? TRW_NODE_GROUPED_G_CAP_BLOCK_INDEX
               : TRW_NODE_GROUPED_G_CAP_BLOCK_WEIGHTED;
}

template <RandomPickerType PickerType>
HOST DEVICE constexpr inline int coop_warp_smem_g_cap() {
    return random_pickers::is_index_based_picker_v<PickerType>
               ? TRW_NODE_GROUPED_G_CAP_WARP_INDEX
               : TRW_NODE_GROUPED_G_CAP_WARP_WEIGHTED;
}

// ==========================================================================
// Stage 2b+2c — shared per-walk tail across all four cooperative kernels.
//
// Given a local group position (in [0, G)) and an offset pointer addressing
// the per-node ts-group-offsets slice (either the smem panel preloaded in
// smem kernels, or the (global + node_group_begin) slice pointer in global
// kernels), resolves the selected group's edge range, picks a uniform
// random edge within it, and appends the hop to walk_set.
//
// `offsets_slice` is indexed in [0, G] — offsets_slice[local_pos] is the
// first edge of the selected group, offsets_slice[local_pos + 1] is the
// first edge of the next group (or node_edge_end if local_pos is the last
// group). Both smem and (global + begin) pointers satisfy this contract.
//
// Returns silently on degenerate / empty edge range — the walk simply
// doesn't advance this step and its walk_lens isn't incremented.
// ==========================================================================
template <bool IsDirected, bool Forward>
DEVICE __forceinline__ void sample_edge_and_add_hop(
    const TemporalGraphView& view,
    WalkSetView              walk_set,
    const NodeDirPtrs&       ptrs,
    const size_t*            offsets_slice,
    const long               local_pos,
    const int                G,
    const size_t             node_edge_end,
    const int                node_id,
    const size_t             walk_idx,
    const double             r_edge) {

    const size_t valid_edge_start = offsets_slice[local_pos];
    const size_t valid_edge_end =
        (local_pos + 1 < G)
            ? offsets_slice[local_pos + 1]
            : node_edge_end;
    if (valid_edge_start >= valid_edge_end) return;

    const long edge_idx = static_cast<long>(ptrs.node_ts_sorted_indices[
        valid_edge_start +
        generate_random_number_bounded_by(
            static_cast<int>(valid_edge_end - valid_edge_start),
            r_edge)]);

    if constexpr (IsDirected) {
        walk_set.add_hop(walk_idx,
                         Forward ? view.targets[edge_idx]
                                 : view.sources[edge_idx],
                         view.timestamps[edge_idx],
                         edge_idx);
    } else {
        const int next_node = pick_other_number(
            view.sources[edge_idx], view.targets[edge_idx], node_id);
        walk_set.add_hop(walk_idx, next_node,
                         view.timestamps[edge_idx], edge_idx);
    }
}

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // NODE_GROUPED_KERNELS_COMMON_CUH
