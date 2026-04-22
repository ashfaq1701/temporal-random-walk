#ifndef NODE_GROUPED_KERNELS_COMMON_CUH
#define NODE_GROUPED_KERNELS_COMMON_CUH

// Shared helpers for the node-grouped cooperative kernels. Anything that
// the block and warp tiers both need (but that isn't per-walk sampling
// logic) lives here.

#include "../../../common/macros.cuh"
#include "../../../data/temporal_graph_view.cuh"

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
    size_t        node_ts_sorted_indices_size;
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

    p.node_ts_sorted_indices_size =
        Forward ? view.node_ts_sorted_outbound_indices_size
                : (IsDirected ? view.node_ts_sorted_inbound_indices_size
                              : view.node_ts_sorted_outbound_indices_size);

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

#endif // HAS_CUDA

} // namespace temporal_random_walk

#endif // NODE_GROUPED_KERNELS_COMMON_CUH
