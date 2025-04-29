#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include "temporal_graph.cuh"

namespace temporal_graph {

    /**
     * HOST FUNCTIONS
     */

    /**
     * ***********
     * get_edge_at
     * ***********
     */

    // Bounded, forward, index-based picker
    HOST Edge get_edge_at_bounded_forward_index_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, forward, weight-based picker
    HOST Edge get_edge_at_bounded_forward_weight_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, backward, index-based picker
    HOST Edge get_edge_at_bounded_backward_index_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, backward, weight-based picker
    HOST Edge get_edge_at_bounded_backward_weight_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, forward, index-based picker
    HOST Edge get_edge_at_unbounded_forward_index_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, forward, weight-based picker
    HOST Edge get_edge_at_unbounded_forward_weight_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, backward, index-based picker
    HOST Edge get_edge_at_unbounded_backward_index_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, backward, weight-based picker
    HOST Edge get_edge_at_unbounded_backward_weight_based_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Router function to select the appropriate specialized function
    HOST Edge get_edge_at_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    /**
     * ***********
     * get_node_edge_at
     * ***********
     */

    // Directed graph, with timestamp constraint, forward traversal, index-based picker
    HOST Edge get_node_edge_at_directed_bounded_forward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, forward traversal, weight-based picker
    HOST Edge get_node_edge_at_directed_bounded_forward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, backward traversal, index-based picker
    HOST Edge get_node_edge_at_directed_bounded_backward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, backward traversal, weight-based picker
    HOST Edge get_node_edge_at_directed_bounded_backward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, forward traversal, index-based picker
    HOST Edge get_node_edge_at_directed_unbounded_forward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, forward traversal, weight-based picker
    HOST Edge get_node_edge_at_directed_unbounded_forward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, backward traversal, index-based picker
    HOST Edge get_node_edge_at_directed_unbounded_backward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, backward traversal, weight-based picker
    HOST Edge get_node_edge_at_directed_unbounded_backward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, forward traversal, index-based picker
    HOST Edge get_node_edge_at_undirected_bounded_forward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, forward traversal, weight-based picker
    HOST Edge get_node_edge_at_undirected_bounded_forward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, backward traversal, index-based picker
    HOST Edge get_node_edge_at_undirected_bounded_backward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, backward traversal, weight-based picker
    HOST Edge get_node_edge_at_undirected_bounded_backward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, forward traversal, index-based picker
    HOST Edge get_node_edge_at_undirected_unbounded_forward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, forward traversal, weight-based picker
    HOST Edge get_node_edge_at_undirected_unbounded_forward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, backward traversal, index-based picker
    HOST Edge get_node_edge_at_undirected_unbounded_backward_index_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, backward traversal, weight-based picker
    HOST Edge get_node_edge_at_undirected_unbounded_backward_weight_based_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Router function to select the appropriate specialized function
    HOST Edge get_node_edge_at_host(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    /**
     * Device functions
     */

    #ifdef HAS_CUDA

    /**
     * ***********
     * get_edge_at
     * ***********
     */

    // Bounded, forward, index-based picker
    DEVICE Edge get_edge_at_bounded_forward_index_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, forward, weight-based picker
    DEVICE Edge get_edge_at_bounded_forward_weight_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, backward, index-based picker
    DEVICE Edge get_edge_at_bounded_backward_index_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Bounded, backward, weight-based picker
    DEVICE Edge get_edge_at_bounded_backward_weight_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, forward, index-based picker
    DEVICE Edge get_edge_at_unbounded_forward_index_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, forward, weight-based picker
    DEVICE Edge get_edge_at_unbounded_forward_weight_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, backward, index-based picker
    DEVICE Edge get_edge_at_unbounded_backward_index_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Unbounded, backward, weight-based picker
    DEVICE Edge get_edge_at_unbounded_backward_weight_based_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Router function to select the appropriate specialized function
    DEVICE Edge get_edge_at_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    /**
     * ***********
     * get_node_edge_at
     * ***********
     */

    // Directed graph, with timestamp constraint, forward traversal, index-based picker
    DEVICE Edge get_node_edge_at_directed_bounded_forward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, forward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_directed_bounded_forward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, backward traversal, index-based picker
    DEVICE Edge get_node_edge_at_directed_bounded_backward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, with timestamp constraint, backward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_directed_bounded_backward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, forward traversal, index-based picker
    DEVICE Edge get_node_edge_at_directed_unbounded_forward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, forward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_directed_unbounded_forward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, backward traversal, index-based picker
    DEVICE Edge get_node_edge_at_directed_unbounded_backward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Directed graph, no timestamp constraint, backward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_directed_unbounded_backward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, forward traversal, index-based picker
    DEVICE Edge get_node_edge_at_undirected_bounded_forward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, forward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_undirected_bounded_forward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, backward traversal, index-based picker
    DEVICE Edge get_node_edge_at_undirected_bounded_backward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, with timestamp constraint, backward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_undirected_bounded_backward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, forward traversal, index-based picker
    DEVICE Edge get_node_edge_at_undirected_unbounded_forward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, forward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_undirected_unbounded_forward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, backward traversal, index-based picker
    DEVICE Edge get_node_edge_at_undirected_unbounded_backward_index_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Undirected graph, no timestamp constraint, backward traversal, weight-based picker
    DEVICE Edge get_node_edge_at_undirected_unbounded_backward_weight_based_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    // Router function to select the appropriate specialized function
    DEVICE Edge get_node_edge_at_device(
        TemporalGraphStore* graph,
        int node_id,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

    #endif

}

#endif //EDGE_SELECTORS_CUH
