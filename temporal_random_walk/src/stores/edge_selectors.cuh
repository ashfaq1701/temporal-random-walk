#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include "temporal_graph.cuh"

namespace temporal_graph {

    /**
     * Host functions
     */

    HOST Edge get_edge_at_host(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

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

    DEVICE Edge get_edge_at_device(
        const TemporalGraphStore* graph,
        RandomPickerType picker_type,
        int64_t timestamp,
        bool forward,
        double group_selector_rand_num,
        double edge_selector_rand_num);

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
