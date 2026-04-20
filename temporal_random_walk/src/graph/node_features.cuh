#ifndef NODE_FEATURES_CUH
#define NODE_FEATURES_CUH

#include <cstddef>

#include "../common/macros.cuh"
#include "../data/temporal_graph_data.cuh"

// STAGING FILE for task 5c. Not in CMake. Swapped in by task 5g.

namespace node_features {

    HOST inline size_t size(const TemporalGraphData& data) {
        return data.node_features.size();
    }

    HOST inline bool empty(const TemporalGraphData& data) {
        return data.node_features.size() == 0;
    }

    HOST void ensure_size(TemporalGraphData& data, int max_node_id);

    HOST void set_node_features(
        TemporalGraphData& data,
        int max_node_id,
        const int* node_ids,
        size_t num_nodes,
        const float* node_features_src,
        size_t feature_dim);
}

#endif // NODE_FEATURES_CUH
