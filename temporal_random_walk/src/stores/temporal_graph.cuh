#ifndef TEMPORAL_GRAPH_H
#define TEMPORAL_GRAPH_H

#include "edge_data.cuh"
#include "node_edge_index.cuh"
#include "node_mapping.cuh"

struct TemporalGraph {
    bool use_gpu;

    EdgeData* edge_data;
    NodeEdgeIndex* node_edge_index;
    NodeMapping* node_mapping;

    explicit TemporalGraph(const bool use_gpu): use_gpu(use_gpu) {
        edge_data = new EdgeData(use_gpu);
        node_edge_index = new NodeEdgeIndex(use_gpu);
        node_mapping = new NodeMapping(use_gpu);
    }
};

namespace temporal_graph {

}

#endif // TEMPORAL_GRAPH_H