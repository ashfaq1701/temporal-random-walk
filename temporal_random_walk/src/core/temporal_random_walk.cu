#include "temporal_random_walk.cuh"

HOST void temporal_random_walk::add_multiple_edges(const TemporalRandomWalk* temporal_random_walk, const Edge* edge_infos, const size_t num_edges) {
    if (temporal_random_walk->use_gpu) {
        temporal_graph::add_multiple_edges_cuda(temporal_random_walk->temporal_graph, edge_infos, num_edges);
    } else {
        temporal_graph::add_multiple_edges_std(temporal_random_walk->temporal_graph, edge_infos, num_edges);
    }
}

HOST size_t temporal_random_walk::get_node_count(const TemporalRandomWalk* temporal_random_walk) {
    return temporal_graph::get_node_count(temporal_random_walk->temporal_graph);
}

HOST size_t temporal_random_walk::get_edge_count(const TemporalRandomWalk* temporal_random_walk) {
    return temporal_graph::get_total_edges(temporal_random_walk->temporal_graph);
}

HOST DataBlock<int> temporal_random_walk::get_node_ids(const TemporalRandomWalk* temporal_random_walk) {
    return temporal_graph::get_node_ids(temporal_random_walk->temporal_graph);
}

HOST DataBlock<Edge> temporal_random_walk::get_edges(const TemporalRandomWalk* temporal_random_walk) {
    return temporal_graph::get_edges(temporal_random_walk->temporal_graph);
}

HOST bool temporal_random_walk::get_is_directed(const TemporalRandomWalk* temporal_random_walk) {
    return temporal_random_walk->is_directed;
}

HOST void temporal_random_walk::clear(TemporalRandomWalk* temporal_random_walk) {
   temporal_random_walk->temporal_graph = new TemporalGraph(
       temporal_random_walk->is_directed,
       temporal_random_walk->use_gpu,
       temporal_random_walk->max_time_capacity,
       temporal_random_walk->enable_weight_computation,
       temporal_random_walk->timescale_bound);
}
