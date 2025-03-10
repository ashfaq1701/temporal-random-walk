#include "edge_data.cuh"

#include "../common/memory.cuh"

HOST void edge_data::reserve(EdgeData *edge_data, const size_t size) {
    allocate_memory(&edge_data->sources, size, edge_data->use_gpu);
    allocate_memory(&edge_data->targets, size, edge_data->use_gpu);
    allocate_memory(&edge_data->timestamps, size, edge_data->use_gpu);

    allocate_memory(&edge_data->timestamp_group_offsets, size, edge_data->use_gpu);
    allocate_memory(&edge_data->unique_timestamps, size, edge_data->use_gpu);
}

HOST void edge_data::clear(EdgeData *edge_data) {
    clear_memory(&edge_data->sources, edge_data->use_gpu);
    clear_memory(&edge_data->targets, edge_data->use_gpu);
    clear_memory(&edge_data->timestamps, edge_data->use_gpu);

    clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
    clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);

    clear_memory(&edge_data->forward_cumulative_weights_exponential, edge_data->use_gpu);
    clear_memory(&edge_data->backward_cumulative_weights_exponential, edge_data->use_gpu);
}

HOST size_t edge_data::size(const EdgeData *edge_data) {
    return edge_data->timestamps_size;
}

HOST bool edge_data::empty(const EdgeData *edge_data) {
    return edge_data->timestamps_size == 0;
}
