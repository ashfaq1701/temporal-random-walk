#include "node_edge_index.cuh"

#include "../common/memory.cuh"

HOST void node_edge_index::clear(NodeEdgeIndex* node_edge_index) {
    // Clear edge CSR structures
    clear_memory(&node_edge_index->outbound_offsets, node_edge_index->use_gpu);
    node_edge_index->outbound_offsets_size = 0;

    clear_memory(&node_edge_index->outbound_indices, node_edge_index->use_gpu);
    node_edge_index->outbound_indices_size = 0;

    clear_memory(&node_edge_index->outbound_timestamp_group_offsets, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_offsets_size = 0;

    clear_memory(&node_edge_index->outbound_timestamp_group_indices, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_indices_size = 0;

    // Clear inbound structures
    clear_memory(&node_edge_index->inbound_offsets, node_edge_index->use_gpu);
    node_edge_index->inbound_offsets_size = 0;

    clear_memory(&node_edge_index->inbound_indices, node_edge_index->use_gpu);
    node_edge_index->inbound_indices_size = 0;

    clear_memory(&node_edge_index->inbound_timestamp_group_offsets, node_edge_index->use_gpu);
    node_edge_index->inbound_timestamp_group_offsets_size = 0;

    clear_memory(&node_edge_index->inbound_timestamp_group_indices, node_edge_index->use_gpu);
    node_edge_index->inbound_timestamp_group_indices_size = 0;

    // Clear temporal weights
    clear_memory(&node_edge_index->outbound_forward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->outbound_forward_cumulative_weights_exponential_size = 0;

    clear_memory(&node_edge_index->inbound_forward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->inbound_forward_cumulative_weights_exponential_size = 0;

    clear_memory(&node_edge_index->inbound_backward_cumulative_weights_exponential, node_edge_index->use_gpu);
    node_edge_index->inbound_backward_cumulative_weights_exponential_size = 0;
}

HOST SizeRange node_edge_index::get_edge_range(const NodeEdgeIndex* node_edge_index, const int dense_node_id, const bool forward, const bool is_directed) {
    if (is_directed) {
        const size_t* offsets = forward ? node_edge_index->outbound_offsets : node_edge_index->inbound_offsets;
        size_t offsets_size = forward ? node_edge_index->outbound_offsets_size : node_edge_index->inbound_offsets_size;

        if (dense_node_id < 0 || dense_node_id >= offsets_size - 1) {
            return SizeRange{0, 0};
        }

        size_t start = 0, end = 0;
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&start, &offsets[dense_node_id], sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&end, &offsets[dense_node_id + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
        } else {
            start = offsets[dense_node_id];
            end = offsets[dense_node_id + 1];
        }

        return SizeRange{start, end};
    } else {
        if (dense_node_id < 0 || dense_node_id >= node_edge_index->outbound_offsets_size - 1) {
            return SizeRange{0, 0};
        }

        size_t start = 0, end = 0;
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&start, &node_edge_index->outbound_offsets[dense_node_id], sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&end, &node_edge_index->outbound_offsets[dense_node_id + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
        } else {
            start = node_edge_index->outbound_offsets[dense_node_id];
            end = node_edge_index->outbound_offsets[dense_node_id + 1];
        }

        return SizeRange{start, end};
    }
}

HOST SizeRange node_edge_index::get_timestamp_group_range(const NodeEdgeIndex* node_edge_index, const int dense_node_id, const size_t group_idx, const bool forward, const bool is_directed) {
    size_t* group_offsets = nullptr;
    size_t group_offsets_size = 0;
    size_t* group_indices = nullptr;
    size_t* edge_offsets = nullptr;

    if (is_directed && !forward) {
        group_offsets = node_edge_index->inbound_timestamp_group_offsets;
        group_offsets_size = node_edge_index->inbound_timestamp_group_offsets_size;
        group_indices = node_edge_index->inbound_timestamp_group_indices;
        edge_offsets = node_edge_index->inbound_offsets;
    } else {
        group_offsets = node_edge_index->outbound_timestamp_group_offsets;
        group_offsets_size = node_edge_index->outbound_timestamp_group_offsets_size;
        group_indices = node_edge_index->outbound_timestamp_group_indices;
        edge_offsets = node_edge_index->outbound_offsets;
    }

    if (dense_node_id < 0 || dense_node_id >= group_offsets_size - 1) {
        return SizeRange{0, 0};
    }

    size_t node_group_start = 0, node_group_end = 0;
    if (node_edge_index->use_gpu) {
        cudaMemcpy(&node_group_start, &group_offsets[dense_node_id], sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&node_group_end, &group_offsets[dense_node_id + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        node_group_start = group_offsets[dense_node_id];
        node_group_end = group_offsets[dense_node_id + 1];
    }

    size_t num_groups = node_group_end - node_group_start;
    if (group_idx >= num_groups) {
        return SizeRange{0, 0};
    }

    size_t group_start_idx = node_group_start + group_idx;
    size_t group_start = 0;
    if (node_edge_index->use_gpu) {
        cudaMemcpy(&group_start, &group_indices[group_start_idx], sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        group_start = group_indices[group_start_idx];
    }

    // Group end is either next group's start or node's edge range end
    size_t group_end = 0;
    if (group_idx == num_groups - 1) {
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&group_end, &edge_offsets[dense_node_id + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
        } else {
            group_end = edge_offsets[dense_node_id + 1];
        }
    } else {
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&group_end, &group_indices[group_start_idx + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
        } else {
            group_end = group_indices[group_start_idx + 1];
        }
    }

    return SizeRange{group_start, group_end};
}

HOST size_t node_edge_index::get_timestamp_group_count(const NodeEdgeIndex* node_edge_index, const int dense_node_id, const bool forward, const bool is_directed) {
    // Get the appropriate timestamp offset vector
    DataBlock<size_t> offsets_block = get_timestamp_offset_vector(node_edge_index, forward, is_directed);
    size_t* offsets = offsets_block.data;
    size_t offsets_size = offsets_block.size;

    // Check if the node ID is valid
    if (dense_node_id < 0 || dense_node_id >= offsets_size - 1) {
        return 0;
    }

    // Get start and end offsets for the node
    size_t start = 0, end = 0;
    if (node_edge_index->use_gpu) {
        cudaMemcpy(&start, &offsets[dense_node_id], sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&end, &offsets[dense_node_id + 1], sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        start = offsets[dense_node_id];
        end = offsets[dense_node_id + 1];
    }

    return end - start;
}

HOST DataBlock<size_t> node_edge_index::get_timestamp_offset_vector(const NodeEdgeIndex* node_edge_index, const bool forward, const bool is_directed) {
    if (is_directed && !forward) {
        return DataBlock<size_t>{
            node_edge_index->inbound_timestamp_group_offsets,
            node_edge_index->inbound_timestamp_group_offsets_size,
            node_edge_index->use_gpu
        };
    } else {
        return DataBlock<size_t>{
            node_edge_index->outbound_timestamp_group_offsets,
            node_edge_index->outbound_timestamp_group_offsets_size,
            node_edge_index->use_gpu
        };
    }
}

HOST void node_edge_index::allocate_node_edge_offsets(NodeEdgeIndex* node_edge_index, size_t num_nodes, bool is_directed) {
    allocate_memory(&node_edge_index->outbound_offsets, num_nodes + 1, node_edge_index->use_gpu);
    node_edge_index->outbound_offsets_size = num_nodes + 1;
    fill_memory(node_edge_index->outbound_offsets, num_nodes + 1, static_cast<size_t>(0), node_edge_index->use_gpu);

    allocate_memory(&node_edge_index->outbound_timestamp_group_offsets, num_nodes + 1, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_offsets_size = num_nodes + 1;
    fill_memory(node_edge_index->outbound_timestamp_group_offsets, num_nodes + 1, static_cast<size_t>(0), node_edge_index->use_gpu);

    // For directed graphs, also allocate inbound structures
    if (is_directed) {
        allocate_memory(&node_edge_index->inbound_offsets, num_nodes + 1, node_edge_index->use_gpu);
        node_edge_index->inbound_offsets_size = num_nodes + 1;
        fill_memory(node_edge_index->inbound_offsets, num_nodes + 1, static_cast<size_t>(0), node_edge_index->use_gpu);

        allocate_memory(&node_edge_index->inbound_timestamp_group_offsets, num_nodes + 1, node_edge_index->use_gpu);
        node_edge_index->inbound_timestamp_group_offsets_size = num_nodes + 1;
        fill_memory(node_edge_index->inbound_timestamp_group_offsets, num_nodes + 1, static_cast<size_t>(0), node_edge_index->use_gpu);
    }
}

HOST void node_edge_index::allocate_node_edge_indices(NodeEdgeIndex* node_edge_index, bool is_directed) {
    size_t num_outbound_edges = 0;
    if (node_edge_index->use_gpu) {
        // For GPU memory, we need to copy the value back to host
        cudaMemcpy(&num_outbound_edges,
                  &node_edge_index->outbound_offsets[node_edge_index->outbound_offsets_size - 1],
                  sizeof(size_t),
                  cudaMemcpyDeviceToHost);
    } else {
        // For CPU memory, we can access it directly
        num_outbound_edges = node_edge_index->outbound_offsets[node_edge_index->outbound_offsets_size - 1];
    }

    // Allocate memory for outbound indices
    allocate_memory(&node_edge_index->outbound_indices, num_outbound_edges, node_edge_index->use_gpu);
    node_edge_index->outbound_indices_size = num_outbound_edges;

    // For directed graphs, also allocate inbound indices
    if (is_directed) {
        size_t num_inbound_edges = 0;
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&num_inbound_edges,
                      &node_edge_index->inbound_offsets[node_edge_index->inbound_offsets_size - 1],
                      sizeof(size_t),
                      cudaMemcpyDeviceToHost);
        } else {
            num_inbound_edges = node_edge_index->inbound_offsets[node_edge_index->inbound_offsets_size - 1];
        }

        allocate_memory(&node_edge_index->inbound_indices, num_inbound_edges, node_edge_index->use_gpu);
        node_edge_index->inbound_indices_size = num_inbound_edges;
    }
}

HOST void node_edge_index::allocate_node_timestamp_indices(NodeEdgeIndex* node_edge_index, bool is_directed) {
    size_t num_outbound_groups = 0;
    if (node_edge_index->use_gpu) {
        // For GPU memory, we need to copy the value back to host
        cudaMemcpy(&num_outbound_groups,
                  &node_edge_index->outbound_timestamp_group_offsets[node_edge_index->outbound_timestamp_group_offsets_size - 1],
                  sizeof(size_t),
                  cudaMemcpyDeviceToHost);
    } else {
        // For CPU memory, we can access it directly
        num_outbound_groups = node_edge_index->outbound_timestamp_group_offsets[node_edge_index->outbound_timestamp_group_offsets_size - 1];
    }

    // Allocate memory for outbound timestamp group indices
    allocate_memory(&node_edge_index->outbound_timestamp_group_indices, num_outbound_groups, node_edge_index->use_gpu);
    node_edge_index->outbound_timestamp_group_indices_size = num_outbound_groups;

    // For directed graphs, also allocate inbound timestamp group indices
    if (is_directed) {
        size_t num_inbound_groups = 0;
        if (node_edge_index->use_gpu) {
            cudaMemcpy(&num_inbound_groups,
                      &node_edge_index->inbound_timestamp_group_offsets[node_edge_index->inbound_timestamp_group_offsets_size - 1],
                      sizeof(size_t),
                      cudaMemcpyDeviceToHost);
        } else {
            num_inbound_groups = node_edge_index->inbound_timestamp_group_offsets[node_edge_index->inbound_timestamp_group_offsets_size - 1];
        }

        allocate_memory(&node_edge_index->inbound_timestamp_group_indices, num_inbound_groups, node_edge_index->use_gpu);
        node_edge_index->inbound_timestamp_group_indices_size = num_inbound_groups;
    }
}

HOST void node_edge_index::populate_dense_ids_std(
    EdgeData* edge_data,
    NodeMapping* node_mapping,
    int* dense_sources,
    int* dense_targets
) {
    // Iterate through all edges
    for (size_t i = 0; i < edge_data->timestamps_size; i++) {
        // Convert sparse IDs to dense IDs using node mapping
        int sparse_src = edge_data->sources[i];
        int sparse_tgt = edge_data->targets[i];

        // Use the to_dense function from the node_mapping namespace
        dense_sources[i] = node_mapping::to_dense(node_mapping, sparse_src);
        dense_targets[i] = node_mapping::to_dense(node_mapping, sparse_tgt);
    }
}

HOST void node_edge_index::compute_node_edge_offsets_std(
    NodeEdgeIndex* node_edge_index,
    const EdgeData* edge_data,
    const int* dense_sources,
    const int* dense_targets,
    const bool is_directed
) {
    // First pass: count edges per node
    for (size_t i = 0; i < edge_data->timestamps_size; i++) {
        int src_idx = dense_sources[i];
        int tgt_idx = dense_targets[i];

        // Count outbound edges (increment the count at index src_idx + 1)
        node_edge_index->outbound_offsets[src_idx + 1]++;

        if (is_directed) {
            // For directed graphs, also increment inbound edges count
            node_edge_index->inbound_offsets[tgt_idx + 1]++;
        } else {
            // For undirected graphs, each edge appears in both directions
            node_edge_index->outbound_offsets[tgt_idx + 1]++;
        }
    }

    // Calculate prefix sums for edge offsets
    for (size_t i = 1; i < node_edge_index->outbound_offsets_size; i++) {
        node_edge_index->outbound_offsets[i] += node_edge_index->outbound_offsets[i-1];

        if (is_directed) {
            node_edge_index->inbound_offsets[i] += node_edge_index->inbound_offsets[i-1];
        }
    }
}

HOST void node_edge_index::compute_node_edge_indices_std(
    NodeEdgeIndex* node_edge_index,
    const EdgeData* edge_data,
    const int* dense_sources,
    const int* dense_targets,
    EdgeWithEndpointType* outbound_edge_indices_buffer,
    const bool is_directed
) {
    const size_t edges_size = edge_data->timestamps_size;

    // Fill the buffer with edge information
    for (size_t i = 0; i < edges_size; i++) {
        size_t outbound_index = is_directed ? i : i * 2;
        outbound_edge_indices_buffer[outbound_index] = EdgeWithEndpointType{static_cast<long>(i), true};

        if (is_directed) {
            // For directed graphs, simply assign each edge ID to inbound_indices
            node_edge_index->inbound_indices[i] = i;
        } else {
            // For undirected graphs, add each edge in both directions
            outbound_edge_indices_buffer[outbound_index + 1] = EdgeWithEndpointType{static_cast<long>(i), false};
        }
    }

    const size_t buffer_size = is_directed ? edges_size : edges_size * 2;

    // Sort outbound edge indices by node ID
    std::stable_sort(
        outbound_edge_indices_buffer,
        outbound_edge_indices_buffer + buffer_size,
        [dense_sources, dense_targets](const EdgeWithEndpointType& a, const EdgeWithEndpointType& b) {
            const int node_a = a.is_source ? dense_sources[a.edge_id] : dense_targets[a.edge_id];
            const int node_b = b.is_source ? dense_sources[b.edge_id] : dense_targets[b.edge_id];
            return node_a < node_b;
        }
    );

    // Sort inbound indices for directed graphs by target node
    if (is_directed) {
        std::stable_sort(
            node_edge_index->inbound_indices,
            node_edge_index->inbound_indices + edges_size,
            [dense_targets](size_t a, size_t b) {
                return dense_targets[a] < dense_targets[b];
            }
        );
    }

    // Extract edge IDs from buffer to outbound_indices
    for (size_t i = 0; i < buffer_size; i++) {
        node_edge_index->outbound_indices[i] = outbound_edge_indices_buffer[i].edge_id;
    }
}

HOST void node_edge_index::compute_node_timestamp_offsets_std(
    NodeEdgeIndex* node_edge_index,
    const EdgeData* edge_data,
    const size_t num_nodes,
    const bool is_directed
) {
    // Temporary arrays to store group counts for each node
    auto* outbound_group_count = new size_t[num_nodes]();  // Initialize to zeros
    size_t* inbound_group_count = nullptr;

    if (is_directed) {
        inbound_group_count = new size_t[num_nodes]();  // Initialize to zeros
    }

    // Count timestamp groups for each node
    for (size_t node = 0; node < num_nodes; node++) {
        // Process outbound groups
        size_t start = node_edge_index->outbound_offsets[node];
        size_t end = node_edge_index->outbound_offsets[node + 1];

        if (start < end) {
            outbound_group_count[node] = 1;  // First group always exists if there are edges

            for (size_t i = start + 1; i < end; ++i) {
                size_t curr_edge_id = node_edge_index->outbound_indices[i];
                size_t prev_edge_id = node_edge_index->outbound_indices[i-1];

                if (edge_data->timestamps[curr_edge_id] != edge_data->timestamps[prev_edge_id]) {
                    ++outbound_group_count[node];  // New timestamp group
                }
            }
        }

        // Process inbound groups for directed graphs
        if (is_directed) {
            start = node_edge_index->inbound_offsets[node];
            end = node_edge_index->inbound_offsets[node + 1];

            if (start < end) {
                inbound_group_count[node] = 1;  // First group always exists if there are edges

                for (size_t i = start + 1; i < end; ++i) {
                    size_t curr_edge_id = node_edge_index->inbound_indices[i];
                    size_t prev_edge_id = node_edge_index->inbound_indices[i-1];

                    if (edge_data->timestamps[curr_edge_id] != edge_data->timestamps[prev_edge_id]) {
                        ++inbound_group_count[node];  // New timestamp group
                    }
                }
            }
        }
    }

    // Calculate prefix sums for group offsets
    node_edge_index->outbound_timestamp_group_offsets[0] = 0;  // Start at 0

    for (size_t i = 0; i < num_nodes; i++) {
        node_edge_index->outbound_timestamp_group_offsets[i + 1] =
            node_edge_index->outbound_timestamp_group_offsets[i] + outbound_group_count[i];

        if (is_directed) {
            node_edge_index->inbound_timestamp_group_offsets[i + 1] =
                node_edge_index->inbound_timestamp_group_offsets[i] + inbound_group_count[i];
        }
    }

    // Clean up temporary arrays
    delete[] outbound_group_count;
    delete[] inbound_group_count;
}

HOST void node_edge_index::compute_node_timestamp_indices_std(
    NodeEdgeIndex* node_edge_index,
    const EdgeData* edge_data,
    const size_t num_nodes,
    const bool is_directed
) {
    // Process each node
    for (size_t node = 0; node < num_nodes; node++) {
        // Fill outbound timestamp group indices
        size_t start = node_edge_index->outbound_offsets[node];
        size_t end = node_edge_index->outbound_offsets[node + 1];
        size_t group_pos = node_edge_index->outbound_timestamp_group_offsets[node];

        if (start < end) {
            // First group always starts at the first edge
            node_edge_index->outbound_timestamp_group_indices[group_pos++] = start;

            for (size_t i = start + 1; i < end; ++i) {
                size_t curr_edge_id = node_edge_index->outbound_indices[i];
                size_t prev_edge_id = node_edge_index->outbound_indices[i-1];

                if (edge_data->timestamps[curr_edge_id] != edge_data->timestamps[prev_edge_id]) {
                    // New group starts at current position
                    node_edge_index->outbound_timestamp_group_indices[group_pos++] = i;
                }
            }
        }

        // Fill inbound timestamp group indices for directed graphs
        if (is_directed) {
            start = node_edge_index->inbound_offsets[node];
            end = node_edge_index->inbound_offsets[node + 1];
            group_pos = node_edge_index->inbound_timestamp_group_offsets[node];

            if (start < end) {
                // First group always starts at the first edge
                node_edge_index->inbound_timestamp_group_indices[group_pos++] = start;

                for (size_t i = start + 1; i < end; ++i) {
                    size_t curr_edge_id = node_edge_index->inbound_indices[i];
                    size_t prev_edge_id = node_edge_index->inbound_indices[i-1];

                    if (edge_data->timestamps[curr_edge_id] != edge_data->timestamps[prev_edge_id]) {
                        // New group starts at current position
                        node_edge_index->inbound_timestamp_group_indices[group_pos++] = i;
                    }
                }
            }
        }
    }
}
