#include "node_mapping.cuh"

#include "../common/memory.cuh"


HOST void node_mapping::update_std(
    NodeMapping *node_mapping,
    const EdgeData *edge_data,
    const size_t start_idx,
    const size_t end_idx) {

    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({
            max_node_id,
            static_cast<int>(edge_data->sources[i]),
            static_cast<int>(edge_data->targets[i])
        });
    }

    // Extend sparse_to_dense and is_deleted if needed
    if (max_node_id >= node_mapping->sparse_to_dense_size) {
        // Allocate new larger arrays
        const size_t new_size = max_node_id + 1;

        int* new_sparse_to_dense = nullptr;
        bool* new_is_deleted = nullptr;

        allocate_memory(&new_sparse_to_dense, new_size, node_mapping->use_gpu);
        allocate_memory(&new_is_deleted, new_size, node_mapping->use_gpu);

        // Copy existing data
        if (node_mapping->sparse_to_dense_size > 0) {
            memcpy(new_sparse_to_dense, node_mapping->sparse_to_dense,
                   node_mapping->sparse_to_dense_size * sizeof(int));
            memcpy(new_is_deleted, node_mapping->is_deleted,
                   node_mapping->is_deleted_size * sizeof(bool));
        }

        // Initialize new elements
        for (size_t i = node_mapping->sparse_to_dense_size; i < new_size; i++) {
            new_sparse_to_dense[i] = -1;
            new_is_deleted[i] = true;
        }

        // Free old arrays
        clear_memory(&node_mapping->sparse_to_dense, node_mapping->use_gpu);
        clear_memory(&node_mapping->is_deleted, node_mapping->use_gpu);

        // Update pointers and sizes
        node_mapping->sparse_to_dense = new_sparse_to_dense;
        node_mapping->sparse_to_dense_size = new_size;
        node_mapping->is_deleted = new_is_deleted;
        node_mapping->is_deleted_size = new_size;
    }

    // Collect all nodes from the edges
    std::vector<int> new_nodes;
    new_nodes.reserve((end_idx - start_idx) * 2);

    for (size_t i = start_idx; i < end_idx; i++) {
        new_nodes.push_back(edge_data->sources[i]);
        new_nodes.push_back(edge_data->targets[i]);
    }

    // Sort and remove duplicates
    std::sort(new_nodes.begin(), new_nodes.end());
    new_nodes.erase(std::unique(new_nodes.begin(), new_nodes.end()), new_nodes.end());

    // Count how many new mappings we need
    size_t new_mappings_count = 0;
    for (int node : new_nodes) {
        if (node >= 0 && node_mapping->sparse_to_dense[node] == -1) {
            new_mappings_count++;
        }
    }

    // Resize dense_to_sparse once to accommodate all new mappings
    if (new_mappings_count > 0) {
        const size_t new_dense_size = node_mapping->dense_to_sparse_size + new_mappings_count;
        resize_memory(
            &node_mapping->dense_to_sparse,
            node_mapping->dense_to_sparse_size,
            new_dense_size,
            node_mapping->use_gpu
        );
    }

    // Map unmapped nodes
    size_t dense_index = node_mapping->dense_to_sparse_size;
    for (const int node : new_nodes) {
        if (node < 0) continue;

        // Mark as not deleted
        node_mapping->is_deleted[node] = false;

        // If not mapped yet, add to dense_to_sparse and update sparse_to_dense
        if (node_mapping->sparse_to_dense[node] == -1) {
            // Update sparse_to_dense mapping
            node_mapping->sparse_to_dense[node] = static_cast<int>(dense_index);

            // Add the node to dense_to_sparse
            node_mapping->dense_to_sparse[dense_index] = node;
            dense_index++;
        }
    }

    // Update the dense_to_sparse size
    node_mapping->dense_to_sparse_size = dense_index;
}


