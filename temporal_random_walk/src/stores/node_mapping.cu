#include "node_mapping.cuh"

#include "../common/memory.cuh"
#include <common/cuda_config.cuh>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

HOST int node_mapping::to_dense(const NodeMapping *node_mapping, const int sparse_id) {
    if (sparse_id < 0 || sparse_id >= node_mapping->sparse_to_dense_size) {
        return -1;
    }
    return node_mapping->sparse_to_dense[sparse_id];
}

HOST int node_mapping::to_sparse(const NodeMapping *node_mapping, const int dense_id) {
    if (dense_id < 0 || dense_id >= node_mapping->dense_to_sparse_size) {
        return -1;
    }
    return node_mapping->dense_to_sparse[dense_id];
}

HOST size_t node_mapping::size(const NodeMapping *node_mapping) {
    return node_mapping->dense_to_sparse_size;
}

HOST size_t node_mapping::active_size(const NodeMapping *node_mapping) {
    size_t count = 0;
    for (size_t i = 0; i < node_mapping->is_deleted_size; i++) {
        if (!node_mapping->is_deleted[i]) {
            count++;
        }
    }
    return count;
}

HOST DataBlock<int> node_mapping::get_active_node_ids(const NodeMapping *node_mapping) {
    const size_t active_count = active_size(node_mapping);
    DataBlock<int> result(active_count, false);

    size_t index = 0;
    for (size_t i = 0; i < node_mapping->dense_to_sparse_size; i++) {
        int sparse_id = node_mapping->dense_to_sparse[i];
        if (sparse_id >= 0 && sparse_id < node_mapping->is_deleted_size && !node_mapping->is_deleted[sparse_id]) {
            result.data[index++] = sparse_id;
        }
    }

    return result;
}

HOST void node_mapping::clear(NodeMapping *node_mapping) {
    clear_memory(&node_mapping->sparse_to_dense, node_mapping->use_gpu);
    node_mapping->sparse_to_dense_size = 0;

    clear_memory(&node_mapping->dense_to_sparse, node_mapping->use_gpu);
    node_mapping->dense_to_sparse_size = 0;

    clear_memory(&node_mapping->is_deleted, node_mapping->use_gpu);
    node_mapping->is_deleted_size = 0;
}

HOST void node_mapping::reserve(NodeMapping *node_mapping, size_t size) {
    if (size > node_mapping->sparse_to_dense_size) {
        resize_memory(&node_mapping->sparse_to_dense, node_mapping->sparse_to_dense_size, size, node_mapping->use_gpu);
        node_mapping->sparse_to_dense_size = size;
    }

    if (size > node_mapping->dense_to_sparse_size) {
        resize_memory(&node_mapping->dense_to_sparse, node_mapping->dense_to_sparse_size, size, node_mapping->use_gpu);
        node_mapping->dense_to_sparse_size = size;
    }

    if (size > node_mapping->is_deleted_size) {
        resize_memory(&node_mapping->is_deleted, node_mapping->is_deleted_size, size, node_mapping->use_gpu);
        node_mapping->is_deleted_size = size;
    }
}

HOST void node_mapping::mark_node_deleted(const NodeMapping *node_mapping, const int sparse_id) {
    if (sparse_id >= 0 && sparse_id < node_mapping->is_deleted_size) {
        node_mapping->is_deleted[sparse_id] = true;
    }
}

HOST MemoryView<int> node_mapping::get_all_sparse_ids(const NodeMapping *node_mapping) {
    return MemoryView{node_mapping->dense_to_sparse, node_mapping->dense_to_sparse_size};
}
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
            edge_data->sources[i],
            edge_data->targets[i]
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

HOST void node_mapping::update_cuda(NodeMapping *node_mapping, const EdgeData *edge_data, const size_t start_idx, const size_t end_idx) {
    // Find maximum node ID
    thrust::device_ptr<int> d_sources(edge_data->sources + start_idx);
    thrust::device_ptr<int> d_targets(edge_data->targets + start_idx);

    auto max_source = thrust::max_element(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(end_idx - start_idx)
    );

    auto max_target = thrust::max_element(
        DEVICE_EXECUTION_POLICY,
        d_targets,
        d_targets + static_cast<long>(end_idx - start_idx)
    );

    int max_source_value = 0;
    int max_target_value = 0;

    if (max_source != d_sources + static_cast<long>(end_idx - start_idx)) {
        cudaMemcpy(&max_source_value, edge_data->sources + start_idx + (max_source - d_sources),
                  sizeof(int), cudaMemcpyDeviceToHost);
    }

    if (max_target != d_targets + static_cast<long>(end_idx - start_idx)) {
        cudaMemcpy(&max_target_value, edge_data->targets + start_idx + (max_target - d_targets),
                  sizeof(int), cudaMemcpyDeviceToHost);
    }

    int max_node_id = std::max(max_source_value, max_target_value);

    if (max_node_id < 0) {
        return;
    }

    // Extend arrays if needed
    if (max_node_id >= node_mapping->sparse_to_dense_size) {
        size_t new_size = max_node_id + 1;

        // Allocate new arrays
        int* new_sparse_to_dense = nullptr;
        bool* new_is_deleted = nullptr;

        cudaMalloc(&new_sparse_to_dense, new_size * sizeof(int));
        cudaMalloc(&new_is_deleted, new_size * sizeof(bool));

        // Copy existing data
        if (node_mapping->sparse_to_dense_size > 0) {
            cudaMemcpy(new_sparse_to_dense, node_mapping->sparse_to_dense,
                      node_mapping->sparse_to_dense_size * sizeof(int),
                      cudaMemcpyDeviceToDevice);

            cudaMemcpy(new_is_deleted, node_mapping->is_deleted,
                      node_mapping->is_deleted_size * sizeof(bool),
                      cudaMemcpyDeviceToDevice);
        }

        // Initialize new elements
        cudaMemset(new_sparse_to_dense + node_mapping->sparse_to_dense_size,
                  0xFF, (new_size - node_mapping->sparse_to_dense_size) * sizeof(int)); // -1 in 2's complement

        cudaMemset(new_is_deleted + node_mapping->is_deleted_size,
                  0x01, (new_size - node_mapping->is_deleted_size) * sizeof(bool)); // true

        // Free old arrays
        if (node_mapping->sparse_to_dense) cudaFree(node_mapping->sparse_to_dense);
        if (node_mapping->is_deleted) cudaFree(node_mapping->is_deleted);

        // Update pointers and sizes
        node_mapping->sparse_to_dense = new_sparse_to_dense;
        node_mapping->sparse_to_dense_size = new_size;
        node_mapping->is_deleted = new_is_deleted;
        node_mapping->is_deleted_size = new_size;
    }

    // Allocate flags array for new nodes
    int* d_new_node_flags = nullptr;
    cudaMalloc(&d_new_node_flags, (max_node_id + 1) * sizeof(int));
    cudaMemset(d_new_node_flags, 0, (max_node_id + 1) * sizeof(int));

    // Mark nodes as not deleted and identify new nodes
    int* sparse_to_dense_ptr = node_mapping->sparse_to_dense;
    bool* is_deleted_ptr = node_mapping->is_deleted;
    int* sources_ptr = edge_data->sources;
    int* targets_ptr = edge_data->targets;

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(start_idx),
        thrust::make_counting_iterator<size_t>(end_idx),
        [sparse_to_dense_ptr, is_deleted_ptr, sources_ptr, targets_ptr, d_new_node_flags]
        DEVICE (const size_t idx) {
            const int source = sources_ptr[idx];
            const int target = targets_ptr[idx];

            if (source >= 0) {
                is_deleted_ptr[source] = false;
                if (sparse_to_dense_ptr[source] == -1) {
                    d_new_node_flags[source] = 1;
                }
            }

            if (target >= 0) {
                is_deleted_ptr[target] = false;
                if (sparse_to_dense_ptr[target] == -1) {
                    d_new_node_flags[target] = 1;
                }
            }
        }
    );

    // Calculate positions for new nodes
    int* d_new_node_positions = nullptr;
    cudaMalloc(&d_new_node_positions, (max_node_id + 1) * sizeof(int));

    thrust::device_ptr<int> d_flags(d_new_node_flags);
    thrust::device_ptr<int> d_positions(d_new_node_positions);

    // Get total count of new nodes using reduce
    int new_nodes_count = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_flags,
        d_flags + (max_node_id + 1)
    );

    // Calculate prefix sum for positions
    thrust::exclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_flags,
        d_flags + (max_node_id + 1),
        d_positions
    );

    // Resize dense_to_sparse array
    size_t old_size = node_mapping->dense_to_sparse_size;
    size_t new_size = old_size + new_nodes_count;

    if (new_nodes_count > 0) {
        int* new_dense_to_sparse = nullptr;
        cudaMalloc(&new_dense_to_sparse, new_size * sizeof(int));

        // Copy existing data
        if (old_size > 0) {
            cudaMemcpy(new_dense_to_sparse,
                      node_mapping->dense_to_sparse,
                      old_size * sizeof(int),
                      cudaMemcpyDeviceToDevice);
        }

        // Free old array
        if (node_mapping->dense_to_sparse) cudaFree(node_mapping->dense_to_sparse);

        // Update pointer and size
        node_mapping->dense_to_sparse = new_dense_to_sparse;
        node_mapping->dense_to_sparse_size = new_size;
    }

    // Assign dense indices in parallel
    int* dense_to_sparse_ptr = node_mapping->dense_to_sparse;

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(max_node_id + 1),
        [sparse_to_dense_ptr, dense_to_sparse_ptr, d_new_node_flags, d_new_node_positions, old_size]
        DEVICE (const size_t idx) {
            if (d_new_node_flags[idx]) {
                const int new_dense_idx = static_cast<int>(old_size) + d_new_node_positions[idx];
                sparse_to_dense_ptr[idx] = new_dense_idx;
                dense_to_sparse_ptr[new_dense_idx] = static_cast<int>(idx);
            }
        }
    );

    // Free temporary device memory
    cudaFree(d_new_node_flags);
    cudaFree(d_new_node_positions);
}

DEVICE int node_mapping::to_dense_device(const NodeMapping *node_mapping, const int sparse_id) {
    if (sparse_id < 0 || sparse_id >= node_mapping->sparse_to_dense_size) {
        return -1;
    }
    return node_mapping->sparse_to_dense[sparse_id];
}

DEVICE int node_mapping::to_dense_from_ptr_device(const int *sparse_to_dense, const int sparse_id, const size_t size) {
    return (sparse_id >= 0 && sparse_id < size) ? sparse_to_dense[sparse_id] : -1;
}

DEVICE void node_mapping::mark_node_deleted_from_ptr(bool *is_deleted, const int sparse_id, const int size) {
    if (sparse_id >= 0 && sparse_id < size) {
        is_deleted[sparse_id] = true;
    }
}

DEVICE bool node_mapping::has_node(const NodeMapping *node_mapping, const int sparse_id) {
    return sparse_id >= 0 &&
           sparse_id < node_mapping->sparse_to_dense_size &&
           node_mapping->sparse_to_dense[sparse_id] != -1;
}

HOST NodeMapping* node_mapping::to_device_ptr(const NodeMapping* node_mapping) {
    // Create a new NodeMapping object on the device
    NodeMapping* device_node_mapping;
    cudaMalloc(&device_node_mapping, sizeof(NodeMapping));

    // If already using GPU, just copy the struct with its pointers
    if (node_mapping->use_gpu) {
        cudaMemcpy(device_node_mapping, node_mapping, sizeof(NodeMapping), cudaMemcpyHostToDevice);
    } else {
        // Create a temporary copy to modify for device pointers
        NodeMapping temp_node_mapping = *node_mapping;

        // Copy each array to device if it exists
        if (node_mapping->sparse_to_dense) {
            int* d_sparse_to_dense;
            cudaMalloc(&d_sparse_to_dense, node_mapping->sparse_to_dense_size * sizeof(int));
            cudaMemcpy(d_sparse_to_dense, node_mapping->sparse_to_dense, node_mapping->sparse_to_dense_size * sizeof(int), cudaMemcpyHostToDevice);
            temp_node_mapping.sparse_to_dense = d_sparse_to_dense;
        }

        if (node_mapping->dense_to_sparse) {
            int* d_dense_to_sparse;
            cudaMalloc(&d_dense_to_sparse, node_mapping->dense_to_sparse_size * sizeof(int));
            cudaMemcpy(d_dense_to_sparse, node_mapping->dense_to_sparse, node_mapping->dense_to_sparse_size * sizeof(int), cudaMemcpyHostToDevice);
            temp_node_mapping.dense_to_sparse = d_dense_to_sparse;
        }

        if (node_mapping->is_deleted) {
            bool* d_is_deleted;
            cudaMalloc(&d_is_deleted, node_mapping->is_deleted_size * sizeof(bool));
            cudaMemcpy(d_is_deleted, node_mapping->is_deleted, node_mapping->is_deleted_size * sizeof(bool), cudaMemcpyHostToDevice);
            temp_node_mapping.is_deleted = d_is_deleted;
        }

        // Make sure use_gpu is set to true
        temp_node_mapping.use_gpu = true;

        // Copy the updated struct to device
        cudaMemcpy(device_node_mapping, &temp_node_mapping, sizeof(NodeMapping), cudaMemcpyHostToDevice);
    }

    return device_node_mapping;
}
