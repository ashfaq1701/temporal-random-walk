#include "node_mapping.cuh"

#include "../common/memory.cuh"
#include "../common/cuda_config.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#endif

__global__ void copy_count_elements_kernel(size_t* dst, const IntHashMap* src) {
    *dst = src->count_elements;
}

HOST int node_mapping::to_dense(const NodeMappingStore *node_mapping, const int sparse_id) {
    return node_mapping->node_index->get_host(sparse_id);
}

HOST DEVICE size_t node_mapping::size(const NodeMappingStore *node_mapping) {
    return node_mapping->node_index->size();
}

HOST size_t node_mapping::active_size(const NodeMappingStore *node_mapping) {
    if (node_mapping->is_deleted_size == 0) {
        return 0;
    }

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {

        const auto start_ptr = thrust::device_pointer_cast(node_mapping->is_deleted);
        const auto end_ptr = start_ptr + static_cast<long>(node_mapping->is_deleted_size);

        const size_t result = thrust::count(
            DEVICE_EXECUTION_POLICY,
            start_ptr,
            end_ptr,
            false
        );

        return result;
    }
    else
    #endif
    {
        size_t count = 0;
        for (size_t i = 0; i < node_mapping->is_deleted_size; i++) {
            if (!node_mapping->is_deleted[i]) {
                count++;
            }
        }
        return count;
    }
}

HOST DataBlock<int> node_mapping::get_active_node_ids(const NodeMappingStore *node_mapping) {
    const size_t active_count = active_size(node_mapping);
    DataBlock<int> result(active_count, node_mapping->use_gpu);

    if (active_count == 0) {
        return result;
    }

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // Get all keys directly as device pointers
        int* d_all_keys = nullptr;
        int* d_all_values = nullptr;
        size_t key_count = 0;

        node_mapping->node_index->get_all_keys_values(&d_all_keys, &d_all_values, &key_count);

        if (key_count == 0) {
            cudaFree(d_all_keys);
            cudaFree(d_all_values);
            return result;
        }

        // Create device pointers for thrust
        thrust::device_ptr<int> d_keys(d_all_keys);
        thrust::device_ptr<bool> d_is_deleted(node_mapping->is_deleted);
        thrust::device_ptr<int> d_result(result.data);

        // Use thrust to filter out deleted nodes
        thrust::copy_if(
            DEVICE_EXECUTION_POLICY,
            d_keys,
            d_keys + static_cast<long>(key_count),
            d_result,
            [d_is_deleted, size=node_mapping->is_deleted_size] DEVICE (const int sparse_id) {
                return (sparse_id >= 0 && sparse_id < size && !d_is_deleted[sparse_id]);
            }
        );

        // Cleanup
        cudaFree(d_all_keys);
        cudaFree(d_all_values);
    }
    else
    #endif
    {
        // Get all keys on host
        int* all_keys = nullptr;
        int* all_values = nullptr;
        size_t key_count = 0;

        node_mapping->node_index->get_all_keys_values(&all_keys, &all_values, &key_count);

        if (key_count == 0) {
            delete[] all_keys;
            delete[] all_values;
            return result;
        }

        // CPU version - filter active nodes
        size_t index = 0;
        for (size_t i = 0; i < key_count; i++) {
            int sparse_id = all_keys[i];

            // Check if it's active (not deleted)
            if (sparse_id >= 0 &&
                sparse_id < node_mapping->is_deleted_size &&
                !node_mapping->is_deleted[sparse_id]) {

                result.data[index++] = sparse_id;

                // Safety check to avoid buffer overflow
                if (index >= active_count) {
                    break;
                }
            }
        }

        // Clean up
        delete[] all_keys;
        delete[] all_values;
    }

    return result;
}


HOST void node_mapping::clear(NodeMappingStore *node_mapping) {
    node_mapping->node_index->clear();
    clear_memory(&node_mapping->is_deleted, node_mapping->use_gpu);
    node_mapping->is_deleted_size = 0;
}

HOST void node_mapping::reserve(NodeMappingStore *node_mapping, size_t size) {
    if (size > node_mapping->is_deleted_size) {
        resize_memory(&node_mapping->is_deleted, node_mapping->is_deleted_size, size, node_mapping->use_gpu);
        node_mapping->is_deleted_size = size;
    }
}

HOST void node_mapping::mark_node_deleted(const NodeMappingStore *node_mapping, const int sparse_id) {
    if (sparse_id < 0 || sparse_id >= node_mapping->is_deleted_size) {
        return;
    }

    node_mapping->is_deleted[sparse_id] = true;
}

HOST bool node_mapping:: has_node_host(const NodeMappingStore *node_mapping, const int sparse_id) {
    return node_mapping->node_index->has_key_host(sparse_id);
}

DEVICE bool node_mapping:: has_node_device(const NodeMappingStore *node_mapping, const int sparse_id) {
    return node_mapping->node_index->has_key_device(sparse_id);
}

HOST void node_mapping::update_std(
    NodeMappingStore *node_mapping,
    const EdgeDataStore *edge_data,
    const size_t start_idx,
    const size_t end_idx) {

    // First pass: find max node ID for is_deleted array sizing
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({
            max_node_id,
            edge_data->sources[i],
            edge_data->targets[i]
        });
    }

    // Resize is_deleted array if needed
    if (max_node_id >= node_mapping->is_deleted_size) {
        // Allocate new larger array
        const size_t new_size = max_node_id + 1;
        bool* new_is_deleted = nullptr;

        allocate_memory(&new_is_deleted, new_size, node_mapping->use_gpu);

        // Copy existing data
        if (node_mapping->is_deleted_size > 0) {
            memcpy(new_is_deleted, node_mapping->is_deleted,
                   node_mapping->is_deleted_size * sizeof(bool));
        }

        // Initialize new elements to true (deleted)
        for (size_t i = node_mapping->is_deleted_size; i < new_size; i++) {
            new_is_deleted[i] = true;
        }

        // Free old array
        clear_memory(&node_mapping->is_deleted, node_mapping->use_gpu);

        // Update pointer and size
        node_mapping->is_deleted = new_is_deleted;
        node_mapping->is_deleted_size = new_size;
    }

    int next_node_id = 0;

    for (size_t i = start_idx; i < end_idx; i++) {
        const int source = edge_data->sources[i];
        const int target = edge_data->targets[i];

        // Process source node
        if (source >= 0) {
            // Mark as not deleted
            node_mapping->is_deleted[source] = false;
            if (!node_mapping->node_index->has_key_host(source)) {
                node_mapping->node_index->insert_host(source, next_node_id++);
            }
        }

        // Process target node
        if (target >= 0) {
            // Mark as not deleted
            node_mapping->is_deleted[target] = false;
            if (!node_mapping->node_index->has_key_host(target)) {
                node_mapping->node_index->insert_host(target, next_node_id++);
            }
        }
    }
}

#ifdef HAS_CUDA
HOST void node_mapping::update_cuda(NodeMappingStore *node_mapping, const EdgeDataStore *edge_data, const size_t start_idx, const size_t end_idx) {
    // Find maximum node ID
    thrust::device_ptr<int> d_sources(edge_data->sources + start_idx);
    thrust::device_ptr<int> d_targets(edge_data->targets + start_idx);

    const auto max_source = thrust::max_element(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(end_idx - start_idx)
    );

    const auto max_target = thrust::max_element(
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

    const int max_node_id = std::max(max_source_value, max_target_value);

    if (max_node_id < 0) {
        return;
    }

    // Resize is_deleted array if needed
    if (max_node_id >= node_mapping->is_deleted_size) {
        const size_t new_size = max_node_id + 1;
        bool* new_is_deleted = nullptr;

        allocate_memory(&new_is_deleted, new_size, true);

        // Copy existing data
        if (node_mapping->is_deleted_size > 0) {
            cudaMemcpy(new_is_deleted, node_mapping->is_deleted,
                      node_mapping->is_deleted_size * sizeof(bool),
                      cudaMemcpyDeviceToDevice);
        }

        // Initialize new elements to true (deleted)
        fill_memory(
            new_is_deleted + node_mapping->is_deleted_size,
            new_size - node_mapping->is_deleted_size,
            true,
            true);

        // Free old array
        if (node_mapping->is_deleted) cudaFree(node_mapping->is_deleted);

        // Update pointer and size
        node_mapping->is_deleted = new_is_deleted;
        node_mapping->is_deleted_size = new_size;
    }

    int *d_next_node_id;
    cudaMalloc(&d_next_node_id, sizeof(int));
    cudaMemset(d_next_node_id, static_cast<int>(node_mapping->node_index->size()), sizeof(int));

    IntHashMap* d_node_index = node_mapping->node_index->to_device_ptr();
    int* sources = edge_data->sources;
    int* targets = edge_data->targets;
    bool* is_deleted = node_mapping->is_deleted;

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(start_idx),
        thrust::make_counting_iterator<size_t>(end_idx),
        [d_node_index, sources, targets, is_deleted, d_next_node_id] __device__ (size_t i) {
            const int source = sources[i];
            const int target = targets[i];

            // Process Source Node
            if (source >= 0) {
                // Regular update to is_deleted flag
                is_deleted[source] = false;

                bool inserted = false;

                while (!inserted) {
                    if (d_node_index->has_key_device(source)) {
                        // Another thread has already inserted it
                        inserted = true;
                    } else {
                        // Reserve a new ID
                        const int new_id = atomicAdd(d_next_node_id, 1);

                        // Try to insert atomically
                        inserted = d_node_index->insert_if_absent_device(source, new_id);

                        // If insertion failed (another thread won the race), return the ID
                        if (!inserted) {
                            atomicSub(d_next_node_id, 1);
                        }
                    }
                }
            }

            // Process Target Node
            if (target >= 0) {
                // Regular update to is_deleted flag
                is_deleted[target] = false;

                bool inserted = false;

                while (!inserted) {
                    if (d_node_index->has_key_device(target)) {
                        // Another thread has already inserted it
                        inserted = true;
                    } else {
                        // Reserve a new ID
                        const int new_id = atomicAdd(d_next_node_id, 1);

                        // Try to insert atomically
                        inserted = d_node_index->insert_if_absent_device(target, new_id);

                        // If insertion failed (another thread won the race), return the ID
                        if (!inserted) {
                            atomicSub(d_next_node_id, 1);
                        }
                    }
                }
            }
        }
    );

    size_t* d_count;
    cudaMalloc(&d_count, sizeof(size_t));
    copy_count_elements_kernel<<<1, 1>>>(d_count, d_node_index);
    cudaMemcpy(&(node_mapping->node_index->count_elements), d_count, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
}

DEVICE int node_mapping::to_dense_device(const NodeMappingStore *node_mapping, const int sparse_id) {
    return node_mapping->node_index->get_device(sparse_id);
}

DEVICE int node_mapping::to_dense_from_ptr_device(const int *sparse_to_dense, const int sparse_id, const size_t size) {
    return (sparse_id >= 0 && sparse_id < size) ? sparse_to_dense[sparse_id] : -1;
}

DEVICE void node_mapping::mark_node_deleted_from_ptr(bool *is_deleted, const int sparse_id, const int size) {
    if (sparse_id >= 0 && sparse_id < size) {
        is_deleted[sparse_id] = true;
    }
}

HOST NodeMappingStore* node_mapping::to_device_ptr(const NodeMappingStore* node_mapping) {
    // Create a new NodeMapping object on the device
    NodeMappingStore* device_node_mapping;
    cudaMalloc(&device_node_mapping, sizeof(NodeMappingStore));

    // If already using GPU, just copy the struct with its pointers
    if (node_mapping->use_gpu) {
        // Create a temporary copy to modify for device pointers
        NodeMappingStore temp_node_mapping = *node_mapping;
        temp_node_mapping.node_index = node_mapping->node_index->to_device_ptr();
        cudaMemcpy(device_node_mapping, &temp_node_mapping, sizeof(NodeMappingStore), cudaMemcpyHostToDevice);
    } else {
        // Create a temporary copy to modify for device pointers
        NodeMappingStore temp_node_mapping = *node_mapping;

        // Copy each array to device if it exists
        temp_node_mapping.node_index = node_mapping->node_index->to_device_ptr();

        if (node_mapping->is_deleted) {
            bool* d_is_deleted;
            cudaMalloc(&d_is_deleted, node_mapping->is_deleted_size * sizeof(bool));
            cudaMemcpy(d_is_deleted, node_mapping->is_deleted, node_mapping->is_deleted_size * sizeof(bool), cudaMemcpyHostToDevice);
            temp_node_mapping.is_deleted = d_is_deleted;
        }

        // Make sure use_gpu is set to true
        temp_node_mapping.use_gpu = true;

        // Copy the updated struct to device
        cudaMemcpy(device_node_mapping, &temp_node_mapping, sizeof(NodeMappingStore), cudaMemcpyHostToDevice);
    }

    return device_node_mapping;
}

#endif
