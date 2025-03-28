#include "node_mapping.cuh"

#include "../common/memory.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#endif

HOST DEVICE size_t hash_function(const int key, const size_t capacity) {
    auto k = static_cast<uint32_t>(key);

    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;

    return static_cast<size_t>(k) & (capacity - 1);
}

#ifdef HAS_CUDA
__global__ void add_nodes_kernel(int* node_index, const int capacity, const int* node_ids, const size_t num_nodes, size_t* size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    const int key = node_ids[idx];
    if (key < 0) {
        return;
    }

    uint32_t hash = hash_function(key, capacity);
    const size_t start = hash;

    while (true) {
        const int old_value = atomicCAS(&node_index[hash], -1, key);

        if (old_value == -1) {
            // We successfully inserted the key
            atomicAdd(reinterpret_cast<unsigned long long *>(size), 1ULL); // Atomically increment the size counter
            return;
        }

        if (old_value == key) {
            // Key already exists, no need to insert
            return;
        }

        // Move to next slot (linear probing)
        hash = (hash + 1) % capacity;

        if (hash == start) {
            // Table is full (came back to the start)
            printf("Error: Hash table is full for key %d!\n", key);
            return;
        }
    }
}
#endif

HOST void add_nodes_host(int* node_index, const int capacity, const int* node_ids, const size_t num_nodes, size_t* size) {
    for (size_t idx = 0; idx < num_nodes; idx++) {
        const int key = node_ids[idx];
        if (key < 0) {
            continue;
        }

        uint32_t hash = hash_function(key, capacity);
        const size_t start = hash;

        while (true) {
            // Check if slot is empty or already contains our key
            if (node_index[hash] == -1) {
                // Empty slot, insert the key
                node_index[hash] = key;
                (*size)++; // Increment the size counter
                break;
            }

            if (node_index[hash] == key) {
                // Key already exists, no need to insert
                break;
            }

            // Move to next slot (linear probing)
            hash = (hash + 1) % capacity;

            if (hash == start) {
                // Table is full (came back to the start)
                std::cerr << "Error: Hash table is full for key " << key << "!" << std::endl;
                return;
            }
        }
    }
}

HOST DEVICE bool check_if_has_node(const int* node_index, const int capacity, const int node_id) {
    if (node_id < 0) {
        return false;
    }

    uint32_t hash = hash_function(node_id, capacity);
    const size_t start = hash;

    while (true) {
        // Check if the slot contains our key
        if (node_index[hash] == node_id) {
            return true;
        }

        // If we hit an empty slot, the key doesn't exist
        if (node_index[hash] == -1) {
            return false;
        }

        // Move to next slot (linear probing)
        hash = (hash + 1) % capacity;

        // If we've checked the entire table, the key doesn't exist
        if (hash == start) {
            return false;
        }
    }
}

#ifdef HAS_CUDA
__global__ void get_index_kernel(int* result, const int* node_index, const int capacity, const int node_id) {
    // Only one thread needs to execute this
    if (node_id < 0) {
        *result = -1;
        return;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t hash = hash_function(node_id, capacity);
        const size_t start = hash;

        while (true) {
            // Check if the slot contains our key
            if (node_index[hash] == node_id) {
                *result = static_cast<int>(hash);  // Return the index where the node is found
                return;
            }

            // If we hit an empty slot, the key doesn't exist
            if (node_index[hash] == -1) {
                *result = -1;  // Indicate not found
                return;
            }

            // Move to next slot (linear probing)
            hash = (hash + 1) % capacity;

            // If we've checked the entire table, the key doesn't exist
            if (hash == start) {
                *result = -1;  // Indicate not found
                return;
            }
        }
    }
}
#endif

HOST DEVICE int get_index(const int* node_index, const int capacity, const int node_id) {
    if (node_id < 0) {
        return -1;
    }

    uint32_t hash = hash_function(node_id, capacity);
    const size_t start = hash;

    while (true) {
        // Check if the slot contains our key
        if (node_index[hash] == node_id) {
            return static_cast<int>(hash);  // Return the index where the node is found
        }

        // If we hit an empty slot, the key doesn't exist
        if (node_index[hash] == -1) {
            return -1;  // Indicate not found
        }

        // Move to next slot (linear probing)
        hash = (hash + 1) % capacity;

        // If we've checked the entire table, the key doesn't exist
        if (hash == start) {
            return -1;  // Indicate not found
        }
    }
}

HOST int node_mapping::to_dense(const NodeMappingStore *node_mapping, const int sparse_id) {
    if (!node_mapping || !node_mapping->node_index) {
        return -1;
    }

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // For GPU implementation, find the dense ID directly
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));
        CUDA_CHECK_AND_CLEAR(cudaMemset(d_result, -1, sizeof(int))); // Initialize with -1

        // Launch kernel to find the key's index
        get_index_kernel<<<1, 1>>>(d_result, node_mapping->node_index, node_mapping->capacity, sparse_id);
        CUDA_KERNEL_CHECK("After get_index_kernel execution");

        // Get the result
        int dense_id;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&dense_id, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));

        return dense_id;
    }
    else
    #endif
    {
        // For CPU implementation, directly get the index
        return get_index(node_mapping->node_index, node_mapping->capacity, sparse_id);
    }
}

HOST DEVICE size_t node_mapping::size(const NodeMappingStore *node_mapping) {
    return node_mapping->node_size;
}

HOST size_t node_mapping::active_size(const NodeMappingStore *node_mapping) {
    if (!node_mapping || !node_mapping->node_index || node_mapping->capacity == 0) {
        return 0;
    }

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // For GPU, we need a CUDA kernel to count active nodes
        size_t* d_count;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_count, sizeof(size_t)));
        CUDA_CHECK_AND_CLEAR(cudaMemset(d_count, 0, sizeof(size_t)));

        // Extract fields to avoid referencing struct fields in device code
        int capacity = node_mapping->capacity;
        int* node_index = node_mapping->node_index;
        bool* is_deleted = node_mapping->is_deleted;

        // Define and launch kernel to count active nodes
        auto count_active_nodes = [node_index, is_deleted, capacity, d_count] __device__ (const int idx) {
            if (idx < capacity && node_index[idx] != -1) {
                if (!is_deleted[idx]) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(d_count), 1ULL);
                }
            }
        };

        thrust::for_each(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(capacity),
            count_active_nodes
        );
        CUDA_KERNEL_CHECK("After thrust for_each in active_size");

        // Copy result back to host
        size_t host_count;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&host_count, d_count, sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_count));

        return host_count;
    }
    else
    #endif
    {
        // For CPU, directly iterate through the hash table
        size_t count = 0;
        for (int i = 0; i < node_mapping->capacity; i++) {
            if (node_mapping->node_index[i] != -1) {
                if (!node_mapping->is_deleted[i]) {
                    count++;
                }
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
        // Extract fields to avoid referencing struct fields in device code
        const int capacity = node_mapping->capacity;
        int* node_index = node_mapping->node_index;
        bool* is_deleted = node_mapping->is_deleted;
        int* result_data = result.data;

        // Use atomics to collect active nodes
        int* d_index;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_index, sizeof(int)));
        CUDA_CHECK_AND_CLEAR(cudaMemset(d_index, 0, sizeof(int)));

        auto collect_active = [node_index, is_deleted, capacity, d_index, result_data] __device__ (const int idx) {
            if (idx < capacity && node_index[idx] != -1 && !is_deleted[idx]) {
                int dest = atomicAdd(d_index, 1);
                if (dest < capacity) {
                    result_data[dest] = node_index[idx];
                }
            }
        };

        thrust::for_each(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(capacity),
            collect_active
        );
        CUDA_KERNEL_CHECK("After thrust for_each in get_active_node_ids");

        CUDA_CHECK_AND_CLEAR(cudaFree(d_index));
    }
    else
    #endif
    {
        // For CPU, directly iterate through the hash table
        size_t index = 0;
        for (int i = 0; i < node_mapping->capacity; i++) {
            if (node_mapping->node_index[i] != -1 && !node_mapping->is_deleted[i]) {
                result.data[index++] = node_mapping->node_index[i];
                if (index >= active_count) {
                    break;
                }
            }
        }
    }

    return result;
}



HOST void node_mapping::clear(const NodeMappingStore *node_mapping) {
    fill_memory(node_mapping->node_index, node_mapping->capacity, -1, node_mapping->use_gpu);
    fill_memory(node_mapping->is_deleted, node_mapping->capacity, false, node_mapping->use_gpu);
    node_mapping->node_size = 0;
}

HOST void node_mapping::mark_node_deleted(const NodeMappingStore *node_mapping, const int sparse_id) {
    if (!node_mapping || !node_mapping->node_index || sparse_id < 0) {
        return;
    }

    // Find the position of this node in the hash table
    int hash_idx = get_index(node_mapping->node_index, node_mapping->capacity, sparse_id);

    // If the node exists in the hash table, mark its position as deleted
    if (hash_idx != -1) {
        node_mapping->is_deleted[hash_idx] = true;
    }
}

HOST DataBlock<int> node_mapping::get_all_sparse_ids(const NodeMappingStore *node_mapping) {
    if (!node_mapping || !node_mapping->node_index) {
        return DataBlock<int>{0, false};
    }

    // Get the count of valid entries (not -1)
    const size_t valid_count = node_mapping->node_size;

    // Create result DataBlock
    DataBlock<int> result(valid_count, node_mapping->use_gpu);

    // If no valid entries, return empty block
    if (valid_count == 0) {
        return result;
    }

    #ifdef HAS_CUDA
    if (node_mapping->use_gpu) {
        // Extract fields to avoid referencing struct fields in device code
        const int capacity = node_mapping->capacity;
        int* node_index = node_mapping->node_index;

        // Use thrust to copy non-empty entries
        const thrust::device_ptr<int> d_node_index(node_index);
        const thrust::device_ptr<int> d_result(result.data);

        // Copy all values that are not -1
        thrust::copy_if(
            thrust::device,
            d_node_index,
            d_node_index + capacity,
            d_result,
            [] __device__ (const int node_id) {
                return node_id != -1;
            }
        );
        CUDA_KERNEL_CHECK("After thrust copy_if in get_all_sparse_ids");
    }
    else
    #endif
    {
        // For CPU, iterate through the hash table and collect non-empty entries
        size_t index = 0;
        for (int i = 0; i < node_mapping->capacity; i++) {
            if (node_mapping->node_index[i] != -1) {
                result.data[index++] = node_mapping->node_index[i];

                // Safety check to avoid buffer overflow
                if (index >= valid_count) {
                    break;
                }
            }
        }
    }

    return result;
}

HOST DEVICE bool node_mapping::has_node(const NodeMappingStore *node_mapping, const int sparse_id) {
    if (!node_mapping || !node_mapping->node_index || sparse_id < 0) {
        return false;
    }

    return check_if_has_node(node_mapping->node_index, node_mapping->capacity, sparse_id);
}

HOST void node_mapping::update_std(
    NodeMappingStore *node_mapping,
    const EdgeDataStore *edge_data,
    const size_t start_idx,
    const size_t end_idx) {

    // First, gather all unique node IDs from the edges
    std::vector<int> node_ids;
    node_ids.reserve((end_idx - start_idx) * 2); // Reserve space for worst case

    for (size_t i = start_idx; i < end_idx; i++) {
        const int source = edge_data->sources[i];
        const int target = edge_data->targets[i];

        if (source >= 0) node_ids.push_back(source);
        if (target >= 0) node_ids.push_back(target);
    }

    // Add nodes to the hash table
    if (!node_ids.empty()) {
        add_nodes_host(node_mapping->node_index, node_mapping->capacity,
                    node_ids.data(), node_ids.size(), &node_mapping->node_size);
    }

    for (const int node : node_ids) {
        if (node >= 0) {
            int hash_idx = get_index(node_mapping->node_index, node_mapping->capacity, node);
            if (hash_idx != -1) {
                node_mapping->is_deleted[hash_idx] = false;
            }
        }
    }
}

#ifdef HAS_CUDA

HOST void node_mapping::update_cuda(NodeMappingStore *node_mapping, const EdgeDataStore *edge_data, const size_t start_idx, const size_t end_idx) {
    // First, gather node IDs from edge data
    const size_t num_edges = end_idx - start_idx;
    int* d_node_ids;
    size_t* d_node_count;

    // Allocate memory for node IDs (worst case: 2 nodes per edge)
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_node_ids, num_edges * 2 * sizeof(int)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_node_count, sizeof(size_t)));
    CUDA_CHECK_AND_CLEAR(cudaMemset(d_node_count, 0, sizeof(size_t)));

    // Extract fields for device code
    int* sources = edge_data->sources + start_idx;
    int* targets = edge_data->targets + start_idx;

    // Gather nodes from edges
    auto gather_nodes = [sources, targets, d_node_ids, d_node_count, num_edges] __device__ (const int idx) {
        if (idx < num_edges) {
            const int source = sources[idx];
            const int target = targets[idx];

            if (source >= 0) {
                const size_t pos = atomicAdd(reinterpret_cast<unsigned long long *>(d_node_count), 1);
                d_node_ids[pos] = source;
            }

            if (target >= 0) {
                const size_t pos = atomicAdd(reinterpret_cast<unsigned long long *>(d_node_count), 1);
                d_node_ids[pos] = target;
            }
        }
    };


    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(static_cast<int>(num_edges)),
        gather_nodes
    );
    CUDA_KERNEL_CHECK("After thrust for_each gather_nodes in update_cuda");

    // Get total node count
    size_t node_count;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&node_count, d_node_count, sizeof(size_t), cudaMemcpyDeviceToHost));

    // Add nodes to the hash table
    if (node_count > 0) {
        constexpr int block_size = 256;
        int num_blocks = static_cast<int>(num_edges + block_size - 1) / block_size;

        // Call the device function to add nodes
        add_nodes_kernel<<<num_blocks, block_size>>>(
            node_mapping->node_index,
            node_mapping->capacity,
            d_node_ids, node_count,
            &node_mapping->node_size);
        CUDA_KERNEL_CHECK("After add_nodes_kernel in update_cuda");

        auto node_index = node_mapping->node_index;
        auto is_deleted = node_mapping->is_deleted;
        auto capacity = node_mapping->capacity;

        // Mark nodes as not deleted
        auto mark_not_deleted = [node_index, capacity, is_deleted, d_node_ids] __device__ (const int idx) {
            const int node_id = d_node_ids[idx];
            int hash_idx = get_index(node_index, capacity, node_id);
            if (hash_idx != -1) {
                is_deleted[hash_idx] = false;
            }
        };

        thrust::for_each(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(static_cast<int>(node_count)),
            mark_not_deleted
        );
        CUDA_KERNEL_CHECK("After thrust for_each mark_not_deleted in update_cuda");
    }

    // Free temporary memory
    CUDA_CHECK_AND_CLEAR(cudaFree(d_node_ids));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_node_count));
}

DEVICE int node_mapping::to_dense_device(const NodeMappingStore *node_mapping, const int sparse_id) {
    if (!node_mapping || !node_mapping->node_index || sparse_id < 0) {
        return -1;
    }

    return get_index(node_mapping->node_index, node_mapping->capacity, sparse_id);
}

DEVICE int node_mapping::to_dense_from_ptr_device(const int* node_index, const int sparse_id, const int capacity) {
    if (sparse_id < 0) return -1;
    return get_index(node_index, capacity, sparse_id);
}

DEVICE void node_mapping::mark_node_deleted_from_ptr(bool *is_deleted, const int *node_index, const int sparse_id, const int capacity) {
    if (sparse_id < 0) return;

    // Find the position of this node in the hash table
    int hash_idx = get_index(node_index, capacity, sparse_id);

    // If the node exists in the hash table, mark its position as deleted
    if (hash_idx != -1) {
        is_deleted[hash_idx] = true;
    }
}

HOST NodeMappingStore* node_mapping::to_device_ptr(const NodeMappingStore* node_mapping) {
    // Create a new NodeMapping object on the device
    NodeMappingStore* device_node_mapping;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_node_mapping, sizeof(NodeMappingStore)));

    // Create a temporary copy to modify for device pointers
    NodeMappingStore temp_node_mapping = *node_mapping;
    temp_node_mapping.owns_data = false;

    // If already using GPU, just copy the struct with its pointers
    if (node_mapping->use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_node_mapping, node_mapping, sizeof(NodeMappingStore), cudaMemcpyHostToDevice));
    } else {
        temp_node_mapping.owns_data = true;

        // Copy each array to device if it exists
        if (node_mapping->node_index) {
            int* d_node_index;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_node_index, node_mapping->capacity * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_node_index, node_mapping->node_index, node_mapping->capacity * sizeof(int), cudaMemcpyHostToDevice));
            temp_node_mapping.node_index = d_node_index;
        }

        if (node_mapping->is_deleted) {
            bool* d_is_deleted;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_is_deleted, node_mapping->capacity * sizeof(bool)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_is_deleted, node_mapping->is_deleted, node_mapping->capacity * sizeof(bool), cudaMemcpyHostToDevice));
            temp_node_mapping.is_deleted = d_is_deleted;
        }

        // Make sure use_gpu is set to true
        temp_node_mapping.use_gpu = true;

        // Copy the updated struct to device
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_node_mapping, &temp_node_mapping, sizeof(NodeMappingStore), cudaMemcpyHostToDevice));
    }

    temp_node_mapping.owns_data = false;

    return device_node_mapping;
}

#endif
