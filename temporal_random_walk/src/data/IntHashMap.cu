#include "IntHashMap.cuh"

#include <cstdint>
#include <thrust/device_ptr.h>
#include <thrust/detail/copy_if.h>

#include "../common/memory.cuh"

HOST DEVICE size_t hash_index(const int key, const size_t capacity) {
    // MurmurHash3 integer finalizer - fast and high quality
    auto h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & (capacity - 1);
}

HOST DEVICE bool get_value_from_hashmap(const int* keys, const int* values, const int key, int& value, const size_t capacity) {
    if (key == EMPTY_KEY) return false;

    // Calculate initial index
    size_t idx = hash_index(key, capacity);
    const size_t original_idx = idx;

    do {
        if (keys[idx] == key) {
            value = values[idx];
            return true;
        }

        if (keys[idx] == EMPTY_KEY) {
            return false;
        }

        // Move to next slot
        idx = (idx + 1) & (capacity - 1);
    } while (idx != original_idx);

    return false;
}

HOST void insert_into_hashmap_host(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    size_t idx = hash_index(key, capacity);
    const size_t original_idx = idx;

    do {
        if (keys[idx] == EMPTY_KEY || keys[idx] == key) {
            if (keys[idx] == EMPTY_KEY) {
                (*count_elements)++;
            }

            keys[idx] = key;
            values[idx] = value;
            return;
        }

        // Move to next slot
        idx = (idx + 1) & (capacity - 1);
    } while (idx != original_idx);
}

DEVICE void insert_into_hashmap_device(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    size_t idx = hash_index(key, capacity);
    const size_t original_idx = idx;

    do {
        // Atomically check and insert key
        int expected = EMPTY_KEY;
        if (atomicCAS(&keys[idx], expected, key) == EMPTY_KEY) {
            // We successfully inserted the key, now store value
            values[idx] = value;

            // Atomically increment element count
            atomicAdd(reinterpret_cast<unsigned long long *>(count_elements), 1ULL);
            return;
        }

        // If the key already exists, update value
        if (keys[idx] == key) {
            values[idx] = value;
            return;
        }

        // Move to next slot
        idx = (idx + 1) & (capacity - 1);
    } while (idx != original_idx);
}

HOST bool insert_into_hashmap_if_absent_host(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    if (key == EMPTY_KEY) return false; // Cannot insert EMPTY_KEY

    size_t idx = hash_index(key, capacity);
    const size_t original_idx = idx;

    do {
        // Check if key already exists
        if (keys[idx] == key) {
            // Key already exists, abort insertion
            return false;
        }

        // If slot is empty, insert the key-value pair
        if (keys[idx] == EMPTY_KEY) {
            keys[idx] = key;
            values[idx] = value;
            (*count_elements)++;
            return true;
        }

        // Move to next slot (linear probing)
        idx = (idx + 1) & (capacity - 1);
    } while (idx != original_idx);

    // Map is full if we reached here
    return false;
}

DEVICE bool insert_into_hashmap_if_absent_device(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    if (key == EMPTY_KEY) return false; // Cannot insert EMPTY_KEY

    size_t idx = hash_index(key, capacity);
    const size_t original_idx = idx;

    do {
        // Check if key already exists
        if (keys[idx] == key) {
            // Key already exists, abort insertion
            return false;
        }

        // Try to atomically insert the key if the slot is empty
        int expected = EMPTY_KEY;
        if (atomicCAS(&keys[idx], expected, key) == EMPTY_KEY) {
            // We successfully inserted the key, now store value
            values[idx] = value;

            // Atomically increment element count
            atomicAdd(reinterpret_cast<unsigned long long*>(count_elements), 1ULL);
            return true;
        }

        // If another thread just inserted the same key, check again
        if (keys[idx] == key) {
            return false;
        }

        // Move to next slot (linear probing)
        idx = (idx + 1) & (capacity - 1);
    } while (idx != original_idx);

    // Map is full if we reached here
    return false;
}

HOST DEVICE int get_from_hashmap(const int* keys, const int* values, const int key, const int default_value, const size_t capacity) {
    int result;
    if (get_value_from_hashmap(keys, values, key, result, capacity)) {
        return result;
    }
    return default_value;
}

HOST DEVICE bool has_key_in_hashmap(const int* keys, const int* values, const int key, const size_t capacity) {
    int result;
    return get_value_from_hashmap(keys, values, key, result, capacity);
}

__global__ void insert_kernel(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    insert_into_hashmap_device(keys, values, key, value, capacity, count_elements);
}

__global__ void insert_if_absent_kernel(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements, bool* success) {
    *success = insert_into_hashmap_if_absent_device(keys, values, key, value, capacity, count_elements);
}

__global__ void get_kernel(int* result, const int* keys, const int* values, const int key, const int default_value, const size_t capacity) {
    *result = get_from_hashmap(keys, values, key, default_value, capacity);
}

__global__ void has_key_kernel(bool* result, const int* keys, const int* values, const int key, const size_t capacity) {
    *result = has_key_in_hashmap(keys, values, key, capacity);
}

__global__ void mark_valid_keys(const int* keys, bool* is_valid, const size_t capacity) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < capacity) {
        is_valid[idx] = (keys[idx] != EMPTY_KEY);  // Mark as valid if not EMPTY_KEY
    }
}

HOST IntHashMap::IntHashMap(const size_t fixed_capacity, const bool on_gpu)
: use_gpu(on_gpu) {

    // Ensure capacity is a power of 2 for fast modulo
    capacity = 1;
    while (capacity < fixed_capacity) {
        capacity <<= 1;
    }

    // Allocate memory for keys and values
    allocate_memory(&keys, capacity, use_gpu);
    allocate_memory(&values, capacity, use_gpu);

    // Initialize all keys to EMPTY_KEY
    fill_memory(keys, capacity, EMPTY_KEY, use_gpu);
}

HOST void IntHashMap::insert_host(const int key, const int value) {
    if (use_gpu) {
        size_t* d_count;
        cudaMalloc(&d_count, sizeof(size_t));
        cudaMemcpy(d_count, &count_elements, sizeof(size_t), cudaMemcpyHostToDevice);

        insert_kernel<<<1, 1>>>(keys, values, key, value, capacity, d_count);
        cudaDeviceSynchronize();

        // Copy updated count back
        cudaMemcpy(&count_elements, d_count, sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaFree(d_count);
    } else {
        insert_into_hashmap_host(keys, values, key, value, capacity, &count_elements);
    }
}

HOST bool IntHashMap::insert_if_absent_host(const int key, const int value) {
    bool success = false;

    if (use_gpu) {
        size_t* d_count;
        bool* d_success;

        cudaMalloc(&d_count, sizeof(size_t));
        cudaMalloc(&d_success, sizeof(bool));

        cudaMemcpy(d_count, &count_elements, sizeof(size_t), cudaMemcpyHostToDevice);

        // Updated kernel call to include success flag
        insert_if_absent_kernel<<<1, 1>>>(keys, values, key, value, capacity, d_count, d_success);
        cudaDeviceSynchronize();

        // Copy updated count back
        cudaMemcpy(&count_elements, d_count, sizeof(size_t), cudaMemcpyDeviceToHost);
        // Copy success result back
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_count);
        cudaFree(d_success);
    } else {
        success = insert_into_hashmap_if_absent_host(keys, values, key, value, capacity, &count_elements);
    }

    return success;
}

HOST int IntHashMap::get_host(const int key, const int default_value) const {
    if (use_gpu) {
        int* d_result;
        cudaMalloc(&d_result, sizeof(int));

        get_kernel<<<1, 1>>>(d_result, keys, values, key, default_value, capacity);
        cudaDeviceSynchronize();

        int result;
        cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    } else {
        return get_from_hashmap(keys, values, key, default_value, capacity);
    }
}

HOST bool IntHashMap::has_key_host(const int key) const {
    if (use_gpu) {
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));

        has_key_kernel<<<1, 1>>>(d_result, keys, values, key, capacity);
        cudaDeviceSynchronize();

        bool result;
        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    } else {
        return has_key_in_hashmap(keys, values, key, capacity);
    }
}

DEVICE void IntHashMap::insert_device(const int key, const int value) {
    insert_into_hashmap_device(keys, values, key, value, capacity, &count_elements);
}

DEVICE bool IntHashMap::insert_if_absent_device(const int key, const int value) {
    return insert_into_hashmap_if_absent_device(keys, values, key, value, capacity, &count_elements);
}

DEVICE int IntHashMap::get_device(const int key, const int default_value) const {
    return get_from_hashmap(keys, values, key, default_value, capacity);
}

DEVICE bool IntHashMap::has_key_device(const int key) const {
    return has_key_in_hashmap(keys, values, key, capacity);
}

HOST DEVICE size_t IntHashMap::size() const {
    return count_elements;
}

void IntHashMap::clear() {
    // Allocate memory for keys and values
    allocate_memory(&keys, capacity, use_gpu);
    allocate_memory(&values, capacity, use_gpu);

    // Initialize all keys to EMPTY_KEY
    fill_memory(keys, capacity, EMPTY_KEY, use_gpu);

    count_elements = 0;
}

HOST void IntHashMap::get_all_keys_values(int** all_keys, int** all_values, size_t* key_count) const {
    if (!keys || !values) return;

    *key_count = size();  // Use size() to determine valid entry count

    if (use_gpu) {
        // Allocate device memory for output
        cudaMalloc(all_keys, (*key_count) * sizeof(int));
        cudaMalloc(all_values, (*key_count) * sizeof(int));

        // Allocate temporary arrays on the GPU
        bool* d_valid_flags;
        cudaMalloc(&d_valid_flags, capacity * sizeof(bool));

        // Launch kernel to mark valid keys
        size_t threads_per_block = 256;
        size_t num_blocks = (capacity + threads_per_block - 1) / threads_per_block;
        mark_valid_keys<<<num_blocks, threads_per_block>>>(keys, d_valid_flags, capacity);
        cudaDeviceSynchronize();

        // Use Thrust to filter out invalid entries
        const thrust::device_ptr<int> d_keys_ptr(keys);
        const thrust::device_ptr<int> d_values_ptr(values);
        const thrust::device_ptr<bool> d_valid_ptr(d_valid_flags);
        const thrust::device_ptr<int> d_output_keys(*all_keys);
        const thrust::device_ptr<int> d_output_values(*all_values);

        thrust::copy_if(d_keys_ptr, d_keys_ptr + static_cast<long>(capacity), d_valid_ptr, d_output_keys, thrust::identity<bool>());
        thrust::copy_if(d_values_ptr, d_values_ptr + static_cast<long>(capacity), d_valid_ptr, d_output_values, thrust::identity<bool>());

        cudaFree(d_valid_flags);
    } else {
        // Host implementation (sequential)
        *all_keys = new int[*key_count];
        *all_values = new int[*key_count];

        size_t index = 0;
        for (size_t i = 0; i < capacity; i++) {
            if (keys[i] != EMPTY_KEY) {
                (*all_keys)[index] = keys[i];
                (*all_values)[index] = values[i];
                index++;
            }
        }
    }
}

HOST IntHashMap* IntHashMap::to_device_ptr() const {
    // Create a new IntHashMap object on the device
    IntHashMap* device_hash_map;
    cudaMalloc(&device_hash_map, sizeof(IntHashMap));

    if (use_gpu) {
        cudaMemcpy(device_hash_map, this, sizeof(IntHashMap), cudaMemcpyHostToDevice);
    } else {
        IntHashMap temp_hash_map = *this;

        if (keys) {
            int* d_keys;
            cudaMalloc(&d_keys, capacity * sizeof(int));
            cudaMemcpy(d_keys, keys, capacity * sizeof(int), cudaMemcpyHostToDevice);
            temp_hash_map.keys = d_keys;
        }

        if (values) {
            int* d_values;
            cudaMalloc(&d_values, capacity * sizeof(int));
            cudaMemcpy(d_values, values, capacity * sizeof(int), cudaMemcpyHostToDevice);
            temp_hash_map.values = d_values;
        }

        temp_hash_map.use_gpu = true;

        cudaMemcpy(device_hash_map, &temp_hash_map, sizeof(IntHashMap), cudaMemcpyHostToDevice);
    }

    return device_hash_map;
}
