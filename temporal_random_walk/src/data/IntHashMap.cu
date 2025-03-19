#include "IntHashMap.cuh"

#include <cstdint>
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

HOST DEVICE void insert_into_hashmap(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
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

HOST DEVICE int get_from_hashmap(const int* keys, const int* values, const int key, const int default_value, const size_t capacity) {
    int result;
    if (get_value_from_hashmap(keys, values, key, result, capacity)) {
        return result;
    }
    return default_value;
}

HOST DEVICE bool has_value_in_hashmap(const int* keys, const int* values, const int key, const size_t capacity) {
    int result;
    return get_value_from_hashmap(keys, values, key, result, capacity);
}

__global__ void insert_kernel(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements) {
    insert_into_hashmap(keys, values, key, value, capacity, count_elements);
}

__global__ void get_kernel(int* result, const int* keys, const int* values, const int key, const int default_value, const size_t capacity) {
    *result = get_from_hashmap(keys, values, key, default_value, capacity);
}

__global__ void has_value_kernel(bool* result, const int* keys, const int* values, const int key, const size_t capacity) {
    *result = has_value_in_hashmap(keys, values, key, capacity);
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

HOST IntHashMap::~IntHashMap() {
    clear_memory(&keys, use_gpu);
    clear_memory(&values, use_gpu);
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
        insert_into_hashmap(keys, values, key, value, capacity, &count_elements);
    }
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

HOST bool IntHashMap::has_value_host(const int key) const {
    if (use_gpu) {
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));

        has_value_kernel<<<1, 1>>>(d_result, keys, values, key, capacity);
        cudaDeviceSynchronize();

        bool result;
        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    } else {
        return has_value_in_hashmap(keys, values, key, capacity);
    }
}

DEVICE void IntHashMap::insert_device(const int key, const int value) {
    insert_into_hashmap(keys, values, key, value, capacity, &count_elements);
}

DEVICE int IntHashMap::get_device(const int key, const int default_value) const {
    return get_from_hashmap(keys, values, key, default_value, capacity);
}

DEVICE bool IntHashMap::has_value_device(const int key) const {
    return has_value_in_hashmap(keys, values, key, capacity);
}

HOST DEVICE size_t IntHashMap::size() const {
    return count_elements;
}
