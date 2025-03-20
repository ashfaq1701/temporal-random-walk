
#ifndef INT_HASH_MAP_H
#define INT_HASH_MAP_H

#include <cstddef>
#include "../common/macros.cuh"

constexpr int EMPTY_KEY = -1;
constexpr int DEFAULT_VALUE = -1;

HOST DEVICE size_t hash_index(int key, size_t capacity);

HOST DEVICE bool get_value_from_hashmap(const int* keys, const int* values, int key, int& value, size_t capacity);

HOST void insert_into_hashmap_host(int* keys, int* values, int key, int value, size_t capacity, size_t* count_elements);

DEVICE void insert_into_hashmap_device(int* keys, int* values, const int key, const int value, const size_t capacity, size_t* count_elements);

HOST bool insert_into_hashmap_if_absent_host(int* keys, int* values, int key, int value, size_t capacity, size_t* count_elements);

DEVICE bool insert_into_hashmap_if_absent_device(int* keys, int* values, int key, int value, size_t capacity, size_t* count_elements);

HOST DEVICE int get_from_hashmap(const int* keys, const int* values, int key, int default_value, size_t capacity);

HOST DEVICE bool has_key_in_hashmap(const int* keys, const int* values, int key, size_t capacity);

__global__ void insert_kernel(int* keys, int* values, int key, int value, size_t capacity);

__global__ void get_kernel(int* result, const int* keys, const int* values, int key, int default_value, size_t capacity);

__global__ void has_key_kernel(bool* result, const int* keys, const int* values, int key, size_t capacity);

__global__ void mark_valid_keys(const int* keys, bool* is_valid, const size_t capacity);

struct IntHashMap {
    int* keys = nullptr;
    int* values = nullptr;
    size_t capacity;
    bool use_gpu;
    size_t count_elements = 0;

public:
    HOST explicit IntHashMap(size_t fixed_capacity, bool on_gpu = false);

    HOST void insert_host(int key, int value);

    HOST bool insert_if_absent_host(int key, int value);

    HOST int get_host(int key, int default_value=DEFAULT_VALUE) const;

    HOST bool has_key_host(int key) const;

    DEVICE void insert_device(int key, int value);

    DEVICE bool insert_if_absent_device(int key, int value);

    DEVICE int get_device(int key, int default_value=DEFAULT_VALUE) const;

    DEVICE bool has_key_device(int key) const;

    HOST DEVICE size_t size() const;

    void clear();

    HOST void get_all_keys_values(int** all_keys, int** all_values, size_t* key_count) const;

    HOST IntHashMap* to_device_ptr() const;
};

#endif // INT_HASH_MAP_H
