#ifndef INT_HASH_MAP_H
#define INT_HASH_MAP_H

#include <cstddef>
#include "../common/macros.cuh"

constexpr int EMPTY_KEY = -1;

HOST DEVICE size_t hash_index(int key, size_t capacity);

HOST DEVICE bool get_value_from_hashmap(const int* keys, const int* values, int key, int& value, size_t capacity);

HOST DEVICE void insert_into_hashmap(int* keys, int* values, int key, int value, size_t capacity, size_t* count_elements);

HOST DEVICE int get_from_hashmap(const int* keys, const int* values, int key, int default_value, size_t capacity);

HOST DEVICE bool has_value_in_hashmap(const int* keys, const int* values, int key, size_t capacity);

__global__ void insert_kernel(int* keys, int* values, int key, int value, size_t capacity);

__global__ void get_kernel(int* result, const int* keys, const int* values, int key, int default_value, size_t capacity);

__global__ void has_value_kernel(bool* result, const int* keys, const int* values, int key, size_t capacity);

struct IntHashMap {
    int* keys = nullptr;
    int* values = nullptr;
    size_t capacity;
    bool use_gpu;
    size_t count_elements = 0;

public:
    HOST explicit IntHashMap(size_t fixed_capacity, bool on_gpu = false);

    HOST ~IntHashMap();

    HOST void insert_host(int key, int value);

    HOST int get_host(int key, int default_value) const;

    HOST bool has_value_host(int key) const;

    DEVICE void insert_device(int key, int value);

    DEVICE int get_device(int key, int default_value) const;

    DEVICE bool has_value_device(int key) const;

    size_t size() const;
};

#endif // INT_HASH_MAP_H
