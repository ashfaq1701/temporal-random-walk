#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>
#include "macros.cuh"

template <typename T>
HOST void allocate_memory(T** data_ptr, int size, bool use_gpu) {

}

template <typename T>
HOST void resize_memory(T** data_ptr, size_t size, size_t new_size, bool use_gpu) {

}

template <typename T>
HOST void fill_memory(T* memory, size_t size, T value, bool use_gpu) {

}

#endif // MEMORY_H
