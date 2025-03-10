#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>
#include "macros.cuh"

template <typename T>
HOST T* allocate_memory(int size, bool use_gpu) {
}

template <typename T>
HOST T* resize_memory(T* memory, size_t size, size_t new_size, bool use_gpu) {

}

template <typename T>
HOST void fill_memory(T* memory, size_t size, T value, bool use_gpu) {

}

#endif // MEMORY_H
