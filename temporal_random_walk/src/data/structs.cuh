#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstddef>

#include "../common/memory.cuh"
#include "../common/macros.cuh"

struct Edge {
    int u;
    int i;
    int64_t ts;

    const float* weights;
    int weights_size;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1), weights(nullptr), weights_size(0) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts, const float* weights, const int weights_size)
        : u(u), i(i), ts(ts), weights(weights), weights_size(weights_size) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts)
        : u(u), i(i), ts(ts), weights(nullptr), weights_size(0) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
            weights = other.weights;
            weights_size = other.weights_size;
        }
        return *this;
    }
};

struct InternalEdge : Edge {
    int64_t edge_id;

    HOST DEVICE InternalEdge()
        : Edge(), edge_id(-1) {}

    HOST DEVICE InternalEdge(const int u, const int i, const int64_t ts, const int64_t edge_id)
        : Edge(u, i, ts), edge_id(edge_id) {}
};

struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(const size_t f, const size_t t) : from(f), to(t) {}

    HOST DEVICE SizeRange& operator=(const SizeRange& other)
    {
        if (this != &other)
        {
            from = other.from;
            to = other.to;
        }
        return *this;
    }
};

template <typename T>
struct DataBlock {
    T* data = nullptr;
    size_t size;
    bool use_gpu;

    // Constructor allocates memory internally
    HOST DataBlock(const size_t size, const bool use_gpu) : size(size), use_gpu(use_gpu) {
        if (size == 0) {
            data = nullptr;
        }
        #ifdef HAS_CUDA
        else if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&data, size * sizeof(T)));
        }
        #endif
        else {
            data = static_cast<T *>(malloc(sizeof(T) * size));  // CPU allocation
        }
    }

    HOST ~DataBlock() {
        if (data) {
            #ifdef HAS_CUDA
            if (use_gpu) {
                CUDA_CHECK_AND_CLEAR(cudaFree(data));
            }
            else
            #endif
            {
                free(data);
            }
        }
    }
};

template <typename T>
struct MemoryView {
    T* data;
    size_t size;
};

#endif // STRUCTS_H