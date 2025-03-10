#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstddef>
#include "../common/memory.cuh"
#include "../common/macros.cuh"

struct Edge {
    int u;
    int i;
    int64_t ts;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts) : u(u), i(i), ts(ts) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
        }
        return *this;
    }
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
    T* data;
    size_t size;
};

struct WalkSet {
    size_t num_walks;
    size_t max_len;
    bool use_gpu;

    int* nodes;
    int64_t* timestamps;
    size_t* walk_lens;

    size_t total_len;

    HOST WalkSet(): num_walks(0), max_len(0), use_gpu(false),
        nodes(nullptr), timestamps(nullptr), walk_lens(nullptr), total_len(0) {}

    HOST WalkSet(const size_t num_walks, const size_t max_len, const bool use_gpu):
        num_walks(num_walks), max_len(max_len), use_gpu(use_gpu) {

        total_len = num_walks * max_len;

        allocate_memory(&nodes, total_len, use_gpu);
        allocate_memory(&timestamps, total_len, use_gpu);
        allocate_memory(&walk_lens, num_walks, use_gpu);
    }
};

#endif // STRUCTS_H