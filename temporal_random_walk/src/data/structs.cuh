#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstddef>
#include <cstring>
#include "../common/memory.cuh"
#include "../common/macros.cuh"
#include "walk_set/walk_set.cuh"

struct Edge {
    int u;
    int i;
    int64_t ts;

    const float* features;
    int feature_dim;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1), features(nullptr), feature_dim(0) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts, const float* features, const int feature_dim)
        : u(u), i(i), ts(ts), features(features), feature_dim(feature_dim) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts)
        : u(u), i(i), ts(ts), features(nullptr), feature_dim(0) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
            features = other.features;
            feature_dim = other.feature_dim;
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

struct WalksWithEdgeFeatures {
    WalkSet* walk_set;
    float* walk_edge_features;
    int feature_dim;

    HOST WalksWithEdgeFeatures(WalkSet* walk_set, const int feature_dim)
        : walk_set(walk_set), walk_edge_features(nullptr), feature_dim(feature_dim) {
        if (walk_set == nullptr || feature_dim <= 0) {
            return;
        }

        const size_t walk_edge_features_size = walk_set->edge_ids_size * static_cast<size_t>(feature_dim);
        if (walk_edge_features_size == 0) {
            return;
        }

        allocate_memory(&walk_edge_features, walk_edge_features_size, false);
        std::memset(walk_edge_features, 0, walk_edge_features_size * sizeof(float));
    }

    HOST ~WalksWithEdgeFeatures() {
        clear_memory(&walk_edge_features, false);
    }

    HOST void populate_walk_edge_features(const float* edge_features) const {
        if (walk_set == nullptr || edge_features == nullptr || feature_dim <= 0) {
            return;
        }

        const auto feature_dim_size_t = static_cast<size_t>(feature_dim);
        const size_t walk_edges_count = walk_set->edge_ids_size;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < walk_edges_count; ++i) {
            const auto edge_id = static_cast<size_t>(walk_set->edge_ids[i]);
            float* dst = walk_edge_features + (i * feature_dim_size_t);
            const float* src = edge_features + (edge_id * feature_dim_size_t);
            std::memcpy(dst, src, feature_dim_size_t * sizeof(float));
        }
    }
};

#endif // STRUCTS_H
