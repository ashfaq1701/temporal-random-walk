#ifndef I_EDGEDATA_H
#define I_EDGEDATA_H

#include <cstdint>

#include "../../data/structs.cuh"
#include "../../data/enums.h"
#include "../../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class IEdgeData {
protected:
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64Vector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using DoubleVector = typename SelectVectorType<double, GPUUsage>::type;
    using EdgeVector = typename SelectVectorType<Edge, GPUUsage>::type;

public:
    virtual ~IEdgeData() = default;

    // Core edge data
    IntVector sources{};
    IntVector targets{};
    Int64Vector timestamps{};

    // Timestamp grouping
    SizeVector timestamp_group_offsets{};     // Start of each timestamp group
    Int64Vector unique_timestamps{};          // Corresponding unique timestamps

    DoubleVector forward_cumulative_weights_exponential{};  // For forward temporal sampling
    DoubleVector backward_cumulative_weights_exponential{}; // For backward temporal sampling

    int* sources_ptr = nullptr;
    size_t sources_size = 0;
    int* targets_ptr = nullptr;
    size_t targets_size = 0;
    int64_t* timestamps_ptr = nullptr;
    size_t timestamps_size = 0;

    size_t* timestamp_group_offsets_ptr = nullptr;
    size_t timestamp_group_offsets_size = 0;
    int64_t* unique_timestamps_ptr = nullptr;
    size_t unique_timestamps_size = 0;

    double* forward_cumulative_weights_exponential_ptr = nullptr;
    size_t forward_cumulative_weights_exponential_size = 0;
    double* backward_cumulative_weights_exponential_ptr = nullptr;
    size_t backward_cumulative_weights_exponential_size = 0;

    /**
    * HOST METHODS
    */
    virtual HOST void reserve(size_t size);
    virtual HOST void clear();
    [[nodiscard]] virtual HOST size_t size() const;
    [[nodiscard]] virtual HOST bool empty() const;
    virtual HOST void resize(size_t new_size);

    virtual HOST void add_edges(int* src, int* tgt, int64_t* ts, size_t size);
    virtual HOST void push_back(int src, int tgt, int64_t ts);

    virtual HOST EdgeVector get_edges();

    // Group management
    virtual HOST void update_timestamp_groups() {}  // Call after sorting

    virtual HOST void compute_temporal_weights(double timescale_bound) {}
    virtual HOST void update_temporal_weights(double timescale_bound);

    [[nodiscard]] virtual HOST SizeRange get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] virtual HOST size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] virtual HOST size_t find_group_after_timestamp(int64_t timestamp) const { return 0; }  // For forward walks
    [[nodiscard]] virtual HOST size_t find_group_before_timestamp(int64_t timestamp) const { return 0; } // For backward walks
};

#endif //I_EDGEDATA_H
