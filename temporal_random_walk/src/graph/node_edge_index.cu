#include "node_edge_index.cuh"

#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>

#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include "../common/cuda_sort.cuh"
#endif

#include "../utils/omp_utils.cuh"
#include "../common/nvtx.cuh"
#include "../common/parallel_algorithms.cuh"
#include "../common/cuda_config.cuh"
#include "../data/buffer.cuh"

#ifdef HAS_CUDA
#include "../common/cuda_scan.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#endif

/**
 * Common Functions
 */

HOST void node_edge_index::clear(TemporalGraphData& data) {
    data.node_group_outbound_offsets.shrink_to_fit_empty();
    data.node_group_inbound_offsets.shrink_to_fit_empty();
    data.node_ts_sorted_outbound_indices.shrink_to_fit_empty();
    data.node_ts_sorted_inbound_indices.shrink_to_fit_empty();
    data.count_ts_group_per_node_outbound.shrink_to_fit_empty();
    data.count_ts_group_per_node_inbound.shrink_to_fit_empty();
    data.node_ts_group_outbound_offsets.shrink_to_fit_empty();
    data.node_ts_group_inbound_offsets.shrink_to_fit_empty();
    data.outbound_forward_cumulative_weights_exponential.shrink_to_fit_empty();
    data.outbound_backward_cumulative_weights_exponential.shrink_to_fit_empty();
    data.inbound_backward_cumulative_weights_exponential.shrink_to_fit_empty();
}

HOST SizeRange node_edge_index::get_edge_range(
    const TemporalGraphData& data,
    const int dense_node_id,
    const bool forward) {
    const size_t* offsets;
    size_t offsets_size;

    if (data.is_directed) {
        offsets = forward
            ? data.node_group_outbound_offsets.data()
            : data.node_group_inbound_offsets.data();
        offsets_size = forward
            ? data.node_group_outbound_offsets.size()
            : data.node_group_inbound_offsets.size();
    } else {
        offsets = data.node_group_outbound_offsets.data();
        offsets_size = data.node_group_outbound_offsets.size();
    }

    if (dense_node_id < 0 || dense_node_id >= static_cast<int>(offsets_size) - 1) {
        return SizeRange{0, 0};
    }

    const size_t start = read_one_host_safe(offsets + dense_node_id,     data.use_gpu);
    const size_t end   = read_one_host_safe(offsets + dense_node_id + 1, data.use_gpu);
    return SizeRange{start, end};
}

HOST SizeRange node_edge_index::get_timestamp_group_range(
    const TemporalGraphData& data,
    const int dense_node_id,
    const size_t group_idx,
    const bool forward) {
    const size_t* group_offsets;
    size_t group_offsets_size;
    const size_t* group_indices;
    const size_t* edge_offsets;

    if (data.is_directed && !forward) {
        group_offsets      = data.count_ts_group_per_node_inbound.data();
        group_offsets_size = data.count_ts_group_per_node_inbound.size();
        group_indices      = data.node_ts_group_inbound_offsets.data();
        edge_offsets       = data.node_group_inbound_offsets.data();
    } else {
        group_offsets      = data.count_ts_group_per_node_outbound.data();
        group_offsets_size = data.count_ts_group_per_node_outbound.size();
        group_indices      = data.node_ts_group_outbound_offsets.data();
        edge_offsets       = data.node_group_outbound_offsets.data();
    }

    if (dense_node_id < 0 || dense_node_id >= static_cast<int>(group_offsets_size) - 1) {
        return SizeRange{0, 0};
    }

    const size_t node_group_start = read_one_host_safe(group_offsets + dense_node_id,     data.use_gpu);
    const size_t node_group_end   = read_one_host_safe(group_offsets + dense_node_id + 1, data.use_gpu);

    const size_t num_groups = node_group_end - node_group_start;
    if (group_idx >= num_groups) {
        return SizeRange{0, 0};
    }

    const size_t group_start_idx = node_group_start + group_idx;
    const size_t group_start     = read_one_host_safe(group_indices + group_start_idx, data.use_gpu);

    size_t group_end;
    if (group_idx == num_groups - 1) {
        group_end = read_one_host_safe(edge_offsets + dense_node_id + 1, data.use_gpu);
    } else {
        group_end = read_one_host_safe(group_indices + group_start_idx + 1, data.use_gpu);
    }

    return SizeRange{group_start, group_end};
}

HOST const Buffer<size_t>& node_edge_index::get_timestamp_offset_vector(
    const TemporalGraphData& data,
    const bool forward) {
    // Returns the live count-per-node buffer. Its .data() pointer is only
    // dereferenceable on the side (host / device) matching data.use_gpu;
    // host-side callers must go through read_one_host_safe or copy out.
    if (data.is_directed && !forward) {
        return data.count_ts_group_per_node_inbound;
    }
    return data.count_ts_group_per_node_outbound;
}

HOST size_t node_edge_index::get_timestamp_group_count(
    const TemporalGraphData& data,
    const int dense_node_id,
    const bool forward) {
    const Buffer<size_t>& offsets_buf = get_timestamp_offset_vector(data, forward);
    const size_t* offsets      = offsets_buf.data();
    const size_t  offsets_size = offsets_buf.size();

    if (dense_node_id < 0 || dense_node_id >= static_cast<int>(offsets_size) - 1) {
        return 0;
    }

    const size_t start = read_one_host_safe(offsets + dense_node_id,     data.use_gpu);
    const size_t end   = read_one_host_safe(offsets + dense_node_id + 1, data.use_gpu);
    return end - start;
}

/**
 * Rebuild allocation
 */

HOST void node_edge_index::allocate_node_group_offsets(
    TemporalGraphData& data,
    const size_t node_index_capacity) {
    data.node_group_outbound_offsets.resize(node_index_capacity);
    data.node_group_outbound_offsets.fill(static_cast<size_t>(0));

    if (data.is_directed) {
        data.node_group_inbound_offsets.resize(node_index_capacity);
        data.node_group_inbound_offsets.fill(static_cast<size_t>(0));
    } else {
        data.node_group_inbound_offsets.shrink_to_fit_empty();
    }
}

HOST void node_edge_index::allocate_node_ts_sorted_indices(TemporalGraphData& data) {
    const size_t outbound_offsets_size = data.node_group_outbound_offsets.size();
    const size_t num_outbound_edges = read_one_host_safe(
        data.node_group_outbound_offsets.data() + (outbound_offsets_size - 1),
        data.use_gpu);

    data.node_ts_sorted_outbound_indices.resize(num_outbound_edges);

    if (data.is_directed) {
        const size_t inbound_offsets_size = data.node_group_inbound_offsets.size();
        const size_t num_inbound_edges = read_one_host_safe(
            data.node_group_inbound_offsets.data() + (inbound_offsets_size - 1),
            data.use_gpu);
        data.node_ts_sorted_inbound_indices.resize(num_inbound_edges);
    } else {
        data.node_ts_sorted_inbound_indices.shrink_to_fit_empty();
    }
}

/**
 * Std implementations
 */
HOST void node_edge_index::compute_node_group_offsets_std(TemporalGraphData& data) {
    const size_t num_edges = data.timestamps.size();
    const bool is_directed = data.is_directed;

    auto* outbound_offsets = data.node_group_outbound_offsets.data();
    auto* inbound_offsets  = data.node_group_inbound_offsets.data();
    const auto* sources    = data.sources.data();
    const auto* targets    = data.targets.data();

    const size_t offset_size = data.node_group_outbound_offsets.size();

    // Step 1: Zero out offset arrays
    std::fill_n(outbound_offsets, offset_size, 0);
    if (is_directed) {
        std::fill_n(inbound_offsets, data.node_group_inbound_offsets.size(), 0);
    }

    // Step 2: Count edge occurrences (use atomic to avoid collisions)
    #pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
        const int src_idx = sources[i];
        const int tgt_idx = targets[i];

        #pragma omp atomic
        outbound_offsets[src_idx + 1]++;

        if (is_directed) {
            #pragma omp atomic
            inbound_offsets[tgt_idx + 1]++;
        } else {
            #pragma omp atomic
            outbound_offsets[tgt_idx + 1]++;
        }
    }

    // Step 3: Inclusive scan over offsets[1..]
    parallel_inclusive_scan(outbound_offsets + 1, offset_size - 1);

    if (is_directed) {
        parallel_inclusive_scan(inbound_offsets + 1, data.node_group_inbound_offsets.size() - 1);
    }
}

HOST void node_edge_index::compute_node_ts_sorted_indices_std(
    TemporalGraphData& data,
    const size_t outbound_buffer_size,
    int* outbound_node_ids,
    int* inbound_node_ids) {
    const bool is_directed = data.is_directed;
    const size_t edges_size = data.timestamps.size();

    const int* sources = data.sources.data();
    const int* targets = data.targets.data();
    size_t* outbound_indices = data.node_ts_sorted_outbound_indices.data();

    // === Step 1: Initialize node_ts_sorted_outbound_indices ===
    #pragma omp parallel for
    for (size_t i = 0; i < edges_size; ++i) {
        if (is_directed) {
            outbound_indices[i] = i;
        } else {
            outbound_indices[i * 2]     = i;
            outbound_indices[i * 2 + 1] = i;
        }
    }

    // === Step 2: Generate node keys for sorting ===
    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        const size_t edge_id = outbound_indices[i];
        const bool is_source = is_directed || (i % 2 == 0);
        outbound_node_ids[i] = is_source ? sources[edge_id] : targets[edge_id];
    }

    // === Step 3: Build a permutation array ===
    std::vector<size_t> indices(outbound_buffer_size);
    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        indices[i] = i;
    }

    // === Step 4: Stable sort the permutation by node ID ===
    parallel::stable_sort(
        indices.begin(),
        indices.end(),
        [&outbound_node_ids](const size_t a, const size_t b) {
            return outbound_node_ids[a] < outbound_node_ids[b];
        }
    );

    // === Step 5: Apply permutation ===
    std::vector<size_t> sorted_outbound_indices(outbound_buffer_size);
    std::vector<int> sorted_outbound_node_ids(outbound_buffer_size);

    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        sorted_outbound_indices[i] = outbound_indices[indices[i]];
        sorted_outbound_node_ids[i] = outbound_node_ids[indices[i]];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < outbound_buffer_size; ++i) {
        outbound_indices[i] = sorted_outbound_indices[i];
        outbound_node_ids[i] = sorted_outbound_node_ids[i];
    }

    // === Step 6: Inbound indices for directed graphs ===
    if (is_directed) {
        size_t* inbound_indices = data.node_ts_sorted_inbound_indices.data();

        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_indices[i] = i;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_node_ids[i] = data.targets.data()[i];
        }

        parallel::stable_sort(inbound_indices, inbound_indices + edges_size,
            [inbound_node_ids](size_t a, size_t b) {
                return inbound_node_ids[a] < inbound_node_ids[b];
            }
        );

        std::vector<int> sorted_inbound_node_ids(edges_size);
        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            sorted_inbound_node_ids[i] = inbound_node_ids[inbound_indices[i]];
        }

        #pragma omp parallel for
        for (size_t i = 0; i < edges_size; ++i) {
            inbound_node_ids[i] = sorted_inbound_node_ids[i];
        }
    }
}

HOST void node_edge_index::allocate_and_compute_node_ts_group_counts_and_offsets_std(
    TemporalGraphData& data,
    const size_t node_count,
    const int* outbound_node_ids,
    const int* inbound_node_ids) {
    const bool is_directed = data.is_directed;
    const int64_t* timestamps = data.timestamps.data();

    const size_t* outbound_indices = data.node_ts_sorted_outbound_indices.data();
    const size_t* inbound_indices = data.node_ts_sorted_inbound_indices.data();

    const size_t num_outbound = data.node_ts_sorted_outbound_indices.size();
    const size_t num_inbound = data.node_ts_sorted_inbound_indices.size();

    // === OUTBOUND ===
    {
        std::vector<size_t> flags(num_outbound, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (i == 0) {
                flags[i] = 1;
                continue;
            }
            const int curr_node = outbound_node_ids[i];
            const int prev_node = outbound_node_ids[i - 1];
            const int64_t curr_ts = timestamps[outbound_indices[i]];
            const int64_t prev_ts = timestamps[outbound_indices[i - 1]];
            flags[i] = (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
        }

        size_t num_groups = 0;
        #pragma omp parallel for reduction(+:num_groups)
        for (size_t i = 0; i < num_outbound; ++i) {
            num_groups += flags[i];
        }

        data.node_ts_group_outbound_offsets.resize(num_groups);

        size_t* group_indices_out = data.node_ts_group_outbound_offsets.data();

        std::vector<size_t> flag_scan(num_outbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_outbound);

        // Fused pass (mirrors GPU thrust::for_each): if flags[i] is set,
        // scatter i into group_indices_out AND atomically bump the per-node
        // group count in one loop.
        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (!flags[i]) continue;
            group_indices_out[flag_scan[i]] = i;
            const int node = outbound_node_ids[i];
            if (node >= 0 && node < static_cast<int>(node_count)) {
                #pragma omp atomic
                group_counts[node]++;
            }
        }

        // Exclusive scan into count_ts_group_per_node_outbound — mirrors the
        // GPU's memset(data, 0) + inclusive_scan(group_counts -> data + 1).
        data.count_ts_group_per_node_outbound.resize(node_count + 1);
        parallel_exclusive_scan(
            group_counts.data(),
            data.count_ts_group_per_node_outbound.data(),
            node_count);
    }

    // === INBOUND ===
    if (is_directed) {
        std::vector<size_t> flags(num_inbound, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (i == 0) {
                flags[i] = 1;
                continue;
            }
            const int curr_node = inbound_node_ids[i];
            const int prev_node = inbound_node_ids[i - 1];
            const int64_t curr_ts = timestamps[inbound_indices[i]];
            const int64_t prev_ts = timestamps[inbound_indices[i - 1]];
            flags[i] = (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
        }

        size_t num_groups = 0;
        #pragma omp parallel for reduction(+:num_groups)
        for (size_t i = 0; i < num_inbound; ++i) {
            num_groups += flags[i];
        }

        data.node_ts_group_inbound_offsets.resize(num_groups);

        size_t* group_indices_out = data.node_ts_group_inbound_offsets.data();

        std::vector<size_t> flag_scan(num_inbound + 1, 0);
        parallel_exclusive_scan(flags.data(), flag_scan.data(), num_inbound);

        // Fused pass: see outbound block above for rationale. Mirrors the
        // inbound GPU fused thrust::for_each.
        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (!flags[i]) continue;
            group_indices_out[flag_scan[i]] = i;
            const int node = inbound_node_ids[i];
            if (node >= 0 && node < static_cast<int>(node_count)) {
                #pragma omp atomic
                group_counts[node]++;
            }
        }

        // Exclusive scan into count_ts_group_per_node_inbound — mirrors the
        // GPU's memset(data, 0) + inclusive_scan(group_counts -> data + 1).
        data.count_ts_group_per_node_inbound.resize(node_count + 1);
        parallel_exclusive_scan(
            group_counts.data(),
            data.count_ts_group_per_node_inbound.data(),
            node_count);
    }
}

namespace {

// Host-side per-direction pipeline used by update_temporal_weights_std.
// Runs the full weight stack (group→node map, per-node min/max + time
// scale, raw weights, per-node sums, normalize, per-node inclusive
// cumulative) for a single direction. Outbound calls with ComputeForward
// = true (sampler reads both fwd and bwd cumulative arrays); the inbound
// call in directed graphs uses ComputeForward = false (only bwd needed).
template <bool ComputeForward>
void compute_per_node_direction_weights_std(
    const size_t* ts_group_per_node_offsets, // count_ts_group_per_node_(out|in)bound
    const size_t* node_ts_group_offsets,     // node_ts_group_(out|in)bound_offsets
    const size_t* edge_sorted_indices,       // node_ts_sorted_(out|in)bound_indices
    const size_t* node_to_edge_offsets,      // node_group_(out|in)bound_offsets
    const int64_t* timestamps,
    const size_t groups_size,
    const size_t node_index_capacity,
    const double timescale_bound,
    double* forward_cum_out,                 // unused when !ComputeForward
    double* backward_cum_out)
{
    if (groups_size == 0 || node_index_capacity == 0) return;

    // Group index -> owning node.
    std::vector<size_t> group_to_node(groups_size);
    #pragma omp parallel for
    for (size_t node = 0; node < node_index_capacity; ++node) {
        const size_t start = ts_group_per_node_offsets[node];
        const size_t end   = ts_group_per_node_offsets[node + 1];
        for (size_t pos = start; pos < end; ++pos) {
            group_to_node[pos] = node;
        }
    }

    // Per-node min/max timestamps and time-scale factor.
    std::vector<int64_t> node_min_ts(node_index_capacity);
    std::vector<int64_t> node_max_ts(ComputeForward ? node_index_capacity : 0);
    std::vector<double>  node_time_scale(node_index_capacity);

    #pragma omp parallel for
    for (size_t node = 0; node < node_index_capacity; ++node) {
        const size_t start = ts_group_per_node_offsets[node];
        const size_t end   = ts_group_per_node_offsets[node + 1];
        if (start >= end) {
            node_min_ts[node] = 0;
            if constexpr (ComputeForward) node_max_ts[node] = 0;
            node_time_scale[node] = 1.0;
            continue;
        }
        const int64_t min_ts = timestamps[edge_sorted_indices[node_ts_group_offsets[start]]];
        const int64_t max_ts = timestamps[edge_sorted_indices[node_ts_group_offsets[end - 1]]];
        const auto time_diff = static_cast<double>(max_ts - min_ts);
        const double time_scale = (timescale_bound > 0 && time_diff > 0)
                                      ? timescale_bound / time_diff
                                      : 1.0;
        node_min_ts[node] = min_ts;
        if constexpr (ComputeForward) node_max_ts[node] = max_ts;
        node_time_scale[node] = time_scale;
    }

    // Raw (unnormalized) weights per group position.
    std::vector<double> raw_forward (ComputeForward ? groups_size : 0);
    std::vector<double> raw_backward(groups_size);

    #pragma omp parallel for
    for (size_t pos = 0; pos < groups_size; ++pos) {
        const size_t node       = group_to_node[pos];
        const size_t edge_start = node_ts_group_offsets[pos];
        const size_t edge_end   =
            (pos + 1 < groups_size && group_to_node[pos + 1] == node)
                ? node_ts_group_offsets[pos + 1]
                : node_to_edge_offsets[node + 1];

        const auto group_sz = static_cast<double>(edge_end - edge_start);
        const int64_t group_ts = timestamps[edge_sorted_indices[edge_start]];
        const int64_t min_ts   = node_min_ts[node];
        const double  scale    = node_time_scale[node];

        const double b_scaled = (timescale_bound > 0)
                                    ? static_cast<double>(group_ts - min_ts) * scale
                                    : static_cast<double>(group_ts - min_ts);
        raw_backward[pos] = group_sz * std::exp(b_scaled);

        if constexpr (ComputeForward) {
            const int64_t max_ts = node_max_ts[node];
            const double f_scaled = (timescale_bound > 0)
                                        ? static_cast<double>(max_ts - group_ts) * scale
                                        : static_cast<double>(max_ts - group_ts);
            raw_forward[pos] = group_sz * std::exp(f_scaled);
        }
    }

    // Per-node sums via scattered atomic adds.
    std::vector<double> node_fwd_sum(ComputeForward ? node_index_capacity : 0, 0.0);
    std::vector<double> node_bwd_sum(node_index_capacity, 0.0);

    #pragma omp parallel for
    for (size_t pos = 0; pos < groups_size; ++pos) {
        const size_t node = group_to_node[pos];
        #pragma omp atomic
        node_bwd_sum[node] += raw_backward[pos];
        if constexpr (ComputeForward) {
            #pragma omp atomic
            node_fwd_sum[node] += raw_forward[pos];
        }
    }

    // Normalize raw -> probability mass per group position.
    std::vector<double> norm_forward (ComputeForward ? groups_size : 0);
    std::vector<double> norm_backward(groups_size);

    #pragma omp parallel for
    for (size_t pos = 0; pos < groups_size; ++pos) {
        const size_t node = group_to_node[pos];
        norm_backward[pos] = raw_backward[pos] / node_bwd_sum[node];
        if constexpr (ComputeForward) {
            norm_forward[pos] = raw_forward[pos] / node_fwd_sum[node];
        }
    }

    // Per-node inclusive cumulative (sequential within each node,
    // parallel across nodes).
    #pragma omp parallel for
    for (size_t node = 0; node < node_index_capacity; ++node) {
        const size_t start = ts_group_per_node_offsets[node];
        const size_t end   = ts_group_per_node_offsets[node + 1];
        if (start >= end) continue;

        double b_cum = 0.0;
        if constexpr (ComputeForward) {
            double f_cum = 0.0;
            for (size_t pos = start; pos < end; ++pos) {
                f_cum += norm_forward[pos];
                b_cum += norm_backward[pos];
                forward_cum_out[pos]  = f_cum;
                backward_cum_out[pos] = b_cum;
            }
        } else {
            for (size_t pos = start; pos < end; ++pos) {
                b_cum += norm_backward[pos];
                backward_cum_out[pos] = b_cum;
            }
        }
    }
}

} // namespace

HOST void node_edge_index::update_temporal_weights_std(
    TemporalGraphData& data,
    const double timescale_bound) {
    const size_t node_index_capacity = data.node_group_outbound_offsets.size() - 1;
    const size_t outbound_groups_size = data.node_ts_group_outbound_offsets.size();

    data.outbound_forward_cumulative_weights_exponential.resize(outbound_groups_size);
    data.outbound_backward_cumulative_weights_exponential.resize(outbound_groups_size);

    const bool is_directed = data.node_group_inbound_offsets.size() > 0;

    // Outbound: both fwd and bwd cumulative (sampler uses both directions).
    compute_per_node_direction_weights_std</*ComputeForward=*/true>(
        data.count_ts_group_per_node_outbound.data(),
        data.node_ts_group_outbound_offsets.data(),
        data.node_ts_sorted_outbound_indices.data(),
        data.node_group_outbound_offsets.data(),
        data.timestamps.data(),
        outbound_groups_size,
        node_index_capacity,
        timescale_bound,
        data.outbound_forward_cumulative_weights_exponential.data(),
        data.outbound_backward_cumulative_weights_exponential.data());

    // Inbound: bwd cumulative only (directed graphs only).
    if (is_directed) {
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();
        const size_t inbound_node_capacity = data.node_group_inbound_offsets.size() - 1;
        data.inbound_backward_cumulative_weights_exponential.resize(inbound_groups_size);

        compute_per_node_direction_weights_std</*ComputeForward=*/false>(
            data.count_ts_group_per_node_inbound.data(),
            data.node_ts_group_inbound_offsets.data(),
            data.node_ts_sorted_inbound_indices.data(),
            data.node_group_inbound_offsets.data(),
            data.timestamps.data(),
            inbound_groups_size,
            inbound_node_capacity,
            timescale_bound,
            /*forward_cum_out=*/nullptr,
            data.inbound_backward_cumulative_weights_exponential.data());
    }
}

/**
 * Cuda implementations
 */
#ifdef HAS_CUDA

// Per-direction CSR offsets via thrust::sort + vectorized thrust::lower_bound.
// For sorted node IDs s[0..n), lower_bound(s, k) is the count of samples < k;
// taken over k in [0, num_nodes+1) this is exactly the prefix-summed CSR
// offset row. One sort + one search; no histogram, no RLE, no scan.
// Replaces cub::DeviceHistogram::HistogramEven, whose internal ScaleTransform
// computes `(sample - lower) * num_levels` in int32 and overflows when
// num_levels^2 crosses 2^31 (delicious's 33.8M nodes wrap to ~INT_MIN bin
// index and write ~17 GB before the histogram allocation).
static void fill_csr_offsets_via_lower_bound(
    const int* d_node_ids,
    const size_t num_samples,
    size_t* d_offsets,
    const size_t num_offsets) {
    if (num_samples == 0 || num_offsets == 0) return;

    Buffer<int> sorted(/*use_gpu=*/true);
    sorted.resize(num_samples);
    CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
        sorted.data(), d_node_ids, num_samples * sizeof(int),
        cudaMemcpyDeviceToDevice, /*stream=*/0));

    thrust::sort(DEVICE_EXECUTION_POLICY,
                 sorted.data(), sorted.data() + num_samples);

    thrust::lower_bound(
        DEVICE_EXECUTION_POLICY,
        sorted.data(), sorted.data() + num_samples,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_offsets)),
        d_offsets);
}

HOST void node_edge_index::compute_node_group_offsets_cuda(TemporalGraphData& data) {
    const size_t num_edges = data.timestamps.size();
    const bool is_directed = data.is_directed;

    size_t* outbound_offsets_ptr = data.node_group_outbound_offsets.data();
    size_t* inbound_offsets_ptr  = is_directed ? data.node_group_inbound_offsets.data() : nullptr;
    const int* src_ptr = data.sources.data();
    const int* tgt_ptr = data.targets.data();

    const size_t outbound_size = data.node_group_outbound_offsets.size();
    const size_t inbound_size  = is_directed ? data.node_group_inbound_offsets.size() : 0;

    if (num_edges == 0 || outbound_size <= 1) {
        // offsets already zeroed by allocate_node_group_offsets; nothing to do.
        return;
    }

    if (is_directed) {
        fill_csr_offsets_via_lower_bound(src_ptr, num_edges, outbound_offsets_ptr, outbound_size);
        fill_csr_offsets_via_lower_bound(tgt_ptr, num_edges, inbound_offsets_ptr,  inbound_size);
    } else {
        // Undirected: each edge contributes to both endpoints, so concatenate
        // [sources, targets] into one sample stream and run a single pass.
        Buffer<int> concat(/*use_gpu=*/true);
        concat.resize(num_edges * 2);

        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            concat.data(),             src_ptr, num_edges * sizeof(int),
            cudaMemcpyDeviceToDevice, /*stream=*/0));
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            concat.data() + num_edges, tgt_ptr, num_edges * sizeof(int),
            cudaMemcpyDeviceToDevice, /*stream=*/0));

        fill_csr_offsets_via_lower_bound(
            concat.data(), num_edges * 2, outbound_offsets_ptr, outbound_size);
    }
}

HOST void node_edge_index::compute_node_ts_sorted_indices_cuda(
    TemporalGraphData& data,
    const size_t outbound_buffer_size,
    int* outbound_node_ids,
    int* inbound_node_ids) {
    NvtxRange r("node_index_rebuild");

    const bool is_directed = data.is_directed;
    const size_t edges_size = data.timestamps.size();

    const int* sources = data.sources.data();
    const int* targets = data.targets.data();
    size_t* outbound_indices = data.node_ts_sorted_outbound_indices.data();

    // === Step 1: Initialize node_ts_sorted_outbound_indices ===
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(edges_size),
        [outbound_indices, is_directed] DEVICE (const size_t i) {
            if (is_directed) {
                outbound_indices[i] = i;
            } else {
                outbound_indices[i * 2]     = i;
                outbound_indices[i * 2 + 1] = i;
            }
        }
    );
    CUDA_KERNEL_CHECK("Initialized node_ts_sorted_outbound_indices");

    // === Step 2: Fill outbound_node_ids
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(outbound_buffer_size),
        [outbound_node_ids, outbound_indices, sources, targets, is_directed] DEVICE (const size_t i) {
            const size_t edge_id = outbound_indices[i];
            const bool is_source = is_directed || (i % 2 == 0);
            outbound_node_ids[i] = is_source ? sources[edge_id] : targets[edge_id];
        }
    );
    CUDA_KERNEL_CHECK("Generated outbound_node_ids");

    // === Step 3: Build permutation array
    thrust::device_vector<size_t> indices(outbound_buffer_size);
    thrust::sequence(
        DEVICE_EXECUTION_POLICY,
        indices.begin(),
        indices.end()
    );
    CUDA_KERNEL_CHECK("Generated permutation indices");

    // === Step 4: Sort indices by outbound_node_ids using CUB wrapper
    cub_radix_sort_values_by_keys(
        outbound_node_ids,
        thrust::raw_pointer_cast(indices.data()),
        outbound_buffer_size
    );
    CUDA_KERNEL_CHECK("Sorted indices by node keys");

    // === Step 5: Apply permutation
    thrust::device_vector<size_t> sorted_outbound_indices(outbound_buffer_size);
    thrust::device_vector<int> sorted_outbound_node_ids(outbound_buffer_size);

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(outbound_buffer_size),
        [sorted_outbound_indices = sorted_outbound_indices.data(),
         sorted_outbound_node_ids = sorted_outbound_node_ids.data(),
         outbound_indices,
         outbound_node_ids,
         indices = indices.data()] DEVICE (const size_t i) {
            const auto idx = static_cast<long>(i);
            const size_t sorted_idx = indices[idx];
            sorted_outbound_indices[idx] = outbound_indices[sorted_idx];
            sorted_outbound_node_ids[idx] = outbound_node_ids[sorted_idx];
        }
    );
    CUDA_KERNEL_CHECK("Applied permutation");

    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_outbound_indices.begin(),
        sorted_outbound_indices.end(),
        outbound_indices
    );
    thrust::copy(
        DEVICE_EXECUTION_POLICY,
        sorted_outbound_node_ids.begin(),
        sorted_outbound_node_ids.end(),
        outbound_node_ids
    );
    CUDA_KERNEL_CHECK("Copied sorted outbound data");

    // === Step 6: Inbound for directed
    if (is_directed) {
        size_t* inbound_indices = data.node_ts_sorted_inbound_indices.data();

        thrust::sequence(
            DEVICE_EXECUTION_POLICY,
            inbound_indices,
            inbound_indices + edges_size
        );
        CUDA_KERNEL_CHECK("Initialized node_ts_sorted_inbound_indices");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            inbound_node_ids,
            targets,
            sizeof(int) * edges_size,
            cudaMemcpyDeviceToDevice
        ));

        cub_radix_sort_values_by_keys(
            inbound_node_ids,
            inbound_indices,
            edges_size
        );
        CUDA_KERNEL_CHECK("Sorted node_ts_sorted_inbound_indices");

        thrust::device_vector<int> sorted_inbound_node_ids(edges_size);
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(edges_size),
            [sorted_inbound_node_ids = sorted_inbound_node_ids.data(),
             inbound_node_ids, inbound_indices] DEVICE (const size_t i) {
                const auto idx = static_cast<long>(i);
                sorted_inbound_node_ids[idx] = inbound_node_ids[inbound_indices[idx]];
            }
        );
        thrust::copy(
            DEVICE_EXECUTION_POLICY,
            sorted_inbound_node_ids.begin(),
            sorted_inbound_node_ids.end(),
            inbound_node_ids
        );
        CUDA_KERNEL_CHECK("Copied sorted inbound_node_ids");
    }
}

HOST void node_edge_index::allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
    TemporalGraphData& data,
    const size_t node_count,
    const int* outbound_node_ids,
    const int* inbound_node_ids) {
    const bool is_directed = data.is_directed;
    int64_t* timestamps_ptr = data.timestamps.data();

    // === OUTBOUND ===
    {
        const size_t num_edges = data.node_ts_sorted_outbound_indices.size();
        size_t* indices = data.node_ts_sorted_outbound_indices.data();

        thrust::device_vector<int> flags(num_edges, 0);
        auto flags_ptr = thrust::raw_pointer_cast(flags.data());

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags_ptr,
            [outbound_node_ids, indices, timestamps_ptr] DEVICE(const size_t i) -> int {
                if (i == 0) return 1;
                const int curr_node = outbound_node_ids[i];
                const int prev_node = outbound_node_ids[i - 1];
                const int64_t curr_ts = timestamps_ptr[indices[i]];
                const int64_t prev_ts = timestamps_ptr[indices[i - 1]];
                return (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
            }
        );

        const size_t num_groups = thrust::reduce(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
        );

        data.node_ts_group_outbound_offsets.resize(num_groups);

        size_t* group_indices_out = data.node_ts_group_outbound_offsets.data();

        thrust::device_vector<size_t> flag_scan(num_edges);
        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());
        cub_exclusive_sum(flags_ptr, flag_scan_ptr, num_edges);

        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        // Fused pass: each thread reads flags[i] once and, if set, both
        // scatters i into group_indices_out AND bumps the per-node group
        // count. Previously two independent for_each kernels, both
        // num_edges-sized, both memory-bound on flags[i].
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out,
             outbound_node_ids, group_counts_ptr, node_count] DEVICE(const size_t i) {
                if (!flags_ptr[i]) return;
                group_indices_out[flag_scan_ptr[i]] = i;
                const int node = outbound_node_ids[i];
                if (node >= 0 && node < node_count) {
                    atomicAdd(&group_counts_ptr[node], 1u);
                }
            }
        );

        data.count_ts_group_per_node_outbound.resize(node_count + 1);

        CUDA_CHECK_AND_CLEAR(cudaMemset(
            data.count_ts_group_per_node_outbound.data(), 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            data.count_ts_group_per_node_outbound.data() + 1
        );
    }

    // === INBOUND ===
    if (is_directed) {
        const size_t num_edges = data.node_ts_sorted_inbound_indices.size();
        size_t* indices = data.node_ts_sorted_inbound_indices.data();

        thrust::device_vector<int> flags(num_edges, 0);
        auto flags_ptr = thrust::raw_pointer_cast(flags.data());

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            flags_ptr,
            [inbound_node_ids, indices, timestamps_ptr] DEVICE(size_t i) -> int {
                if (i == 0) return 1;
                const int curr_node = inbound_node_ids[i];
                const int prev_node = inbound_node_ids[i - 1];
                const int64_t curr_ts = timestamps_ptr[indices[i]];
                const int64_t prev_ts = timestamps_ptr[indices[i - 1]];
                return (curr_node != prev_node || curr_ts != prev_ts) ? 1 : 0;
            }
        );

        const size_t num_groups = thrust::reduce(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
        );

        data.node_ts_group_inbound_offsets.resize(num_groups);

        size_t* group_indices_out = data.node_ts_group_inbound_offsets.data();

        thrust::device_vector<size_t> flag_scan(num_edges);
        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());
        cub_exclusive_sum(flags_ptr, flag_scan_ptr, num_edges);

        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        // Fused pass: see outbound block above for rationale.
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out,
             inbound_node_ids, group_counts_ptr, node_count] DEVICE(const size_t i) {
                if (!flags_ptr[i]) return;
                group_indices_out[flag_scan_ptr[i]] = i;
                const int node = inbound_node_ids[i];
                if (node >= 0 && node < node_count) {
                    atomicAdd(&group_counts_ptr[node], 1u);
                }
            }
        );

        data.count_ts_group_per_node_inbound.resize(node_count + 1);

        CUDA_CHECK_AND_CLEAR(cudaMemset(
            data.count_ts_group_per_node_inbound.data(), 0, sizeof(size_t)));

        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            group_counts.begin(),
            group_counts.end(),
            data.count_ts_group_per_node_inbound.data() + 1
        );
    }
}

namespace {

// Per-node fused fwd+bwd weight kernel (one block per node). Used for the
// outbound path, where the sampler needs both forward and backward
// cumulative weights.
//
// Replaces the old 8-kernel thrust pipeline (build_group_to_node, min/max
// per node, raw weights, atomicAdd sums, normalize, 2x inclusive_scan_by_key)
// with a single kernel:
//  - Pass 1 (tiled): compute raw_fwd / raw_bwd into scratch, block-reduce
//    per-tile sums into per-node running sums in shared memory.
//  - Pass 2 (tiled): divide raw by node-sum, BlockScan::InclusiveSum with a
//    running carry across tiles, write cumulative normalized weights.
//
// __syncthreads() between passes is both a barrier AND a memory fence for
// the block's writes to raw_*_scratch. Each block only touches its own
// [out_start, out_end) range, so scratch has no inter-block contention.
template <int BLOCK_SIZE>
__global__ void compute_per_node_weights_fwd_bwd_kernel(
    const size_t* __restrict__ group_offsets,        // count_ts_group_per_node_outbound
    const size_t* __restrict__ group_to_edge_start,  // node_ts_group_outbound_offsets
    const size_t* __restrict__ node_to_edge_offsets, // node_group_outbound_offsets
    const size_t* __restrict__ edge_sorted_indices,  // node_ts_sorted_outbound_indices
    const int64_t* __restrict__ timestamps,
    const double timescale_bound,
    double* __restrict__ raw_fwd_scratch,
    double* __restrict__ raw_bwd_scratch,
    double* __restrict__ cum_fwd_out,
    double* __restrict__ cum_bwd_out) {

    const size_t node = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);

    const size_t out_start = group_offsets[node];
    const size_t out_end   = group_offsets[node + 1];
    if (out_start >= out_end) return;

    __shared__ int64_t s_min_ts;
    __shared__ int64_t s_max_ts;
    __shared__ double  s_scale;
    if (tid == 0) {
        const size_t first_group_start = group_to_edge_start[out_start];
        const size_t last_group_start  = group_to_edge_start[out_end - 1];
        const int64_t min_ts = timestamps[edge_sorted_indices[first_group_start]];
        const int64_t max_ts = timestamps[edge_sorted_indices[last_group_start]];
        const double time_diff = static_cast<double>(max_ts - min_ts);
        s_min_ts = min_ts;
        s_max_ts = max_ts;
        s_scale  = (timescale_bound > 0.0 && time_diff > 0.0)
                       ? (timescale_bound / time_diff)
                       : 1.0;
    }
    __syncthreads();

    const int64_t min_ts = s_min_ts;
    const int64_t max_ts = s_max_ts;
    const double  scale  = s_scale;

    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
    using BlockScan   = cub::BlockScan<double, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ typename BlockScan::TempStorage   scan_storage;

    __shared__ double s_sum_fwd;
    __shared__ double s_sum_bwd;
    __shared__ double s_carry_fwd;
    __shared__ double s_carry_bwd;
    if (tid == 0) {
        s_sum_fwd   = 0.0;
        s_sum_bwd   = 0.0;
        s_carry_fwd = 0.0;
        s_carry_bwd = 0.0;
    }
    __syncthreads();

    const size_t num_groups = out_end - out_start;
    const size_t num_tiles  = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t tile = 0; tile < num_tiles; ++tile) {
        const size_t local_idx = tile * BLOCK_SIZE + tid;
        double raw_f = 0.0;
        double raw_b = 0.0;
        if (local_idx < num_groups) {
            const size_t pos = out_start + local_idx;
            const size_t edge_start = group_to_edge_start[pos];
            const size_t edge_end = (local_idx + 1 < num_groups)
                ? group_to_edge_start[pos + 1]
                : node_to_edge_offsets[node + 1];
            const double group_sz  = static_cast<double>(edge_end - edge_start);
            const int64_t group_ts = timestamps[edge_sorted_indices[edge_start]];
            const double tf = static_cast<double>(max_ts - group_ts) * scale;
            const double tb = static_cast<double>(group_ts - min_ts) * scale;
            raw_f = group_sz * exp(tf);
            raw_b = group_sz * exp(tb);
            raw_fwd_scratch[pos] = raw_f;
            raw_bwd_scratch[pos] = raw_b;
        }

        const double tile_sum_f = BlockReduce(reduce_storage).Sum(raw_f);
        __syncthreads(); // reuse reduce_storage
        const double tile_sum_b = BlockReduce(reduce_storage).Sum(raw_b);
        if (tid == 0) {
            s_sum_fwd += tile_sum_f;
            s_sum_bwd += tile_sum_b;
        }
        __syncthreads();
    }

    const double sum_fwd = s_sum_fwd;
    const double sum_bwd = s_sum_bwd;

    for (size_t tile = 0; tile < num_tiles; ++tile) {
        const size_t local_idx = tile * BLOCK_SIZE + tid;
        double n_f = 0.0;
        double n_b = 0.0;
        const bool in_range = (local_idx < num_groups);
        if (in_range) {
            const size_t pos = out_start + local_idx;
            n_f = raw_fwd_scratch[pos] / sum_fwd;
            n_b = raw_bwd_scratch[pos] / sum_bwd;
        }
        double scanned_f;
        double scanned_b;
        double agg_f;
        double agg_b;
        BlockScan(scan_storage).InclusiveSum(n_f, scanned_f, agg_f);
        __syncthreads();
        BlockScan(scan_storage).InclusiveSum(n_b, scanned_b, agg_b);

        if (in_range) {
            const size_t pos = out_start + local_idx;
            cum_fwd_out[pos] = scanned_f + s_carry_fwd;
            cum_bwd_out[pos] = scanned_b + s_carry_bwd;
        }
        __syncthreads();
        if (tid == 0) {
            s_carry_fwd += agg_f;
            s_carry_bwd += agg_b;
        }
        __syncthreads();
    }
}

// Per-node bwd-only weight kernel. Used for the inbound path in directed
// graphs, where the sampler only needs the backward cumulative weights.
//
// Structural mirror of compute_per_node_weights_fwd_bwd_kernel with the
// forward pipeline stripped out — same tiling, same BlockReduce/BlockScan
// cadence, same carry-across-tiles logic. Keep the two in sync when
// touching the shared pattern.
template <int BLOCK_SIZE>
__global__ void compute_per_node_weights_bwd_only_kernel(
    const size_t* __restrict__ group_offsets,        // count_ts_group_per_node_inbound
    const size_t* __restrict__ group_to_edge_start,  // node_ts_group_inbound_offsets
    const size_t* __restrict__ node_to_edge_offsets, // node_group_inbound_offsets
    const size_t* __restrict__ edge_sorted_indices,  // node_ts_sorted_inbound_indices
    const int64_t* __restrict__ timestamps,
    const double timescale_bound,
    double* __restrict__ raw_bwd_scratch,
    double* __restrict__ cum_bwd_out) {

    const size_t node = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);

    const size_t in_start = group_offsets[node];
    const size_t in_end   = group_offsets[node + 1];
    if (in_start >= in_end) return;

    __shared__ int64_t s_min_ts;
    __shared__ double  s_scale;
    if (tid == 0) {
        const size_t first_group_start = group_to_edge_start[in_start];
        const size_t last_group_start  = group_to_edge_start[in_end - 1];
        const int64_t min_ts = timestamps[edge_sorted_indices[first_group_start]];
        const int64_t max_ts = timestamps[edge_sorted_indices[last_group_start]];
        const double time_diff = static_cast<double>(max_ts - min_ts);
        s_min_ts = min_ts;
        s_scale  = (timescale_bound > 0.0 && time_diff > 0.0)
                       ? (timescale_bound / time_diff)
                       : 1.0;
    }
    __syncthreads();

    const int64_t min_ts = s_min_ts;
    const double  scale  = s_scale;

    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
    using BlockScan   = cub::BlockScan<double, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ typename BlockScan::TempStorage   scan_storage;

    __shared__ double s_sum_bwd;
    __shared__ double s_carry_bwd;
    if (tid == 0) {
        s_sum_bwd   = 0.0;
        s_carry_bwd = 0.0;
    }
    __syncthreads();

    const size_t num_groups = in_end - in_start;
    const size_t num_tiles  = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t tile = 0; tile < num_tiles; ++tile) {
        const size_t local_idx = tile * BLOCK_SIZE + tid;
        double raw_b = 0.0;
        if (local_idx < num_groups) {
            const size_t pos = in_start + local_idx;
            const size_t edge_start = group_to_edge_start[pos];
            const size_t edge_end = (local_idx + 1 < num_groups)
                ? group_to_edge_start[pos + 1]
                : node_to_edge_offsets[node + 1];
            const double group_sz  = static_cast<double>(edge_end - edge_start);
            const int64_t group_ts = timestamps[edge_sorted_indices[edge_start]];
            const double tb = static_cast<double>(group_ts - min_ts) * scale;
            raw_b = group_sz * exp(tb);
            raw_bwd_scratch[pos] = raw_b;
        }
        const double tile_sum_b = BlockReduce(reduce_storage).Sum(raw_b);
        if (tid == 0) s_sum_bwd += tile_sum_b;
        __syncthreads();
    }

    const double sum_bwd = s_sum_bwd;

    for (size_t tile = 0; tile < num_tiles; ++tile) {
        const size_t local_idx = tile * BLOCK_SIZE + tid;
        double n_b = 0.0;
        const bool in_range = (local_idx < num_groups);
        if (in_range) {
            const size_t pos = in_start + local_idx;
            n_b = raw_bwd_scratch[pos] / sum_bwd;
        }
        double scanned_b;
        double agg_b;
        BlockScan(scan_storage).InclusiveSum(n_b, scanned_b, agg_b);

        if (in_range) {
            const size_t pos = in_start + local_idx;
            cum_bwd_out[pos] = scanned_b + s_carry_bwd;
        }
        __syncthreads();
        if (tid == 0) s_carry_bwd += agg_b;
        __syncthreads();
    }
}

}  // namespace

HOST void node_edge_index::update_temporal_weights_cuda(
    TemporalGraphData& data,
    double timescale_bound) {
    const size_t node_index_capacity = data.node_group_outbound_offsets.size() - 1;
    const size_t outbound_groups_size = data.node_ts_group_outbound_offsets.size();

    data.outbound_forward_cumulative_weights_exponential.resize(outbound_groups_size);
    data.outbound_backward_cumulative_weights_exponential.resize(outbound_groups_size);

    const bool is_directed = (data.node_group_inbound_offsets.size() > 0);
    if (is_directed) {
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();
        data.inbound_backward_cumulative_weights_exponential.resize(inbound_groups_size);
    }

    int64_t* timestamps_ptr = data.timestamps.data();
    constexpr int BLOCK_SIZE = 128;

    // === OUTBOUND weights (single fused kernel per rebuild) ===
    if (outbound_groups_size > 0) {
        Buffer<double> raw_fwd_scratch(/*use_gpu=*/true);
        Buffer<double> raw_bwd_scratch(/*use_gpu=*/true);
        raw_fwd_scratch.resize(outbound_groups_size);
        raw_bwd_scratch.resize(outbound_groups_size);

        const size_t* group_offsets_ptr       = data.count_ts_group_per_node_outbound.data();
        const size_t* group_to_edge_start_ptr = data.node_ts_group_outbound_offsets.data();
        const size_t* node_to_edge_offsets_ptr= data.node_group_outbound_offsets.data();
        const size_t* edge_indices_ptr        = data.node_ts_sorted_outbound_indices.data();

        double* cum_fwd_out = data.outbound_forward_cumulative_weights_exponential.data();
        double* cum_bwd_out = data.outbound_backward_cumulative_weights_exponential.data();

        const dim3 grid(static_cast<unsigned int>(node_index_capacity));
        const dim3 block(BLOCK_SIZE);
        compute_per_node_weights_fwd_bwd_kernel<BLOCK_SIZE><<<grid, block>>>(
            group_offsets_ptr,
            group_to_edge_start_ptr,
            node_to_edge_offsets_ptr,
            edge_indices_ptr,
            timestamps_ptr,
            timescale_bound,
            raw_fwd_scratch.data(),
            raw_bwd_scratch.data(),
            cum_fwd_out,
            cum_bwd_out);
        CUDA_KERNEL_CHECK("After outbound weights processing in update_temporal_weights_cuda");
    }

    // === INBOUND weights (directed only) ===
    if (is_directed) {
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();
        if (inbound_groups_size > 0) {
            const size_t inbound_node_capacity = data.node_group_inbound_offsets.size() - 1;

            Buffer<double> raw_bwd_scratch(/*use_gpu=*/true);
            raw_bwd_scratch.resize(inbound_groups_size);

            const size_t* group_offsets_ptr       = data.count_ts_group_per_node_inbound.data();
            const size_t* group_to_edge_start_ptr = data.node_ts_group_inbound_offsets.data();
            const size_t* node_to_edge_offsets_ptr= data.node_group_inbound_offsets.data();
            const size_t* edge_indices_ptr        = data.node_ts_sorted_inbound_indices.data();

            double* cum_bwd_out = data.inbound_backward_cumulative_weights_exponential.data();

            const dim3 grid(static_cast<unsigned int>(inbound_node_capacity));
            const dim3 block(BLOCK_SIZE);
            compute_per_node_weights_bwd_only_kernel<BLOCK_SIZE><<<grid, block>>>(
                group_offsets_ptr,
                group_to_edge_start_ptr,
                node_to_edge_offsets_ptr,
                edge_indices_ptr,
                timestamps_ptr,
                timescale_bound,
                raw_bwd_scratch.data(),
                cum_bwd_out);
            CUDA_KERNEL_CHECK("After inbound weights processing in update_temporal_weights_cuda");
        }
    }
}

#endif

HOST void node_edge_index::rebuild(TemporalGraphData& data) {
    // node_index_capacity matches the old code: pass the active-node bitmap
    // size as the per-node CSR size; allocate_node_group_offsets resizes the
    // offsets buffer to that value (which already includes the +1 sentinel).
    const size_t node_index_capacity = data.active_node_ids.size() + 1;

    {
        NVTX_RANGE_COLORED("Node group offsets", nvtx_colors::index_blue);
        allocate_node_group_offsets(data, node_index_capacity);

        #ifdef HAS_CUDA
        if (data.use_gpu) {
            compute_node_group_offsets_cuda(data);
        } else
        #endif
        {
            compute_node_group_offsets_std(data);
        }
    }

    allocate_node_ts_sorted_indices(data);

    const size_t num_edges = data.timestamps.size();
    const size_t outbound_buffer_size = data.is_directed ? num_edges : num_edges * 2;

    Buffer<int> outbound_node_ids(data.use_gpu);
    outbound_node_ids.resize(outbound_buffer_size);

    Buffer<int> inbound_node_ids(data.use_gpu);
    inbound_node_ids.resize(num_edges);

    {
        NVTX_RANGE_COLORED("Sorted indices", nvtx_colors::index_blue);
        #ifdef HAS_CUDA
        if (data.use_gpu) {
            compute_node_ts_sorted_indices_cuda(
                data,
                outbound_buffer_size,
                outbound_node_ids.data(),
                inbound_node_ids.data()
            );
        } else
        #endif
        {
            compute_node_ts_sorted_indices_std(
                data,
                outbound_buffer_size,
                outbound_node_ids.data(),
                inbound_node_ids.data()
            );
        }
    }

    {
        NVTX_RANGE_COLORED("TS group counts/offsets", nvtx_colors::index_blue);
        #ifdef HAS_CUDA
        if (data.use_gpu) {
            allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
                data,
                data.active_node_ids.size(),
                outbound_node_ids.data(),
                inbound_node_ids.data()
            );
        } else
        #endif
        {
            allocate_and_compute_node_ts_group_counts_and_offsets_std(
                data,
                data.active_node_ids.size(),
                outbound_node_ids.data(),
                inbound_node_ids.data()
            );
        }
    }
    // outbound_node_ids and inbound_node_ids RAII-free on scope exit.
}

HOST size_t node_edge_index::get_memory_used(const TemporalGraphData& data) {
    size_t total_memory = 0;

    total_memory += data.node_group_outbound_offsets.size() * sizeof(size_t);
    total_memory += data.node_group_inbound_offsets.size() * sizeof(size_t);

    total_memory += data.node_ts_sorted_outbound_indices.size() * sizeof(size_t);
    total_memory += data.node_ts_sorted_inbound_indices.size() * sizeof(size_t);

    total_memory += data.count_ts_group_per_node_outbound.size() * sizeof(size_t);
    total_memory += data.count_ts_group_per_node_inbound.size() * sizeof(size_t);

    total_memory += data.node_ts_group_outbound_offsets.size() * sizeof(size_t);
    total_memory += data.node_ts_group_inbound_offsets.size() * sizeof(size_t);

    total_memory += data.outbound_forward_cumulative_weights_exponential.size() * sizeof(double);
    total_memory += data.outbound_backward_cumulative_weights_exponential.size() * sizeof(double);
    total_memory += data.inbound_backward_cumulative_weights_exponential.size() * sizeof(double);

    return total_memory;
}
