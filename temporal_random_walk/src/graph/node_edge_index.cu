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
#include "../common/memory.cuh"

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

HOST MemoryView<size_t> node_edge_index::get_timestamp_offset_vector(
    const TemporalGraphData& data,
    const bool forward) {
    // Returns a raw (possibly-device) pointer + size. The pointer is only
    // dereferenceable on the side (host / device) matching data.use_gpu;
    // callers on the host side must copy through cudaMemcpy if use_gpu.
    if (data.is_directed && !forward) {
        return MemoryView<size_t>{
            const_cast<size_t*>(data.count_ts_group_per_node_inbound.data()),
            data.count_ts_group_per_node_inbound.size()
        };
    } else {
        return MemoryView<size_t>{
            const_cast<size_t*>(data.count_ts_group_per_node_outbound.data()),
            data.count_ts_group_per_node_outbound.size()
        };
    }
}

HOST size_t node_edge_index::get_timestamp_group_count(
    const TemporalGraphData& data,
    const int dense_node_id,
    const bool forward) {
    const MemoryView<size_t> offsets_block = get_timestamp_offset_vector(data, forward);
    const size_t* offsets      = offsets_block.data;
    const size_t  offsets_size = offsets_block.size;

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

        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (flags[i]) {
                group_indices_out[flag_scan[i]] = i;
            }
        }

        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_outbound; ++i) {
            if (!flags[i]) continue;
            const int node = outbound_node_ids[i];

            if (node >= 0 && node < static_cast<int>(node_count)) {
                #pragma omp atomic
                group_counts[node]++;
            }
        }

        data.count_ts_group_per_node_outbound.resize(node_count + 1);

        data.count_ts_group_per_node_outbound.data()[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);

        #pragma omp parallel for
        for (size_t i = 0; i < node_count; ++i) {
            data.count_ts_group_per_node_outbound.data()[i + 1] = group_counts[i];
        }
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

        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (flags[i]) {
                group_indices_out[flag_scan[i]] = i;
            }
        }

        std::vector<size_t> group_counts(node_count, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < num_inbound; ++i) {
            if (!flags[i]) continue;
            const int node = inbound_node_ids[i];
            #pragma omp atomic
            group_counts[node]++;
        }

        data.count_ts_group_per_node_inbound.resize(node_count + 1);

        data.count_ts_group_per_node_inbound.data()[0] = 0;
        parallel_inclusive_scan(group_counts.data(), node_count);

        #pragma omp parallel for
        for (size_t i = 0; i < node_count; ++i) {
            data.count_ts_group_per_node_inbound.data()[i + 1] = group_counts[i];
        }
    }
}

HOST void node_edge_index::update_temporal_weights_std(
    TemporalGraphData& data,
    const double timescale_bound) {
    const size_t node_index_capacity = data.node_group_outbound_offsets.size() - 1;
    const size_t outbound_groups_size = data.node_ts_group_outbound_offsets.size();

    data.outbound_forward_cumulative_weights_exponential.resize(outbound_groups_size);
    data.outbound_backward_cumulative_weights_exponential.resize(outbound_groups_size);

    const bool is_directed = data.node_group_inbound_offsets.size() > 0;

    if (is_directed) {
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();
        data.inbound_backward_cumulative_weights_exponential.resize(inbound_groups_size);
    }

    // Process outbound weights
    {
        auto outbound_offsets = get_timestamp_offset_vector(data, true);

        std::vector<size_t> group_to_node(outbound_groups_size);

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            for (size_t pos = out_start; pos < out_end; ++pos) {
                group_to_node[pos] = node;
            }
        }

        std::vector<int64_t> node_min_ts(node_index_capacity);
        std::vector<int64_t> node_max_ts(node_index_capacity);
        std::vector<double> node_time_scale(node_index_capacity);

        const auto* ts_group_indices = data.node_ts_group_outbound_offsets.data();
        const auto* edge_indices = data.node_ts_sorted_outbound_indices.data();
        const auto* timestamps = data.timestamps.data();

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            if (out_start >= out_end) {
                node_min_ts[node] = 0;
                node_max_ts[node] = 0;
                node_time_scale[node] = 1.0;
                continue;
            }

            const int64_t min_ts = timestamps[edge_indices[ts_group_indices[out_start]]];
            const int64_t max_ts = timestamps[edge_indices[ts_group_indices[out_end - 1]]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

            node_min_ts[node] = min_ts;
            node_max_ts[node] = max_ts;
            node_time_scale[node] = time_scale;
        }

        std::vector<double> raw_forward_weights(outbound_groups_size);
        std::vector<double> raw_backward_weights(outbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const size_t edge_start = ts_group_indices[pos];
            const size_t edge_end =
                (pos + 1 < outbound_groups_size && group_to_node[pos + 1] == node)
                    ? ts_group_indices[pos + 1]
                    : data.node_group_outbound_offsets.data()[node + 1];

            const auto group_size = static_cast<double>(edge_end - edge_start);

            const int64_t group_ts = timestamps[edge_indices[edge_start]];
            const int64_t min_ts = node_min_ts[node];
            const int64_t max_ts = node_max_ts[node];
            const double time_scale = node_time_scale[node];

            const double f_scaled = (timescale_bound > 0) ? static_cast<double>(max_ts - group_ts) * time_scale : static_cast<double>(max_ts - group_ts);
            const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);

            raw_forward_weights[pos] = group_size * std::exp(f_scaled);
            raw_backward_weights[pos] = group_size * std::exp(b_scaled);
        }

        std::vector<double> node_forward_sums(node_index_capacity, 0.0);
        std::vector<double> node_backward_sums(node_index_capacity, 0.0);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];

            #pragma omp atomic
            node_forward_sums[node] += raw_forward_weights[pos];

            #pragma omp atomic
            node_backward_sums[node] += raw_backward_weights[pos];
        }

        std::vector<double> normalized_forward_weights(outbound_groups_size);
        std::vector<double> normalized_backward_weights(outbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < outbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const double forward_sum = node_forward_sums[node];
            const double backward_sum = node_backward_sums[node];

            normalized_forward_weights[pos] = raw_forward_weights[pos] / forward_sum;
            normalized_backward_weights[pos] = raw_backward_weights[pos] / backward_sum;
        }

        auto* f_weights = data.outbound_forward_cumulative_weights_exponential.data();
        auto* b_weights = data.outbound_backward_cumulative_weights_exponential.data();

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t out_start = outbound_offsets.data[node];
            const size_t out_end = outbound_offsets.data[node + 1];

            if (out_start >= out_end) continue;

            double f_cumsum = 0.0;
            double b_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                f_cumsum += normalized_forward_weights[pos];
                b_cumsum += normalized_backward_weights[pos];
                f_weights[pos] = f_cumsum;
                b_weights[pos] = b_cumsum;
            }
        }
    }

    // Process inbound weights (only backward)
    if (is_directed) {
        auto inbound_offsets = get_timestamp_offset_vector(data, false);
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();

        std::vector<size_t> group_to_node(inbound_groups_size);

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            for (size_t pos = in_start; pos < in_end; ++pos) {
                group_to_node[pos] = node;
            }
        }

        std::vector<int64_t> node_min_ts(node_index_capacity);
        std::vector<int64_t> node_max_ts(node_index_capacity);
        std::vector<double> node_time_scale(node_index_capacity);

        const auto* ts_group_indices = data.node_ts_group_inbound_offsets.data();
        const auto* edge_indices = data.node_ts_sorted_inbound_indices.data();
        const auto* timestamps = data.timestamps.data();

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            if (in_start >= in_end) {
                node_min_ts[node] = 0;
                node_max_ts[node] = 0;
                node_time_scale[node] = 1.0;
                continue;
            }

            const int64_t min_ts = timestamps[edge_indices[ts_group_indices[in_start]]];
            const int64_t max_ts = timestamps[edge_indices[ts_group_indices[in_end - 1]]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

            node_min_ts[node] = min_ts;
            node_max_ts[node] = max_ts;
            node_time_scale[node] = time_scale;
        }

        std::vector<double> raw_backward_weights(inbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const size_t edge_start = ts_group_indices[pos];
            const size_t edge_end =
                (pos + 1 < inbound_groups_size && group_to_node[pos + 1] == node)
                    ? ts_group_indices[pos + 1]
                    : data.node_group_inbound_offsets.data()[node + 1];

            const auto group_size = static_cast<double>(edge_end - edge_start);

            const int64_t group_ts = timestamps[edge_indices[edge_start]];
            const int64_t min_ts = node_min_ts[node];
            const double time_scale = node_time_scale[node];

            const double b_scaled = (timescale_bound > 0) ? static_cast<double>(group_ts - min_ts) * time_scale : static_cast<double>(group_ts - min_ts);
            raw_backward_weights[pos] = group_size * std::exp(b_scaled);
        }

        std::vector<double> node_backward_sums(node_index_capacity, 0.0);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];

            #pragma omp atomic
            node_backward_sums[node] += raw_backward_weights[pos];
        }

        std::vector<double> normalized_backward_weights(inbound_groups_size);

        #pragma omp parallel for
        for (size_t pos = 0; pos < inbound_groups_size; ++pos) {
            const size_t node = group_to_node[pos];
            const double backward_sum = node_backward_sums[node];
            normalized_backward_weights[pos] = raw_backward_weights[pos] / backward_sum;
        }

        auto* b_weights = data.inbound_backward_cumulative_weights_exponential.data();

        #pragma omp parallel for
        for (size_t node = 0; node < node_index_capacity; ++node) {
            const size_t in_start = inbound_offsets.data[node];
            const size_t in_end = inbound_offsets.data[node + 1];

            if (in_start >= in_end) continue;

            double b_cumsum = 0.0;
            for (size_t pos = in_start; pos < in_end; ++pos) {
                b_cumsum += normalized_backward_weights[pos];
                b_weights[pos] = b_cumsum;
            }
        }
    }
}

/**
 * Cuda implementations
 */
#ifdef HAS_CUDA

HOST void node_edge_index::compute_node_group_offsets_cuda(TemporalGraphData& data) {
    const size_t num_edges = data.timestamps.size();
    const bool is_directed = data.is_directed;

    size_t* outbound_offsets_ptr = data.node_group_outbound_offsets.data();
    size_t* inbound_offsets_ptr = is_directed ? data.node_group_inbound_offsets.data() : nullptr;
    int* src_ptr = data.sources.data();
    int* tgt_ptr = data.targets.data();

    // Count edges per node using 64-bit atomics on the full size_t slot.
    // Rationale: outbound_offsets_ptr / inbound_offsets_ptr are size_t (64-bit)
    // arrays. The 32-bit reinterpret that used to live here silently overflowed
    // for any node reaching 2^32 incident edges and was endian-dependent. CUDA
    // provides atomicAdd on unsigned long long natively on all compute
    // capabilities >= 3.5 (we target 75/80/86/89/90).
    static_assert(sizeof(size_t) == sizeof(unsigned long long),
                  "size_t must be 64-bit for compute_node_group_offsets_cuda's "
                  "atomicAdd; revisit the cast if this platform differs.");

    auto counter_device_lambda = [
                outbound_offsets_ptr, inbound_offsets_ptr,
                src_ptr, tgt_ptr, is_directed] DEVICE (const size_t i) {
        const int src_idx = src_ptr[i];
        const int tgt_idx = tgt_ptr[i];

        atomicAdd(reinterpret_cast<unsigned long long *>(&outbound_offsets_ptr[src_idx + 1]),
                  static_cast<unsigned long long>(1));
        if (is_directed) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&inbound_offsets_ptr[tgt_idx + 1]),
                      static_cast<unsigned long long>(1));
        } else {
            atomicAdd(reinterpret_cast<unsigned long long *>(&outbound_offsets_ptr[tgt_idx + 1]),
                      static_cast<unsigned long long>(1));
        }
    };

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        counter_device_lambda);
    CUDA_KERNEL_CHECK("After thrust for_each in compute_node_group_offsets_cuda");

    thrust::device_ptr<size_t> d_outbound_offsets(outbound_offsets_ptr);
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_outbound_offsets + 1,
        d_outbound_offsets + static_cast<long>(data.node_group_outbound_offsets.size()),
        d_outbound_offsets + 1
    );
    CUDA_KERNEL_CHECK("After thrust inclusive_scan outbound in compute_node_group_offsets_cuda");

    if (is_directed) {
        const thrust::device_ptr<size_t> d_inbound_offsets(inbound_offsets_ptr);
        thrust::inclusive_scan(
            DEVICE_EXECUTION_POLICY,
            d_inbound_offsets + 1,
            d_inbound_offsets + static_cast<long>(data.node_group_inbound_offsets.size()),
            d_inbound_offsets + 1
        );
        CUDA_KERNEL_CHECK("After thrust inclusive_scan inbound in compute_node_group_offsets_cuda");
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

        thrust::device_vector<size_t> flag_scan(num_edges + 1, 0);
        thrust::exclusive_scan(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            flag_scan.begin()
        );

        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    group_indices_out[flag_scan_ptr[i]] = i;
                }
            }
        );

        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, outbound_node_ids, group_counts_ptr, node_count] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    const int node = outbound_node_ids[i];
                    if (node >= 0 && node < node_count) {
                        atomicAdd(&group_counts_ptr[node], 1u);
                    }
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

        thrust::device_vector<size_t> flag_scan(num_edges + 1, 0);
        thrust::exclusive_scan(
            DEVICE_EXECUTION_POLICY,
            flags.begin(),
            flags.end(),
            flag_scan.begin()
        );

        auto flag_scan_ptr = thrust::raw_pointer_cast(flag_scan.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, flag_scan_ptr, group_indices_out] DEVICE(const size_t i) {
                if (flags_ptr[i]) {
                    group_indices_out[flag_scan_ptr[i]] = i;
                }
            }
        );

        thrust::device_vector<unsigned int> group_counts(node_count, 0);
        auto group_counts_ptr = thrust::raw_pointer_cast(group_counts.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_edges),
            [flags_ptr, inbound_node_ids, group_counts_ptr, node_count] DEVICE(size_t i) {
                if (flags_ptr[i]) {
                    const int node = inbound_node_ids[i];
                    if (node >= 0 && node < node_count) {
                        atomicAdd(&group_counts_ptr[node], 1u);
                    }
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

HOST void node_edge_index::update_temporal_weights_cuda(
    TemporalGraphData& data,
    double timescale_bound) {
    size_t node_index_capacity = data.node_group_outbound_offsets.size() - 1;
    const size_t outbound_groups_size = data.node_ts_group_outbound_offsets.size();

    data.outbound_forward_cumulative_weights_exponential.resize(outbound_groups_size);
    data.outbound_backward_cumulative_weights_exponential.resize(outbound_groups_size);

    const bool is_directed = (data.node_group_inbound_offsets.size() > 0);
    if (is_directed) {
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();
        data.inbound_backward_cumulative_weights_exponential.resize(inbound_groups_size);
    }

    int64_t* timestamps_ptr = data.timestamps.data();

    auto build_group_to_node = [](size_t* offsets_ptr,
                                  const size_t node_count,
                                  const size_t groups_size,
                                  thrust::device_vector<size_t>& group_to_node) {
        thrust::upper_bound(
            DEVICE_EXECUTION_POLICY,
            offsets_ptr,
            offsets_ptr + node_count + 1,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(groups_size),
            group_to_node.begin()
        );

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            group_to_node.begin(),
            [] DEVICE(const size_t x) { return (x == 0) ? 0 : (x - 1); }
        );
    };

    // === OUTBOUND weights ===
    {
        MemoryView<size_t> outbound_offsets = get_timestamp_offset_vector(data, true);
        size_t* outbound_offsets_ptr = outbound_offsets.data;

        thrust::device_vector<size_t> group_to_node(outbound_groups_size);
        build_group_to_node(outbound_offsets_ptr, node_index_capacity, outbound_groups_size, group_to_node);
        auto group_to_node_ptr = thrust::raw_pointer_cast(group_to_node.data());

        thrust::device_vector<int64_t> node_min_ts(node_index_capacity);
        thrust::device_vector<int64_t> node_max_ts(node_index_capacity);
        thrust::device_vector<double>  node_time_scale(node_index_capacity);

        auto node_min_ts_ptr     = thrust::raw_pointer_cast(node_min_ts.data());
        auto node_max_ts_ptr     = thrust::raw_pointer_cast(node_max_ts.data());
        auto node_time_scale_ptr = thrust::raw_pointer_cast(node_time_scale.data());

        size_t* outbound_indices_ptr       = data.node_ts_sorted_outbound_indices.data();
        size_t* outbound_group_indices_ptr = data.node_ts_group_outbound_offsets.data();

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [=] DEVICE(const size_t node) {
                const size_t out_start = outbound_offsets_ptr[node];
                const size_t out_end   = outbound_offsets_ptr[node + 1];

                if (out_start >= out_end) {
                    node_min_ts_ptr[node] = 0;
                    node_max_ts_ptr[node] = 0;
                    node_time_scale_ptr[node] = 1.0;
                    return;
                }

                const size_t first_group_start = outbound_group_indices_ptr[out_start];
                const size_t last_group_start  = outbound_group_indices_ptr[out_end - 1];

                const int64_t min_ts = timestamps_ptr[outbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[outbound_indices_ptr[last_group_start]];

                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? (timescale_bound / time_diff)
                    : 1.0;

                node_min_ts_ptr[node] = min_ts;
                node_max_ts_ptr[node] = max_ts;
                node_time_scale_ptr[node] = time_scale;
            }
        );

        thrust::device_vector<double> raw_forward_weights(outbound_groups_size);
        thrust::device_vector<double> raw_backward_weights(outbound_groups_size);

        auto raw_forward_weights_ptr  = thrust::raw_pointer_cast(raw_forward_weights.data());
        auto raw_backward_weights_ptr = thrust::raw_pointer_cast(raw_backward_weights.data());

        size_t* node_group_outbound_offsets = data.node_group_outbound_offsets.data();

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];

                const size_t edge_start = outbound_group_indices_ptr[pos];

                const size_t edge_end =
                    (pos + 1 < outbound_groups_size && group_to_node_ptr[pos + 1] == node)
                        ? outbound_group_indices_ptr[pos + 1]
                        : node_group_outbound_offsets[node + 1];

                const auto group_size = static_cast<double>(edge_end - edge_start);

                const int64_t group_ts  = timestamps_ptr[outbound_indices_ptr[edge_start]];

                const int64_t min_ts = node_min_ts_ptr[node];
                const int64_t max_ts = node_max_ts_ptr[node];
                const double  scale  = node_time_scale_ptr[node];

                const double tf = static_cast<double>(max_ts - group_ts) * scale;
                const double tb = static_cast<double>(group_ts - min_ts) * scale;

                raw_forward_weights_ptr[pos]  = group_size * exp(tf);
                raw_backward_weights_ptr[pos] = group_size * exp(tb);
            }
        );

        thrust::device_vector<double> node_forward_sums(node_index_capacity, 0.0);
        thrust::device_vector<double> node_backward_sums(node_index_capacity, 0.0);

        auto node_forward_sums_ptr  = thrust::raw_pointer_cast(node_forward_sums.data());
        auto node_backward_sums_ptr = thrust::raw_pointer_cast(node_backward_sums.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                atomicAdd(&node_forward_sums_ptr[node],  raw_forward_weights_ptr[pos]);
                atomicAdd(&node_backward_sums_ptr[node], raw_backward_weights_ptr[pos]);
            }
        );

        thrust::device_vector<double> normalized_forward_weights(outbound_groups_size);
        thrust::device_vector<double> normalized_backward_weights(outbound_groups_size);

        auto normalized_forward_weights_ptr  = thrust::raw_pointer_cast(normalized_forward_weights.data());
        auto normalized_backward_weights_ptr = thrust::raw_pointer_cast(normalized_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(outbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const double fsum = node_forward_sums_ptr[node];
                const double bsum = node_backward_sums_ptr[node];

                normalized_forward_weights_ptr[pos]  = raw_forward_weights_ptr[pos]  / fsum;
                normalized_backward_weights_ptr[pos] = raw_backward_weights_ptr[pos] / bsum;
            }
        );

        double* final_forward  = data.outbound_forward_cumulative_weights_exponential.data();
        double* final_backward = data.outbound_backward_cumulative_weights_exponential.data();

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_forward_weights_ptr,
            final_forward
        );

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_backward_weights_ptr,
            final_backward
        );

        CUDA_KERNEL_CHECK("After outbound weights processing in update_temporal_weights_cuda");
    }

    // === INBOUND weights ===
    if (is_directed) {
        node_index_capacity = data.node_group_inbound_offsets.size() - 1;

        MemoryView<size_t> inbound_offsets = get_timestamp_offset_vector(data, false);
        size_t* inbound_offsets_ptr = inbound_offsets.data;
        const size_t inbound_groups_size = data.node_ts_group_inbound_offsets.size();

        thrust::device_vector<size_t> group_to_node(inbound_groups_size);
        build_group_to_node(inbound_offsets_ptr, node_index_capacity, inbound_groups_size, group_to_node);
        auto group_to_node_ptr = thrust::raw_pointer_cast(group_to_node.data());

        thrust::device_vector<int64_t> node_min_ts(node_index_capacity);
        thrust::device_vector<int64_t> node_max_ts(node_index_capacity);
        thrust::device_vector<double>  node_time_scale(node_index_capacity);

        auto node_min_ts_ptr     = thrust::raw_pointer_cast(node_min_ts.data());
        auto node_max_ts_ptr     = thrust::raw_pointer_cast(node_max_ts.data());
        auto node_time_scale_ptr = thrust::raw_pointer_cast(node_time_scale.data());

        size_t* inbound_indices_ptr       = data.node_ts_sorted_inbound_indices.data();
        size_t* inbound_group_indices_ptr = data.node_ts_group_inbound_offsets.data();

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(node_index_capacity),
            [=] DEVICE(const size_t node) {
                const size_t in_start = inbound_offsets_ptr[node];
                const size_t in_end   = inbound_offsets_ptr[node + 1];

                if (in_start >= in_end) {
                    node_min_ts_ptr[node] = 0;
                    node_max_ts_ptr[node] = 0;
                    node_time_scale_ptr[node] = 1.0;
                    return;
                }

                const size_t first_group_start = inbound_group_indices_ptr[in_start];
                const size_t last_group_start  = inbound_group_indices_ptr[in_end - 1];

                const int64_t min_ts = timestamps_ptr[inbound_indices_ptr[first_group_start]];
                const int64_t max_ts = timestamps_ptr[inbound_indices_ptr[last_group_start]];

                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0.0 && time_diff > 0.0)
                    ? (timescale_bound / time_diff)
                    : 1.0;

                node_min_ts_ptr[node] = min_ts;
                node_max_ts_ptr[node] = max_ts;
                node_time_scale_ptr[node] = time_scale;
            }
        );

        thrust::device_vector<double> raw_backward_weights(inbound_groups_size);
        auto raw_backward_weights_ptr = thrust::raw_pointer_cast(raw_backward_weights.data());

        size_t* node_group_inbound_offsets_ptr = data.node_group_inbound_offsets.data();

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];

                const size_t edge_start = inbound_group_indices_ptr[pos];

                const size_t edge_end =
                    (pos + 1 < inbound_groups_size && group_to_node_ptr[pos + 1] == node)
                        ? inbound_group_indices_ptr[pos + 1]
                        : node_group_inbound_offsets_ptr[node + 1];

                const auto group_size = static_cast<double>(edge_end - edge_start);

                const int64_t group_ts  = timestamps_ptr[inbound_indices_ptr[edge_start]];

                const int64_t min_ts = node_min_ts_ptr[node];
                const double  scale  = node_time_scale_ptr[node];

                const double tb = static_cast<double>(group_ts - min_ts) * scale;
                raw_backward_weights_ptr[pos] = group_size * exp(tb);
            }
        );

        thrust::device_vector<double> node_backward_sums(node_index_capacity, 0.0);
        auto node_backward_sums_ptr = thrust::raw_pointer_cast(node_backward_sums.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                atomicAdd(&node_backward_sums_ptr[node], raw_backward_weights_ptr[pos]);
            }
        );

        thrust::device_vector<double> normalized_backward_weights(inbound_groups_size);
        auto normalized_backward_weights_ptr = thrust::raw_pointer_cast(normalized_backward_weights.data());

        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(inbound_groups_size),
            [=] DEVICE(const size_t pos) {
                const size_t node = group_to_node_ptr[pos];
                const double bsum = node_backward_sums_ptr[node];
                normalized_backward_weights_ptr[pos] = raw_backward_weights_ptr[pos] / bsum;
            }
        );

        double* final_backward = data.inbound_backward_cumulative_weights_exponential.data();

        thrust::inclusive_scan_by_key(
            DEVICE_EXECUTION_POLICY,
            group_to_node.begin(), group_to_node.end(),
            normalized_backward_weights_ptr,
            final_backward
        );

        CUDA_KERNEL_CHECK("After inbound weights processing in update_temporal_weights_cuda");
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
