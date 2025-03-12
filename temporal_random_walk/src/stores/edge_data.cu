#include "edge_data.cuh"

#include <thrust/device_vector.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>

#include "../common/memory.cuh"
#include "../common/cuda_config.cuh"

HOST void edge_data::reserve(EdgeData *edge_data, const size_t size) {
    allocate_memory(&edge_data->sources, size, edge_data->use_gpu);
    allocate_memory(&edge_data->targets, size, edge_data->use_gpu);
    allocate_memory(&edge_data->timestamps, size, edge_data->use_gpu);

    allocate_memory(&edge_data->timestamp_group_offsets, size, edge_data->use_gpu);
    allocate_memory(&edge_data->unique_timestamps, size, edge_data->use_gpu);
}

HOST void edge_data::clear(EdgeData *edge_data) {
    clear_memory(&edge_data->sources, edge_data->use_gpu);
    edge_data->sources_size = 0;

    clear_memory(&edge_data->targets, edge_data->use_gpu);
    edge_data->targets_size = 0;

    clear_memory(&edge_data->timestamps, edge_data->use_gpu);
    edge_data->timestamps_size = 0;

    clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
    edge_data->timestamp_group_offsets_size = 0;

    clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);
    edge_data->unique_timestamps_size = 0;

    clear_memory(&edge_data->forward_cumulative_weights_exponential, edge_data->use_gpu);
    edge_data->forward_cumulative_weights_exponential_size = 0;

    clear_memory(&edge_data->backward_cumulative_weights_exponential, edge_data->use_gpu);
    edge_data->backward_cumulative_weights_exponential_size = 0;
}

HOST size_t edge_data::size(const EdgeData* edge_data) {
    return edge_data->timestamps_size;
}

HOST void edge_data::set_size(EdgeData* edge_data, size_t size) {
    edge_data->sources_size = size;
    edge_data->targets_size = size;
    edge_data->timestamps_size = size;
}

HOST bool edge_data::empty(const EdgeData *edge_data) {
    return edge_data->timestamps_size == 0;
}

HOST void edge_data::add_edges(EdgeData *edge_data, const int *sources, const int *targets, const int64_t *timestamps, const size_t size) {
    append_memory(&edge_data->sources, edge_data->sources_size, sources, size, edge_data->use_gpu);
    append_memory(&edge_data->targets, edge_data->targets_size, targets, size, edge_data->use_gpu);
    append_memory(&edge_data->timestamps, edge_data->timestamps_size, timestamps, size, edge_data->use_gpu);
}

HOST DataBlock<Edge> edge_data::get_edges(const EdgeData *edge_data) {
    DataBlock<Edge> result(edge_data->timestamps_size, edge_data->use_gpu);

    for (size_t i = 0; i < edge_data->timestamps_size; i++) {
        result.data[i] = Edge{ edge_data->sources[i], edge_data->targets[i], edge_data->timestamps[i] };
    }

    return result;
}

HOST DEVICE SizeRange edge_data::get_timestamp_group_range(const EdgeData *edge_data, size_t group_idx) {
    if (group_idx >= edge_data->unique_timestamps_size) {
        return SizeRange{0, 0};
    }

    return SizeRange{edge_data->timestamp_group_offsets[group_idx], edge_data->timestamp_group_offsets[group_idx + 1]};
}

HOST DEVICE size_t edge_data::get_timestamp_group_count(const EdgeData *edge_data) {
    return edge_data->unique_timestamps_size;
}

HOST size_t find_group_after_timestamp(const EdgeData *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = std::upper_bound(begin, end, timestamp);
    return it - begin;
}

HOST size_t find_group_before_timestamp(const EdgeData *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

HOST void edge_data::update_timestamp_groups_std(EdgeData *edge_data) {
    if (edge_data->timestamps_size == 0) {
        clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
        clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);
        return;
    }

    clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
    clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);

    allocate_memory(&edge_data->timestamp_group_offsets, edge_data->timestamps_size + 1, edge_data->use_gpu);
    allocate_memory(&edge_data->unique_timestamps, edge_data->timestamps_size, edge_data->use_gpu);

    size_t current_index = 0;
    edge_data->timestamp_group_offsets[current_index] = 0;
    edge_data->unique_timestamps[current_index++] = edge_data->timestamps[0];

    for (size_t i = 1; i < edge_data->timestamps_size; i++) {
        if (edge_data->timestamps[i] != edge_data->timestamps[i-1]) {
            edge_data->timestamp_group_offsets[current_index] = i;
            edge_data->unique_timestamps[current_index++] = edge_data->timestamps[i];
        }
    }
    edge_data->timestamp_group_offsets[current_index] = edge_data->timestamps_size;

    edge_data->timestamp_group_offsets_size = current_index + 1;
    edge_data->unique_timestamps_size = current_index;

    resize_memory(
        &edge_data->timestamp_group_offsets,
        edge_data->timestamps_size + 1,
        edge_data->timestamp_group_offsets_size,
        edge_data->use_gpu);

    resize_memory(
        &edge_data->unique_timestamps,
        edge_data->timestamps_size,
        edge_data->unique_timestamps_size,
        edge_data->use_gpu);
}

HOST void edge_data::update_temporal_weights_std(EdgeData *edge_data, const double timescale_bound) {
    const int64_t min_timestamp = edge_data->timestamps[0];
    const int64_t max_timestamp = edge_data->timestamps[edge_data->timestamps_size - 1];
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count(edge_data);

    resize_memory(
        &edge_data->forward_cumulative_weights_exponential,
        edge_data->forward_cumulative_weights_exponential_size,
        num_groups,
        edge_data->use_gpu);
    edge_data->forward_cumulative_weights_exponential_size = num_groups;

    resize_memory(
        &edge_data->backward_cumulative_weights_exponential,
        edge_data->backward_cumulative_weights_exponential_size,
        num_groups,
        edge_data->use_gpu);
    edge_data->backward_cumulative_weights_exponential_size = num_groups;

    double forward_sum = 0.0, backward_sum = 0.0;

    // First calculate all weights and total sums
    for (size_t group = 0; group < num_groups; group++) {
        const size_t start = edge_data->timestamp_group_offsets[group];
        const int64_t group_timestamp = edge_data->timestamps[start];

        const auto time_diff_forward = static_cast<double>(max_timestamp - group_timestamp);
        const auto time_diff_backward = static_cast<double>(group_timestamp - min_timestamp);

        const double forward_scaled = timescale_bound > 0 ?
            time_diff_forward * time_scale : time_diff_forward;
        const double backward_scaled = timescale_bound > 0 ?
            time_diff_backward * time_scale : time_diff_backward;

        const double forward_weight = exp(forward_scaled);
        const double backward_weight = exp(backward_scaled);

        forward_sum += forward_weight;
        backward_sum += backward_weight;

        edge_data->forward_cumulative_weights_exponential[group] = forward_weight;
        edge_data->backward_cumulative_weights_exponential[group] = backward_weight;
    }

    // Then normalize and compute cumulative sums
    double forward_cumsum = 0.0, backward_cumsum = 0.0;
    for (size_t group = 0; group < num_groups; group++) {
        edge_data->forward_cumulative_weights_exponential[group] /= forward_sum;
        edge_data->backward_cumulative_weights_exponential[group] /= backward_sum;

        // Update with cumulative sums
        forward_cumsum += edge_data->forward_cumulative_weights_exponential[group];
        backward_cumsum += edge_data->backward_cumulative_weights_exponential[group];

        edge_data->forward_cumulative_weights_exponential[group] = forward_cumsum;
        edge_data->backward_cumulative_weights_exponential[group] = backward_cumsum;
    }
}

HOST void edge_data::update_timestamp_groups_cuda(EdgeData *edge_data) {
    if (edge_data->timestamps_size == 0) {
        // Just clear memory and update sizes
        clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
        edge_data->timestamp_group_offsets_size = 0;

        clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);
        edge_data->unique_timestamps_size = 0;
        return;
    }

    const size_t n = edge_data->timestamps_size;

    // Create flags vector on device
    int* d_flags = nullptr;
    cudaMalloc(&d_flags, n * sizeof(int));

    // Wrap raw pointers with thrust device pointers for algorithm use
    thrust::device_ptr<int64_t> d_timestamps(edge_data->timestamps);
    thrust::device_ptr<int> d_flags_ptr(d_flags);

    // Compute flags: 1 where timestamp changes, 0 otherwise
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_timestamps + 1,
        d_timestamps + static_cast<long>(n),
        d_timestamps,
        d_flags_ptr + 1,
        [] HOST DEVICE (const int64_t curr, const int64_t prev) { return curr != prev ? 1 : 0; });

    // First element is always a group start
    thrust::fill_n(d_flags_ptr, 1, 1);

    // Count total groups (sum of flags)
    const size_t num_groups = thrust::reduce(d_flags_ptr, d_flags_ptr + static_cast<long>(n));

    // Resize output arrays
    resize_memory(
        &edge_data->timestamp_group_offsets,
        edge_data->timestamp_group_offsets_size,
        num_groups + 1,
        edge_data->use_gpu);
    edge_data->timestamp_group_offsets_size = num_groups + 1;

    resize_memory(
        &edge_data->unique_timestamps,
        edge_data->unique_timestamps_size,
        num_groups,
        edge_data->use_gpu);
    edge_data->unique_timestamps_size = num_groups;

    // Wrap pointers for algorithm use
    thrust::device_ptr<size_t> d_group_offsets(edge_data->timestamp_group_offsets);
    thrust::device_ptr<int64_t> d_unique_timestamps(edge_data->unique_timestamps);

    // Find positions of group boundaries
    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n),
        d_flags_ptr,
        d_group_offsets,
        [] HOST DEVICE (const int flag) { return flag == 1; });

    // Add final offset
    thrust::fill_n(d_group_offsets + static_cast<long>(num_groups), 1, n);

    // Get unique timestamps at group boundaries
    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        d_timestamps,
        d_timestamps + static_cast<long>(n),
        d_flags_ptr,
        d_unique_timestamps,
        [] HOST DEVICE (const int flag) { return flag == 1; });

    // Free temporary memory
    cudaFree(d_flags);
}

HOST void edge_data::update_temporal_weights_cuda(EdgeData *edge_data, double timescale_bound) {
    if (edge_data->timestamps_size == 0) {
        clear_memory(&edge_data->forward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->forward_cumulative_weights_exponential_size = 0;

        clear_memory(&edge_data->backward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->backward_cumulative_weights_exponential_size = 0;
        return;
    }

    const int64_t min_timestamp = edge_data->timestamps[0];
    const int64_t max_timestamp = edge_data->timestamps[edge_data->timestamps_size - 1];

    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count(edge_data);

    // Allocate memory for the weights
    resize_memory(
        &edge_data->forward_cumulative_weights_exponential,
        edge_data->forward_cumulative_weights_exponential_size,
        num_groups,
        edge_data->use_gpu);
    edge_data->forward_cumulative_weights_exponential_size = num_groups;

    resize_memory(
        &edge_data->backward_cumulative_weights_exponential,
        edge_data->backward_cumulative_weights_exponential_size,
        num_groups,
        edge_data->use_gpu);
    edge_data->backward_cumulative_weights_exponential_size = num_groups;

    // Allocate temporary memory for unnormalized weights
    double* d_forward_weights = nullptr;
    double* d_backward_weights = nullptr;
    cudaMalloc(&d_forward_weights, num_groups * sizeof(double));
    cudaMalloc(&d_backward_weights, num_groups * sizeof(double));

    // Wrap raw pointers with thrust device pointers
    thrust::device_ptr<int64_t> d_timestamps(edge_data->timestamps);
    thrust::device_ptr<size_t> d_offsets(edge_data->timestamp_group_offsets);
    thrust::device_ptr<double> d_forward_weights_ptr(d_forward_weights);
    thrust::device_ptr<double> d_backward_weights_ptr(d_backward_weights);

    // Calculate weights
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_groups),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_forward_weights_ptr,
            d_backward_weights_ptr
        )),
        [d_offsets, d_timestamps, max_timestamp, min_timestamp, timescale_bound, time_scale]
        HOST DEVICE (const size_t group) {
            const size_t start = d_offsets[static_cast<long>(group)];
            const int64_t group_timestamp = d_timestamps[static_cast<long>(start)];

            const auto time_diff_forward = static_cast<double>(max_timestamp - group_timestamp);
            const auto time_diff_backward = static_cast<double>(group_timestamp - min_timestamp);

            const double forward_scaled = timescale_bound > 0 ?
                time_diff_forward * time_scale : time_diff_forward;
            const double backward_scaled = timescale_bound > 0 ?
                time_diff_backward * time_scale : time_diff_backward;

            return thrust::make_tuple(exp(forward_scaled), exp(backward_scaled));
        }
    );

    // Calculate sums
    double forward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups)
    );

    double backward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups)
    );

    // Normalize weights
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups),
        d_forward_weights_ptr,
        [=] HOST DEVICE (const double w) { return w / forward_sum; }
    );

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups),
        d_backward_weights_ptr,
        [=] HOST DEVICE (const double w) { return w / backward_sum; }
    );

    // Wrap result pointers
    thrust::device_ptr<double> d_forward_cumulative(edge_data->forward_cumulative_weights_exponential);
    thrust::device_ptr<double> d_backward_cumulative(edge_data->backward_cumulative_weights_exponential);

    // Compute cumulative sums
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups),
        d_forward_cumulative
    );

    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups),
        d_backward_cumulative
    );

    // Free temporary memory
    cudaFree(d_forward_weights);
    cudaFree(d_backward_weights);
}

DEVICE size_t find_group_after_timestamp_device(const EdgeData *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = cuda::std::upper_bound(begin, end, timestamp);
    return it - begin;
}

DEVICE size_t find_group_before_timestamp_device(const EdgeData *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    auto it = cuda::std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}
