#include "edge_data.cuh"

#include <cmath>
#include <algorithm>
#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

#include "../common/error_handlers.cuh"
#include "../common/memory.cuh"
#include "../common/cuda_config.cuh"

HOST void edge_data::resize(EdgeDataStore *edge_data, const size_t size) {
    resize_memory(&edge_data->sources, edge_data->sources_size, size, edge_data->use_gpu);
    resize_memory(&edge_data->targets, edge_data->targets_size, size, edge_data->use_gpu);
    resize_memory(&edge_data->timestamps, edge_data->timestamps_size, size, edge_data->use_gpu);

    resize_memory(&edge_data->timestamp_group_offsets, edge_data->timestamp_group_offsets_size,  size, edge_data->use_gpu);
    resize_memory(&edge_data->unique_timestamps, edge_data->unique_timestamps_size, size, edge_data->use_gpu);

    set_size(edge_data, size);
}

HOST void edge_data::clear(EdgeDataStore *edge_data) {
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

HOST DEVICE size_t edge_data::size(const EdgeDataStore* edge_data) {
    return edge_data->timestamps_size;
}

HOST void edge_data::set_size(EdgeDataStore* edge_data, size_t size) {
    edge_data->sources_size = size;
    edge_data->targets_size = size;
    edge_data->timestamps_size = size;
}

HOST DEVICE bool edge_data::empty(const EdgeDataStore *edge_data) {
    return edge_data->timestamps_size == 0;
}

HOST void edge_data::add_edges(EdgeDataStore *edge_data, const int *sources, const int *targets, const int64_t *timestamps, const size_t size) {
    append_memory(&edge_data->sources, edge_data->sources_size, sources, size, edge_data->use_gpu);
    append_memory(&edge_data->targets, edge_data->targets_size, targets, size, edge_data->use_gpu);
    append_memory(&edge_data->timestamps, edge_data->timestamps_size, timestamps, size, edge_data->use_gpu);
}

HOST DataBlock<Edge> edge_data::get_edges(const EdgeDataStore *edge_data) {
    DataBlock<Edge> result(edge_data->timestamps_size, edge_data->use_gpu);

    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        thrust::device_ptr<int> d_sources(edge_data->sources);
        thrust::device_ptr<int> d_targets(edge_data->targets);
        thrust::device_ptr<int64_t> d_timestamps(edge_data->timestamps);
        thrust::device_ptr<Edge> d_result(result.data);

        thrust::transform(
            thrust::device,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(edge_data->timestamps_size),
            d_result,
            [d_sources, d_targets, d_timestamps] __device__ (const size_t i) {
                return Edge{
                    d_sources[static_cast<long>(i)],
                    d_targets[static_cast<long>(i)],
                    d_timestamps[static_cast<long>(i)]
                };
            }
        );

        CUDA_KERNEL_CHECK("After thrust transform in get_edges");
    }
    else
    #endif
    {
        for (size_t i = 0; i < edge_data->timestamps_size; i++) {
            result.data[i] = Edge{ edge_data->sources[i], edge_data->targets[i], edge_data->timestamps[i] };
        }
    }

    return result;
}

HOST SizeRange edge_data::get_timestamp_group_range(const EdgeDataStore *edge_data, size_t group_idx) {
    if (group_idx >= edge_data->unique_timestamps_size) {
        return SizeRange{0, 0};
    }

    return SizeRange{edge_data->timestamp_group_offsets[group_idx], edge_data->timestamp_group_offsets[group_idx + 1]};
}

HOST DEVICE size_t edge_data::get_timestamp_group_count(const EdgeDataStore *edge_data) {
    return edge_data->unique_timestamps_size;
}

HOST size_t edge_data::find_group_after_timestamp(const EdgeDataStore *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = std::upper_bound(begin, end, timestamp);
    return it - begin;
}

HOST size_t edge_data::find_group_before_timestamp(const EdgeDataStore *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

HOST void edge_data::update_timestamp_groups_std(EdgeDataStore *edge_data) {
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

HOST void edge_data::update_temporal_weights_std(EdgeDataStore *edge_data, const double timescale_bound) {
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

#ifdef HAS_CUDA

HOST void edge_data::update_timestamp_groups_cuda(EdgeDataStore *edge_data) {
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
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_flags, n * sizeof(int)));

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
    CUDA_KERNEL_CHECK("After thrust transform in update_timestamp_groups_cuda");

    // First element is always a group start
    thrust::fill_n(d_flags_ptr, 1, 1);
    CUDA_KERNEL_CHECK("After thrust fill_n in update_timestamp_groups_cuda");

    // Count total groups (sum of flags)
    const size_t num_groups = thrust::reduce(d_flags_ptr, d_flags_ptr + static_cast<long>(n));
    CUDA_KERNEL_CHECK("After thrust reduce in update_timestamp_groups_cuda");

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
    CUDA_KERNEL_CHECK("After thrust copy_if group boundaries in update_timestamp_groups_cuda");

    // Add final offset
    thrust::fill_n(d_group_offsets + static_cast<long>(num_groups), 1, n);
    CUDA_KERNEL_CHECK("After thrust fill_n final offset in update_timestamp_groups_cuda");

    // Get unique timestamps at group boundaries
    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        d_timestamps,
        d_timestamps + static_cast<long>(n),
        d_flags_ptr,
        d_unique_timestamps,
        [] HOST DEVICE (const int flag) { return flag == 1; });
    CUDA_KERNEL_CHECK("After thrust copy_if unique timestamps in update_timestamp_groups_cuda");

    // Free temporary memory
    CUDA_CHECK_AND_CLEAR(cudaFree(d_flags));
}

HOST void edge_data::update_temporal_weights_cuda(EdgeDataStore *edge_data, double timescale_bound) {
    if (edge_data->timestamps_size == 0) {
        clear_memory(&edge_data->forward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->forward_cumulative_weights_exponential_size = 0;

        clear_memory(&edge_data->backward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->backward_cumulative_weights_exponential_size = 0;
        return;
    }

    int64_t min_timestamp, max_timestamp;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&min_timestamp, edge_data->timestamps, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&max_timestamp, edge_data->timestamps + (edge_data->timestamps_size - 1), sizeof(int64_t), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_forward_weights, num_groups * sizeof(double)));
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, num_groups * sizeof(double)));

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
    CUDA_KERNEL_CHECK("After thrust transform weights calculation in update_temporal_weights_cuda");

    // Calculate sums
    double forward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups)
    );
    CUDA_KERNEL_CHECK("After thrust reduce forward weights in update_temporal_weights_cuda");

    double backward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups)
    );
    CUDA_KERNEL_CHECK("After thrust reduce backward weights in update_temporal_weights_cuda");

    // Normalize weights
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups),
        d_forward_weights_ptr,
        [=] HOST DEVICE (const double w) { return w / forward_sum; }
    );
    CUDA_KERNEL_CHECK("After thrust transform forward weight normalization in update_temporal_weights_cuda");

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups),
        d_backward_weights_ptr,
        [=] HOST DEVICE (const double w) { return w / backward_sum; }
    );
    CUDA_KERNEL_CHECK("After thrust transform backward weight normalization in update_temporal_weights_cuda");

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
    CUDA_KERNEL_CHECK("After thrust inclusive_scan forward weights in update_temporal_weights_cuda");

    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups),
        d_backward_cumulative
    );
    CUDA_KERNEL_CHECK("After thrust inclusive_scan backward weights in update_temporal_weights_cuda");

    // Free temporary memory
    CUDA_CHECK_AND_CLEAR(cudaFree(d_forward_weights));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_backward_weights));
}

DEVICE size_t edge_data::find_group_after_timestamp_device(const EdgeDataStore *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    const auto it = cuda::std::upper_bound(begin, end, timestamp);
    return it - begin;
}

DEVICE size_t edge_data::find_group_before_timestamp_device(const EdgeDataStore *edge_data, int64_t timestamp) {
    if (edge_data->unique_timestamps_size == 0) return 0;

    const int64_t* begin = edge_data->unique_timestamps;
    const int64_t* end = edge_data->unique_timestamps + edge_data->unique_timestamps_size;

    auto it = cuda::std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

HOST EdgeDataStore* edge_data::to_device_ptr(const EdgeDataStore* edge_data) {
    // Create a new EdgeData object on the device
    EdgeDataStore* device_edge_data;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_edge_data, sizeof(EdgeDataStore)));

    // Create a temporary copy to modify for device pointers
    EdgeDataStore temp_edge_data = *edge_data;
    temp_edge_data.owns_data = false;

    // If already using GPU, just copy the struct with its pointers
    if (!edge_data->use_gpu) {
        temp_edge_data.owns_data = true;

        // Copy each array to device if it exists
        if (edge_data->sources) {
            int* d_sources;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sources, edge_data->sources_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_sources, edge_data->sources, edge_data->sources_size * sizeof(int), cudaMemcpyHostToDevice));
            temp_edge_data.sources = d_sources;
        }

        if (edge_data->targets) {
            int* d_targets;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_targets, edge_data->targets_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_targets, edge_data->targets, edge_data->targets_size * sizeof(int), cudaMemcpyHostToDevice));
            temp_edge_data.targets = d_targets;
        }

        if (edge_data->timestamps) {
            int64_t* d_timestamps;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_timestamps, edge_data->timestamps_size * sizeof(int64_t)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamps, edge_data->timestamps, edge_data->timestamps_size * sizeof(int64_t), cudaMemcpyHostToDevice));
            temp_edge_data.timestamps = d_timestamps;
        }

        if (edge_data->timestamp_group_offsets) {
            size_t* d_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_timestamp_group_offsets, edge_data->timestamp_group_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamp_group_offsets, edge_data->timestamp_group_offsets,
                       edge_data->timestamp_group_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_edge_data.timestamp_group_offsets = d_timestamp_group_offsets;
        }

        if (edge_data->unique_timestamps) {
            int64_t* d_unique_timestamps;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_unique_timestamps, edge_data->unique_timestamps_size * sizeof(int64_t)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_unique_timestamps, edge_data->unique_timestamps,
                       edge_data->unique_timestamps_size * sizeof(int64_t), cudaMemcpyHostToDevice));
            temp_edge_data.unique_timestamps = d_unique_timestamps;
        }

        if (edge_data->forward_cumulative_weights_exponential) {
            double* d_forward_weights;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_forward_weights, edge_data->forward_cumulative_weights_exponential_size * sizeof(double)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_forward_weights, edge_data->forward_cumulative_weights_exponential,
                       edge_data->forward_cumulative_weights_exponential_size * sizeof(double), cudaMemcpyHostToDevice));
            temp_edge_data.forward_cumulative_weights_exponential = d_forward_weights;
        }

        if (edge_data->backward_cumulative_weights_exponential) {
            double* d_backward_weights;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_backward_weights, edge_data->backward_cumulative_weights_exponential_size * sizeof(double)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_backward_weights, edge_data->backward_cumulative_weights_exponential,
                       edge_data->backward_cumulative_weights_exponential_size * sizeof(double), cudaMemcpyHostToDevice));
            temp_edge_data.backward_cumulative_weights_exponential = d_backward_weights;
        }

        // Make sure use_gpu is set to true
        temp_edge_data.use_gpu = true;
    }

    CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_edge_data, &temp_edge_data, sizeof(EdgeDataStore), cudaMemcpyHostToDevice));

    temp_edge_data.owns_data = false;

    return device_edge_data;
}

#endif
