#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#endif

#include "../../cuda_common/cuda_config.cuh"

#ifdef HAS_CUDA

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCUDA<GPUUsage>::update_timestamp_groups() {
    if (this->timestamps.empty()) {
        this->timestamp_group_offsets.clear();
        this->unique_timestamps.clear();
        return;
    }

    this->timestamp_group_offsets.clear();
    this->unique_timestamps.clear();

    const size_t n = this->timestamps.size();
    typename SelectVectorType<int, GPUUsage>::type flags(n);

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        this->timestamps.begin() + 1,
        this->timestamps.end(),
        this->timestamps.begin(),
        flags.begin() + 1,
        [] __host__ __device__ (const int64_t curr, const int64_t prev) { return curr != prev ? 1 : 0; });

    // First element is always a group start
    thrust::fill_n(flags.begin(), 1, 1);

    // Count total groups (sum of flags)
    size_t num_groups = thrust::reduce(flags.begin(), flags.end());

    // Resize output vectors
    this->timestamp_group_offsets.resize(num_groups + 1);  // +1 for end offset
    this->unique_timestamps.resize(num_groups);

    // Find positions of group boundaries
    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n),
        flags.begin(),
        this->timestamp_group_offsets.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });

    // Add final offset
    thrust::fill_n(this->timestamp_group_offsets.begin() + num_groups, 1, n);

    // Get unique timestamps at group boundaries
    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        this->timestamps.begin(),
        this->timestamps.end(),
        flags.begin(),
        this->unique_timestamps.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCUDA<GPUUsage>::compute_temporal_weights(const double timescale_bound) {
    const int64_t min_timestamp = this->timestamps[0];
    const int64_t max_timestamp = this->timestamps[this->timestamps.size() - 1];

    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = this->get_timestamp_group_count();
    this->forward_cumulative_weights_exponential.resize(num_groups);
    this->backward_cumulative_weights_exponential.resize(num_groups);

    typename SelectVectorType<double, GPUUsage>::type forward_weights(num_groups);
    typename SelectVectorType<double, GPUUsage>::type backward_weights(num_groups);

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->timestamps.data());
    const size_t* offsets_ptr = thrust::raw_pointer_cast(this->timestamp_group_offsets.data());

    // Calculate weights using device pointers
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_groups),
        thrust::make_zip_iterator(thrust::make_tuple(
            forward_weights.begin(),
            backward_weights.begin()
        )),
        [offsets_ptr, timestamps_ptr, max_timestamp, min_timestamp, timescale_bound, time_scale]
        __host__ __device__ (const size_t group) {
            const size_t start = offsets_ptr[group];
            const int64_t group_timestamp = timestamps_ptr[start];

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
    const double forward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        forward_weights.begin(),
        forward_weights.end()
    );
    const double backward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        backward_weights.begin(),
        backward_weights.end()
    );

    // Normalize weights
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        forward_weights.begin(),
        forward_weights.end(),
        forward_weights.begin(),
        [=] __host__ __device__ (const double w) { return w / forward_sum; }
    );
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        backward_weights.begin(),
        backward_weights.end(),
        backward_weights.begin(),
        [=] __host__ __device__ (const double w) { return w / backward_sum; }
    );

    // Compute cumulative sums
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        forward_weights.begin(),
        forward_weights.end(),
        this->forward_cumulative_weights_exponential.begin()
    );
    thrust::inclusive_scan(
        DEVICE_EXECUTION_POLICY,
        backward_weights.begin(),
        backward_weights.end(),
        this->backward_cumulative_weights_exponential.begin()
    );
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCUDA<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = thrust::raw_pointer_cast(this->unique_timestamps.data());
    const int64_t* end = begin + this->unique_timestamps.size();

    const auto it = thrust::upper_bound(begin, end, timestamp);
    return it - begin;
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCUDA<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = thrust::raw_pointer_cast(this->unique_timestamps.data());
    const int64_t* end = begin + this->unique_timestamps.size();

    const auto it = thrust::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

template<GPUUsageMode GPUUsage>
HOST EdgeDataCUDA<GPUUsage>* EdgeDataCUDA<GPUUsage>::to_device_ptr() {
    return nullptr;
}

template class EdgeDataCUDA<GPUUsageMode::ON_GPU>;
#endif
