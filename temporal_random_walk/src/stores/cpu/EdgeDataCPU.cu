#include "EdgeDataCPU.cuh"
#include <algorithm>

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::update_timestamp_groups() {
    if (this->timestamps.empty()) {
        this->timestamp_group_offsets.clear();
        this->unique_timestamps.clear();
        return;
    }

    this->timestamp_group_offsets.clear();
    this->unique_timestamps.clear();

    this->timestamp_group_offsets.push_back(0);
    this->unique_timestamps.push_back(this->timestamps[0]);

    for (size_t i = 1; i < this->timestamps.size(); i++) {
        if (this->timestamps[i] != this->timestamps[i-1]) {
            this->timestamp_group_offsets.push_back(i);
            this->unique_timestamps.push_back(this->timestamps[i]);
        }
    }
    this->timestamp_group_offsets.push_back(this->timestamps.size());
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::compute_temporal_weights(const double timescale_bound)
{
    const int64_t min_timestamp = this->timestamps[0];
    const int64_t max_timestamp = this->timestamps.back();
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = this->get_timestamp_group_count();
    this->forward_cumulative_weights_exponential.resize(num_groups);
    this->backward_cumulative_weights_exponential.resize(num_groups);

    double forward_sum = 0.0, backward_sum = 0.0;

    // First calculate all weights and total sums
    for (size_t group = 0; group < num_groups; group++) {
        const size_t start = this->timestamp_group_offsets[group];
        const int64_t group_timestamp = this->timestamps[start];

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

        this->forward_cumulative_weights_exponential[group] = forward_weight;
        this->backward_cumulative_weights_exponential[group] = backward_weight;
    }

    // Then normalize and compute cumulative sums
    double forward_cumsum = 0.0, backward_cumsum = 0.0;
    for (size_t group = 0; group < num_groups; group++) {
        this->forward_cumulative_weights_exponential[group] /= forward_sum;
        this->backward_cumulative_weights_exponential[group] /= backward_sum;

        // Update with cumulative sums
        forward_cumsum += this->forward_cumulative_weights_exponential[group];
        backward_cumsum += this->backward_cumulative_weights_exponential[group];

        this->forward_cumulative_weights_exponential[group] = forward_cumsum;
        this->backward_cumulative_weights_exponential[group] = backward_cumsum;
    }
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = this->unique_timestamps.data();
    const int64_t* end = this->unique_timestamps.data() + this->unique_timestamps.size();

    const auto it = std::upper_bound(begin, end, timestamp);
    return it - begin;
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = this->unique_timestamps.data();
    const int64_t* end = this->unique_timestamps.data() + this->unique_timestamps.size();

    auto it = std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

template<GPUUsageMode GPUUsage>
HOST EdgeDataCPU<GPUUsage>* EdgeDataCPU<GPUUsage>::to_device_ptr() {
    throw std::runtime_error("Unsupported operation");
}

template class EdgeDataCPU<GPUUsageMode::ON_CPU>;
