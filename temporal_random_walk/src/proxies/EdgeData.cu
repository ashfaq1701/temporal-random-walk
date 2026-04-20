#include "EdgeData.cuh"

#include <algorithm>

#include "TemporalRandomWalk.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "../common/cuda_config.cuh"
#endif

EdgeData::EdgeData(const bool use_gpu,
                   const bool enable_weight_computation,
                   const bool enable_temporal_node2vec)
    : self_owned_(std::make_unique<TemporalRandomWalk>(
          /*is_directed=*/true, use_gpu, /*max_time_capacity=*/-1,
          enable_weight_computation, enable_temporal_node2vec)),
      edge_data(&self_owned_->impl()->data()) {}

EdgeData::~EdgeData() = default;

void EdgeData::set_size(const size_t size) const {
    edge_data::set_size(*edge_data, size);
}

void EdgeData::add_edges(const std::vector<int>& sources,
                         const std::vector<int>& targets,
                         const std::vector<int64_t>& timestamps) const {
    if (sources.size() != targets.size() || sources.size() != timestamps.size()) {
        throw std::runtime_error("Vector sizes don't match for add_edges");
    }
    edge_data::add_edges(*edge_data, sources.data(), targets.data(),
                         timestamps.data(), sources.size());
}

void EdgeData::push_back(const int source, const int target, const int64_t timestamp) const {
    // add_edges takes host pointers and uses Buffer::append_from_host
    // internally, which does the H->D copy for GPU buffers.
    const int    srcs[1]  = {source};
    const int    tgts[1]  = {target};
    const int64_t tss[1]  = {timestamp};
    edge_data::add_edges(*edge_data, srcs, tgts, tss, 1);
}

void EdgeData::update_timestamp_groups() const {
#ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        edge_data::update_timestamp_groups_cuda(*edge_data);
        return;
    }
#endif
    edge_data::update_timestamp_groups_std(*edge_data);
}

void EdgeData::update_temporal_weights(const double timescale_bound) const {
#ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        edge_data::update_temporal_weights_cuda(*edge_data, timescale_bound);
        return;
    }
#endif
    edge_data::update_temporal_weights_std(*edge_data, timescale_bound);
}

std::pair<size_t, size_t> EdgeData::get_timestamp_group_range(const size_t group_idx) const {
    if (group_idx >= edge_data->unique_timestamps.size()) {
        return {0, 0};
    }

    const size_t* offsets = edge_data->timestamp_group_offsets.data();
    size_t from_val, to_val;

#ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&from_val, offsets + group_idx,
            sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&to_val, offsets + group_idx + 1,
            sizeof(size_t), cudaMemcpyDeviceToHost));
    } else
#endif
    {
        from_val = offsets[group_idx];
        to_val = offsets[group_idx + 1];
    }
    return {from_val, to_val};
}

size_t EdgeData::find_group_after_timestamp(const int64_t timestamp) const {
    const size_t n = edge_data->unique_timestamps.size();
    if (n == 0) return 0;

#ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        const int64_t* begin = edge_data->unique_timestamps.data();
        const auto it = thrust::upper_bound(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(begin),
            thrust::device_pointer_cast(begin + n),
            timestamp);
        return it - thrust::device_pointer_cast(begin);
    }
#endif
    return edge_data::find_group_after_timestamp(*edge_data, timestamp);
}

size_t EdgeData::find_group_before_timestamp(const int64_t timestamp) const {
    const size_t n = edge_data->unique_timestamps.size();
    if (n == 0) return 0;

#ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        const int64_t* begin = edge_data->unique_timestamps.data();
        const auto it = thrust::lower_bound(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(begin),
            thrust::device_pointer_cast(begin + n),
            timestamp);
        return (it - thrust::device_pointer_cast(begin)) - 1;
    }
#endif
    return edge_data::find_group_before_timestamp(*edge_data, timestamp);
}
