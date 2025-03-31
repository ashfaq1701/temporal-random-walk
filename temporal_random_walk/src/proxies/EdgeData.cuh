#ifndef EDGE_DATA_H
#define EDGE_DATA_H

#include <vector>
#include "../stores/edge_data.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA

__global__ void empty_kernel(bool* result, const EdgeDataStore* edge_data);

__global__ void size_kernel(size_t* result, const EdgeDataStore* edge_data);

__global__ void find_group_after_timestamp_kernel(size_t* result, const EdgeDataStore* edge_data, int64_t timestamp);

__global__ void find_group_before_timestamp_kernel(size_t* result, const EdgeDataStore* edge_data, int64_t timestamp);

__global__ void get_timestamp_group_range_kernel(SizeRange* result, const EdgeDataStore* edge_data, size_t group_idx);

#endif

class EdgeData {
public:

    EdgeDataStore* edge_data;
    bool owns_edge_data;

    std::vector<int> sources() const {
        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<int> result(edge_data->sources_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->sources,
                      edge_data->sources_size * sizeof(int),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<int>(edge_data->sources,
                                   edge_data->sources + edge_data->sources_size);
        }
    }

    std::vector<int> targets() const {
        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<int> result(edge_data->targets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->targets,
                      edge_data->targets_size * sizeof(int),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<int>(edge_data->targets,
                                   edge_data->targets + edge_data->targets_size);
        }
    }

    std::vector<int64_t> timestamps() const {
        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<int64_t> result(edge_data->timestamps_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->timestamps,
                      edge_data->timestamps_size * sizeof(int64_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<int64_t>(edge_data->timestamps,
                                       edge_data->timestamps + edge_data->timestamps_size);
        }
    }

    std::vector<size_t> timestamp_group_offsets() const {
        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<size_t> result(edge_data->timestamp_group_offsets_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->timestamp_group_offsets,
                      edge_data->timestamp_group_offsets_size * sizeof(size_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<size_t>(edge_data->timestamp_group_offsets,
                                     edge_data->timestamp_group_offsets + edge_data->timestamp_group_offsets_size);
        }
    }

    std::vector<int64_t> unique_timestamps() const {
        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<int64_t> result(edge_data->unique_timestamps_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->unique_timestamps,
                      edge_data->unique_timestamps_size * sizeof(int64_t),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<int64_t>(edge_data->unique_timestamps,
                                       edge_data->unique_timestamps + edge_data->unique_timestamps_size);
        }
    }

    std::vector<double> forward_cumulative_weights_exponential() const {
        if (!edge_data->forward_cumulative_weights_exponential) {
            return std::vector<double>();
        }

        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<double> result(edge_data->forward_cumulative_weights_exponential_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->forward_cumulative_weights_exponential,
                      edge_data->forward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<double>(edge_data->forward_cumulative_weights_exponential,
                                     edge_data->forward_cumulative_weights_exponential +
                                     edge_data->forward_cumulative_weights_exponential_size);
        }
    }

    std::vector<double> backward_cumulative_weights_exponential() const {
        if (!edge_data->backward_cumulative_weights_exponential) {
            return std::vector<double>();
        }

        #ifdef HAS_CUDA
        if (edge_data->use_gpu) {
            std::vector<double> result(edge_data->backward_cumulative_weights_exponential_size);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(result.data(), edge_data->backward_cumulative_weights_exponential,
                      edge_data->backward_cumulative_weights_exponential_size * sizeof(double),
                      cudaMemcpyDeviceToHost));
            return result;
        }
        else
        #endif
        {
            return std::vector<double>(edge_data->backward_cumulative_weights_exponential,
                                     edge_data->backward_cumulative_weights_exponential +
                                     edge_data->backward_cumulative_weights_exponential_size);
        }
    }

    explicit EdgeData(bool use_gpu);

    explicit EdgeData(EdgeDataStore* existing_edge_data);

    ~EdgeData();

    EdgeData& operator=(const EdgeData& other);

    void resize(size_t size) const;

    void clear() const;

    [[nodiscard]] size_t size() const;

    void set_size(size_t size) const;

    [[nodiscard]] bool empty() const;

    void add_edges(const std::vector<int>& sources, const std::vector<int>& targets, const std::vector<int64_t>& timestamps) const;

    void push_back(int source, int target, int64_t timestamp) const;

    [[nodiscard]] std::vector<Edge> get_edges() const;

    void update_timestamp_groups() const;

    void update_temporal_weights(double timescale_bound) const;

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;

    [[nodiscard]] size_t get_timestamp_group_count() const;

    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;

    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const;
};

#endif // EDGE_DATA_H
