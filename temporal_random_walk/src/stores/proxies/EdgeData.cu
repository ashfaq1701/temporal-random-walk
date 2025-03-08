#include "EdgeData.cuh"

template <GPUUsageMode GPUUsage>
__global__ void check_empty_kernel(bool* result, EdgeDataCUDA<GPUUsage>* edge_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data->empty_device();
    }
}

template <GPUUsageMode GPUUsage>
__global__ void get_timestamp_group_range_kernel(SizeRange* result, EdgeDataCUDA<GPUUsage>* edge_data, size_t group_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data->get_timestamp_group_range_device(group_idx);
    }
}

template <GPUUsageMode GPUUsage>
__global__ void get_timestamp_group_count_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data->get_timestamp_group_count_device();
    }
}

template <GPUUsageMode GPUUsage>
__global__ void find_group_after_timestamp_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data, int64_t timestamp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data->find_group_after_timestamp_device(timestamp);
    }
}

template <GPUUsageMode GPUUsage>
__global__ void find_group_before_timestamp_kernel(size_t* result, EdgeDataCUDA<GPUUsage>* edge_data, int64_t timestamp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = edge_data->find_group_before_timestamp_device(timestamp);
    }
}

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(): edge_data(new BaseType()) {}

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(IEdgeData<GPUUsage>* edge_data): edge_data(edge_data) {}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::reserve(size_t size)
{
    edge_data->reserve(size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::clear()
{
    edge_data->clear();
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::size() const
{
    return edge_data->size();
}

template <GPUUsageMode GPUUsage>
bool EdgeData<GPUUsage>::empty() const
{
#ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        bool result = false;
        bool* d_result;
        cudaMalloc(&d_result, sizeof(bool));
        cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

        EdgeDataCUDA<GPUUsage>* edge_data_cuda = static_cast<EdgeDataCUDA<GPUUsage>*>(edge_data)->to_device_ptr();
        check_empty_kernel<GPUUsage><<<1, 1>>>(d_result, edge_data_cuda);

        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(edge_data_cuda);

        return result;
    }
    else
    #endif
    {
        return edge_data->empty();
    }
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::resize(size_t new_size)
{
    edge_data->resize(new_size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::add_edges(int* src, int* tgt, int64_t* ts, size_t size)
{
    edge_data->add_edges(src, tgt, ts, size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::push_back(int src, int tgt, int64_t ts)
{
    edge_data->push_back(src, tgt, ts);
}

template <GPUUsageMode GPUUsage>
std::vector<Edge> EdgeData<GPUUsage>::get_edges()
{
    std::vector<Edge> results;
    auto edges = edge_data->get_edges();
    for (auto edge : edges)
    {
        results.push_back(edge);
    }

    return results;
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_timestamp_groups()
{
    edge_data->update_timestamp_groups();
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_temporal_weights(double timescale_bound)
{
    edge_data->update_temporal_weights(timescale_bound);
}

template <GPUUsageMode GPUUsage>
std::pair<size_t, size_t> EdgeData<GPUUsage>::get_timestamp_group_range(size_t group_idx)
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        SizeRange host_result {0, 0};
        SizeRange* d_result;
        cudaMalloc(&d_result, sizeof(SizeRange));

        EdgeDataCUDA<GPUUsage>* edge_data_cuda = static_cast<EdgeDataCUDA<GPUUsage>*>(edge_data)->to_device_ptr();

        get_timestamp_group_range_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            edge_data_cuda,
            group_idx
        );

        cudaMemcpy(&host_result, d_result, sizeof(SizeRange), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(edge_data_cuda);

        return {host_result.from, host_result.to};
    }
    else
    #endif
    {
        auto group_range = edge_data->get_timestamp_group_range(group_idx);
        return {group_range.from, group_range.to};
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::get_timestamp_group_count() const
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        size_t host_result = 0;
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeDataCUDA<GPUUsage>* edge_data_cuda = static_cast<EdgeDataCUDA<GPUUsage>*>(edge_data)->to_device_ptr();

        get_timestamp_group_count_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            edge_data_cuda
        );

        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(edge_data_cuda);

        return host_result;
    }
    else
    #endif
    {
        return edge_data->get_timestamp_group_count();
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        size_t host_result = 0;
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeDataCUDA<GPUUsage>* edge_data_cuda = static_cast<EdgeDataCUDA<GPUUsage>*>(edge_data)->to_device_ptr();

        find_group_after_timestamp_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            edge_data_cuda,
            timestamp
        );

        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(edge_data_cuda);

        return host_result;
    }
    else
    #endif
    {
        return edge_data->find_group_after_timestamp(timestamp);
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const {
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        size_t host_result = 0;
        size_t* d_result;
        cudaMalloc(&d_result, sizeof(size_t));

        EdgeDataCUDA<GPUUsage>* edge_data_cuda = static_cast<EdgeDataCUDA<GPUUsage>*>(edge_data)->to_device_ptr();

        find_group_before_timestamp_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            edge_data_cuda,
            timestamp
        );

        cudaMemcpy(&host_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(edge_data_cuda);

        return host_result;
    }
    else
    #endif
    {
        return edge_data->find_group_before_timestamp(timestamp);
    }
}

template class EdgeData<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class EdgeData<GPUUsageMode::ON_GPU>;
#endif

