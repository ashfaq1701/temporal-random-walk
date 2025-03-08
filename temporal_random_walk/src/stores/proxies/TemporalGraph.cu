#include "TemporalGraph.cuh"

#include "../../cuda_common/setup.cuh"

template <GPUUsageMode GPUUsage>
__global__ void get_edge_at_kernel(
    Edge* result,
    TemporalGraphCUDA<GPUUsage>* temporal_graph,
    RandomPicker<GPUUsage>* picker,
    curandState* rand_states,
    int64_t timestamp,
    bool forward) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph->get_edge_at_device(picker, &rand_states[threadIdx.x], timestamp, forward);
    }
}

template <GPUUsageMode GPUUsage>
__global__ void get_node_edge_at_kernel(
    Edge* result, TemporalGraphCUDA<GPUUsage>* temporal_graph,
    int node_id,
    RandomPicker<GPUUsage>* picker,
    curandState* rand_states,
    int64_t timestamp,
    bool forward) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph->get_node_edge_at_device(node_id, picker, &rand_states[threadIdx.x], timestamp, forward);
    }
}

template<GPUUsageMode GPUUsage>
TemporalGraph<GPUUsage>::TemporalGraph(
    bool directed,
    int64_t window,
    bool enable_weight_computation,
    double timescale_bound)
    : temporal_graph(new BaseType(directed, window, enable_weight_computation, timescale_bound)) {}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::sort_and_merge_edges(size_t start_idx)
{
    temporal_graph->sort_and_merge_edges(start_idx);
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::add_multiple_edges(const std::vector<Edge>& new_edges)
{
    typename ITemporalGraph<GPUUsage>::EdgeVector edge_vector;
    edge_vector.reserve(new_edges.size());
    for (const auto& edge : new_edges)
    {
        edge_vector.push_back(edge);
    }
    temporal_graph->add_multiple_edges(edge_vector);
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::update_temporal_weights()
{
    temporal_graph->update_temporal_weights();
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::delete_old_edges()
{
    temporal_graph->delete_old_edges();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const
{
    return temporal_graph->count_timestamps_less_than(timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const
{
    return temporal_graph->count_timestamps_greater_than(timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const
{
    return temporal_graph->count_node_timestamps_less_than(node_id, timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const
{
    return temporal_graph->count_node_timestamps_greater_than(node_id, timestamp);
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_edge_at(RandomPicker<GPUUsage>* picker, int64_t timestamp, bool forward) const
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        // Setup random state
        curandState* d_rand_states;
        cudaMalloc(&d_rand_states, sizeof(curandState));
        setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));

        // Allocate device memory for the result
        Edge host_result {-1, -1, -1};
        Edge* d_result;
        cudaMalloc(&d_result, sizeof(Edge));

        // Use the picker's to_device_ptr method to copy it to the device
        TemporalGraphCUDA<GPUUsage>* temporal_graph_cuda = static_cast<TemporalGraphCUDA<GPUUsage>*>(temporal_graph)->to_device_ptr();
        RandomPicker<GPUUsage>* d_picker = picker->to_device_ptr();

        // Launch the kernel
        get_edge_at_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            temporal_graph_cuda,
            d_picker,
            d_rand_states,
            timestamp,
            forward
        );

        // Copy the result back
        cudaMemcpy(&host_result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_result);
        cudaFree(d_picker);
        cudaFree(temporal_graph_cuda);
        cudaFree(d_rand_states);

        return host_result;
    }
    else
    #endif
    {
        return temporal_graph->get_edge_at_host(picker, timestamp, forward);
    }
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_node_edge_at(int node_id, RandomPicker<GPUUsage>* picker, int64_t timestamp, bool forward) const
{
    #ifdef HAS_CUDA
    if (GPUUsage == GPUUsageMode::ON_GPU) {
        // Setup random state
        curandState* d_rand_states;
        cudaMalloc(&d_rand_states, sizeof(curandState));
        setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));

        // Allocate device memory for the result
        Edge host_result {-1, -1, -1};
        Edge* d_result;
        cudaMalloc(&d_result, sizeof(Edge));

        // Use the picker's to_device_ptr method to copy it to the device
        TemporalGraphCUDA<GPUUsage>* temporal_graph_cuda = static_cast<TemporalGraphCUDA<GPUUsage>*>(temporal_graph)->to_device_ptr();
        RandomPicker<GPUUsage>* d_picker = picker->to_device_ptr();

        // Launch the kernel
        get_node_edge_at_kernel<GPUUsage><<<1, 1>>>(
            d_result,
            temporal_graph_cuda,
            node_id,
            d_picker,
            d_rand_states,
            timestamp,
            forward
        );

        // Copy the result back
        cudaMemcpy(&host_result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_result);
        cudaFree(d_picker);
        cudaFree(temporal_graph_cuda);
        cudaFree(d_rand_states);

        return host_result;
    }
    else
    #endif
    {
        return temporal_graph->get_node_edge_at_host(node_id, picker, timestamp, forward);
    }
}
template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_total_edges() const
{
    return temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_node_count() const
{
    return temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
int64_t TemporalGraph<GPUUsage>::get_latest_timestamp()
{
    return temporal_graph->get_latest_timestamp();
}

template<GPUUsageMode GPUUsage>
std::vector<int> TemporalGraph<GPUUsage>::get_node_ids() const
{
    std::vector<int> result;
    auto node_ids = temporal_graph->get_node_ids();
    for (int i = 0; i < node_ids.size(); i++)
    {
        result.push_back(node_ids[i]);
    }
    return result;
}

template<GPUUsageMode GPUUsage>
std::vector<Edge> TemporalGraph<GPUUsage>::get_edges()
{
    std::vector<Edge> result;
    auto edges = temporal_graph->get_edges();
    for (int i = 0; i < edges.size(); i++)
    {
        result.push_back(edges[i]);
    }
    return result;
}

template class TemporalGraph<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalGraph<GPUUsageMode::ON_GPU>;
#endif
