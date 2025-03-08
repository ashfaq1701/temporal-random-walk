#include "ITemporalRandomWalk.cuh"

#include "../random/LinearRandomPicker.cuh"
#include "../random/ExponentialIndexRandomPicker.cuh"
#include "../random/WeightBasedRandomPicker.cuh"
#include "../random/UniformRandomPicker.cuh"

bool get_should_walk_forward(const WalkDirection walk_direction) {
    switch (walk_direction)
    {
    case WalkDirection::Forward_In_Time:
        return true;
    case WalkDirection::Backward_In_Time:
        return false;
    default:
        throw std::invalid_argument("Invalid walk direction");
    }
}

template<GPUUsageMode GPUUsage>
HOST RandomPicker<GPUUsage>* ITemporalRandomWalk<GPUUsage>::get_random_picker(const RandomPickerType* picker_type) const {
    if (!picker_type) {
        throw std::invalid_argument("picker_type cannot be nullptr");
    }

    switch (*picker_type) {
    case Uniform:
        return new UniformRandomPicker<GPUUsage>();
    case Linear:
        return new LinearRandomPicker<GPUUsage>();
    case ExponentialIndex:
        return new ExponentialIndexRandomPicker<GPUUsage>();
    case ExponentialWeight:
        if (!this->enable_weight_computation) {
            throw std::invalid_argument("To enable weight based random pickers, set enable_weight_computation constructor argument to true.");
        }
        return new WeightBasedRandomPicker<GPUUsage>();
    default:
        throw std::invalid_argument("Invalid picker type");
    }
}

template<GPUUsageMode GPUUsage>
HOST void ITemporalRandomWalk<GPUUsage>::add_multiple_edges(const EdgeVector& edge_infos) const {
    this->temporal_graph->add_multiple_edges(edge_infos);
}

template<GPUUsageMode GPUUsage>
HOST size_t ITemporalRandomWalk<GPUUsage>::get_node_count() const {
    return this->temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
HOST size_t ITemporalRandomWalk<GPUUsage>::get_edge_count() const {
    return this->temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::IntVector ITemporalRandomWalk<GPUUsage>::get_node_ids() const {
    return this->temporal_graph->get_node_ids();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::EdgeVector ITemporalRandomWalk<GPUUsage>::get_edges() const {
    return this->temporal_graph->get_edges();
}

template<GPUUsageMode GPUUsage>
HOST bool ITemporalRandomWalk<GPUUsage>::get_is_directed() const {
    return this->is_directed;
}

template<GPUUsageMode GPUUsage>
HOST void ITemporalRandomWalk<GPUUsage>::clear() {
    this->temporal_graph = new TemporalGraphType(
        this->is_directed, this->max_time_capacity,
        this->enable_weight_computation, this->timescale_bound);
}

template class ITemporalRandomWalk<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class ITemporalRandomWalk<GPUUsageMode::ON_GPU>;
#endif
