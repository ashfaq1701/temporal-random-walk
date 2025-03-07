#include "TemporalRandomWalkCPU.cuh"

#include "../utils/utils.h"
#include "../utils/rand_utils.cuh"
#include "../random/UniformRandomPicker.cuh"
#include "../random/LinearRandomPicker.cuh"
#include "../random/ExponentialIndexRandomPicker.cuh"
#include "../random/WeightBasedRandomPicker.cuh"


constexpr int DEFAULT_CONTEXT_WINDOW_LEN = 2;

template<GPUUsageMode GPUUsage>
TemporalRandomWalkCPU<GPUUsage>::TemporalRandomWalkCPU(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound,
    size_t n_threads):
    ITemporalRandomWalk<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound),
    n_threads(static_cast<int>(n_threads)), thread_pool(n_threads)
{
    this->temporal_graph = new typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
}

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
HOST RandomPicker<GPUUsage>* TemporalRandomWalkCPU<GPUUsage>::get_random_picker(const RandomPickerType* picker_type) const {
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
HOST WalkSet<GPUUsage> TemporalRandomWalkCPU<GPUUsage>::get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction)
{
    RandomPicker<GPUUsage>* edge_picker = get_random_picker(walk_bias);
    RandomPicker<GPUUsage>* start_picker = initial_edge_bias ? get_random_picker(initial_edge_bias) : edge_picker;

    auto repeated_node_ids = repeat_elements<GPUUsage>(get_node_ids(), num_walks_per_node);
    shuffle_vector_host<int, GPUUsage>(repeated_node_ids);
    auto distributed_node_ids = divide_vector<int, GPUUsage>(repeated_node_ids, n_threads);

    WalkSet<GPUUsage> walk_set(repeated_node_ids.size(), max_walk_len);

    auto generate_walks_thread = [this, &walk_set, &edge_picker, &start_picker, max_walk_len, walk_direction](const typename DividedVector<int, GPUUsage>::GroupIterator& group) {
        for (auto ptr = group.begin(); ptr != group.end(); ++ptr) {
            int walk_idx = ptr->index;
            int start_node_id = ptr->value;
            const bool should_walk_forward = get_should_walk_forward(walk_direction);

            generate_random_walk_and_time(
                walk_idx,
                walk_set,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward,
                start_node_id);
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(distributed_node_ids.num_groups);

    for (size_t i = 0; i < distributed_node_ids.num_groups; i++)
    {
        futures.push_back(thread_pool.enqueue(generate_walks_thread, distributed_node_ids.get_group(i)));
    }

    std::vector<std::vector<NodeWithTime>> walks;

    for (auto& future : futures) {
        future.wait();
    }

    if (start_picker != edge_picker)
    {
        delete start_picker;
    }
    delete edge_picker;

    return walk_set;
}

template<GPUUsageMode GPUUsage>
HOST WalkSet<GPUUsage> TemporalRandomWalkCPU<GPUUsage>::get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction) {

    RandomPicker<GPUUsage>* edge_picker = get_random_picker(walk_bias);
    RandomPicker<GPUUsage>* start_picker = initial_edge_bias ? get_random_picker(initial_edge_bias) : edge_picker;

    WalkSet<GPUUsage> walk_set(num_walks_total, max_walk_len);

    auto generate_walks_thread = [this, &walk_set, &edge_picker, &start_picker, max_walk_len, walk_direction](int start_idx, int num_walks) {
        for (int i = 0; i < num_walks; ++i) {
            int walk_idx = start_idx + i;
            bool should_walk_forward = get_should_walk_forward(walk_direction);

            generate_random_walk_and_time(
                walk_idx,
                walk_set,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward);
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);

    auto walks_per_thread = divide_number<GPUUsage>(num_walks_total, n_threads);

    int start_idx = 0;
    for (int num_walks : walks_per_thread) {
        futures.push_back(thread_pool.enqueue(generate_walks_thread, start_idx, num_walks));
        start_idx += num_walks;
    }

    for (auto& future : futures) {
        future.wait();
    }

    if (start_picker != edge_picker)
    {
        delete start_picker;
    }
    delete edge_picker;

    return walk_set;
}


template<GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCPU<GPUUsage>::generate_random_walk_and_time(
        int walk_idx,
        WalkSet<GPUUsage>& walk_set,
        RandomPicker<GPUUsage>* edge_picker,
        RandomPicker<GPUUsage>* start_picker,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id) const {

    Edge start_edge;
    if (start_node_id == -1) {
        start_edge = this->temporal_graph->get_edge_at_host(
            start_picker,
            -1,
            should_walk_forward);
    } else {
        start_edge = this->temporal_graph->get_node_edge_at_host(
            start_node_id,
            start_picker,
            -1,
            should_walk_forward
        );
    }

    if (start_edge.i == -1) {
        return;
    }

    int current_node = -1;
    size_t current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;
    auto [start_src, start_dst, start_ts] = start_edge;

    if (this->is_directed) {
        if (should_walk_forward) {
            walk_set.add_hop(walk_idx, start_src, current_timestamp);
            current_node = start_dst;
        } else {
            walk_set.add_hop(walk_idx, start_dst, current_timestamp);
            current_node = start_src;
        }
    } else {
        const int picked_node = start_node_id;
        walk_set.add_hop(walk_idx, picked_node, current_timestamp);
        current_node = pick_other_number({start_src, start_dst}, picked_node);
    }

    current_timestamp = start_ts;

    while (walk_set.get_walk_len(walk_idx) < max_walk_len && current_node != -1) {
        walk_set.add_hop(walk_idx, current_node, current_timestamp);

        auto [picked_src, picked_dst, picked_ts] = this->temporal_graph->get_node_edge_at_host(
            current_node,
            edge_picker,
            current_timestamp,
            should_walk_forward
        );

        if (picked_ts == -1) {
            current_node = -1;
            continue;
        }

        if (this->is_directed) {
            current_node = should_walk_forward ? picked_dst : picked_src;
        } else {
            current_node = pick_other_number({picked_src, picked_dst}, current_node);
        }

        current_timestamp = picked_ts;
    }

    if (!should_walk_forward) {
        walk_set.reverse_walk(walk_idx);
    }
}

template<GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCPU<GPUUsage>::add_multiple_edges(const typename ITemporalRandomWalk<GPUUsage>::EdgeVector& edge_infos) const {
    this->temporal_graph->add_multiple_edges(edge_infos);
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalRandomWalkCPU<GPUUsage>::get_node_count() const {
    return this->temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalRandomWalkCPU<GPUUsage>::get_edge_count() const {
    return this->temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::IntVector TemporalRandomWalkCPU<GPUUsage>::get_node_ids() const {
    return this->temporal_graph->get_node_ids();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::EdgeVector TemporalRandomWalkCPU<GPUUsage>::get_edges() const {
    return this->temporal_graph->get_edges();
}

template<GPUUsageMode GPUUsage>
HOST bool TemporalRandomWalkCPU<GPUUsage>::get_is_directed() const {
    return this->is_directed;
}

template<GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCPU<GPUUsage>::clear() {
    this->temporal_graph = new typename ITemporalRandomWalk<GPUUsage>::TemporalGraphType(
        this->is_directed, this->max_time_capacity,
        this->enable_weight_computation, this->timescale_bound);
}

template class TemporalRandomWalkCPU<GPUUsageMode::ON_CPU>;
