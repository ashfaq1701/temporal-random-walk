#include "TemporalGraphCPU.cuh"

#include "NodeMappingCPU.cuh"
#include "EdgeDataCPU.cuh"
#include "NodeEdgeIndexCPU.cuh"

#include <algorithm>
#include <utils/rand_utils.cuh>

#include "../../random/IndexBasedRandomPicker.cuh"
#include "../../random/WeightBasedRandomPicker.cuh"
#include "../../random/RandomPicker.h"

#include "../../utils/utils.h"

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCPU<GPUUsage>::TemporalGraphCPU(
    const bool directed,
    const int64_t window,
    const bool enable_weight_computation,
    const double timescale_bound)
    : ITemporalGraph<GPUUsage>(directed, window, enable_weight_computation, timescale_bound)
{
    this->node_index = new NodeEdgeIndexCPU<GPUUsage>();
    this->edges = new EdgeDataCPU<GPUUsage>();
    this->node_mapping = new NodeMappingCPU<GPUUsage>();
}

template<GPUUsageMode GPUUsage>
HOST void TemporalGraphCPU<GPUUsage>::add_multiple_edges(const typename ITemporalGraph<GPUUsage>::EdgeVector& new_edges) {
    if (new_edges.empty()) return;

    const size_t start_idx = this->edges->size();
    this->edges->reserve(start_idx + new_edges.size());

    typename ITemporalGraph<GPUUsage>::IntVector sources;
    typename ITemporalGraph<GPUUsage>::IntVector targets;
    typename ITemporalGraph<GPUUsage>::Int64TVector timestamps;

    for (const auto& [src, tgt, ts] : new_edges) {
        if (!this->is_directed && src > tgt) {
            sources.push_back(tgt);
            targets.push_back(src);
        } else {
            sources.push_back(src);
            targets.push_back(tgt);
        }
        timestamps.push_back(ts);

        this->latest_timestamp = std::max(this->latest_timestamp, ts);
    }

    this->edges->add_edges(sources.data(), targets.data(), timestamps.data(), new_edges.size());

    // Update node mappings
    this->node_mapping->update(this->edges, start_idx, this->edges->size());

    // Sort and merge new edges
    sort_and_merge_edges(start_idx);

    // Update timestamp groups after sorting
    this->edges->update_timestamp_groups();

    // Handle time window
    if (this->time_window > 0) {
        delete_old_edges();
    }

    // Rebuild edge indices
    this->node_index->rebuild(this->edges, this->node_mapping, this->is_directed);

    if (this->enable_weight_computation) {
        this->update_temporal_weights();
    }
}

template<GPUUsageMode GPUUsage>
HOST void TemporalGraphCPU<GPUUsage>::sort_and_merge_edges(const size_t start_idx) {
    if (start_idx >= this->edges->size()) return;

    // Sort new edges first
    typename ITemporalGraph<GPUUsage>::SizeVector indices(this->edges->size() - start_idx);
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = start_idx + i;
    }

    std::sort(indices.begin(), indices.end(),
        [this](const size_t i, const size_t j) {
            return this->edges->timestamps[i] < this->edges->timestamps[j];
    });

    // Apply permutation in-place using temporary vectors
    typename ITemporalGraph<GPUUsage>::IntVector sorted_sources(this->edges->size() - start_idx);
    typename ITemporalGraph<GPUUsage>::IntVector sorted_targets(this->edges->size() - start_idx);
    typename ITemporalGraph<GPUUsage>::Int64TVector sorted_timestamps(this->edges->size() - start_idx);

    for (size_t i = 0; i < indices.size(); i++) {
        const size_t idx = indices[i];
        sorted_sources[i] = this->edges->sources[idx];
        sorted_targets[i] = this->edges->targets[idx];
        sorted_timestamps[i] = this->edges->timestamps[idx];
    }

    // Copy back sorted edges
    for (size_t i = 0; i < indices.size(); i++) {
        this->edges->sources[start_idx + i] = sorted_sources[i];
        this->edges->targets[start_idx + i] = sorted_targets[i];
        this->edges->timestamps[start_idx + i] = sorted_timestamps[i];
    }

    // Merge with existing edges
    if (start_idx > 0) {
        // Create buffer vectors
        typename ITemporalGraph<GPUUsage>::IntVector merged_sources(this->edges->size());
        typename ITemporalGraph<GPUUsage>::IntVector merged_targets(this->edges->size());
        typename ITemporalGraph<GPUUsage>::Int64TVector merged_timestamps(this->edges->size());

        size_t i = 0;  // Index for existing edges
        size_t j = start_idx;  // Index for new edges
        size_t k = 0;  // Index for merged result

        // Merge while keeping arrays aligned
        while (i < start_idx && j < this->edges->size()) {
            if (this->edges->timestamps[i] <= this->edges->timestamps[j]) {
                merged_sources[k] = this->edges->sources[i];
                merged_targets[k] = this->edges->targets[i];
                merged_timestamps[k] = this->edges->timestamps[i];
                i++;
            } else {
                merged_sources[k] = this->edges->sources[j];
                merged_targets[k] = this->edges->targets[j];
                merged_timestamps[k] = this->edges->timestamps[j];
                j++;
            }
            k++;
        }

        // Copy remaining entries
        while (i < start_idx) {
            merged_sources[k] = this->edges->sources[i];
            merged_targets[k] = this->edges->targets[i];
            merged_timestamps[k] = this->edges->timestamps[i];
            i++;
            k++;
        }

        while (j < this->edges->size()) {
            merged_sources[k] = this->edges->sources[j];
            merged_targets[k] = this->edges->targets[j];
            merged_timestamps[k] = this->edges->timestamps[j];
            j++;
            k++;
        }

        // Copy merged data back to edges
        this->edges->sources = std::move(merged_sources);
        this->edges->targets = std::move(merged_targets);
        this->edges->timestamps = std::move(merged_timestamps);
    }
}

template<GPUUsageMode GPUUsage>
HOST void TemporalGraphCPU<GPUUsage>::delete_old_edges() {
    if (this->time_window <= 0 || this->edges->empty()) return;

    const int64_t cutoff_time = this->latest_timestamp - this->time_window;
    const auto it = std::upper_bound(this->edges->timestamps.begin(), this->edges->timestamps.end(), cutoff_time);
    if (it == this->edges->timestamps.begin()) return;

    const int delete_count = static_cast<int>(it - this->edges->timestamps.begin());
    const size_t remaining = this->edges->size() - delete_count;

    // Track which nodes still have edges
    typename ITemporalGraph<GPUUsage>::BoolVector has_edges(this->node_mapping->sparse_to_dense.size());

    if (remaining > 0) {
        std::move(this->edges->sources.begin() + delete_count, this->edges->sources.end(), this->edges->sources.begin());
        std::move(this->edges->targets.begin() + delete_count, this->edges->targets.end(), this->edges->targets.begin());
        std::move(this->edges->timestamps.begin() + delete_count, this->edges->timestamps.end(), this->edges->timestamps.begin());

        // Mark nodes that still have edges
        for (size_t i = 0; i < remaining; i++) {
            has_edges[this->edges->sources[i]] = true;
            has_edges[this->edges->targets[i]] = true;
        }
    }

    this->edges->resize(remaining);

    // Mark nodes with no edges as deleted
    for (size_t i = 0; i < has_edges.size(); i++) {
        if (!has_edges[i]) {
            this->node_mapping->mark_node_deleted(static_cast<int>(i));
        }
    }

    // Update all data structures after edge deletion
    this->edges->update_timestamp_groups();
    this->node_mapping->update(this->edges, 0, this->edges->size());
    this->node_index->rebuild(this->edges, this->node_mapping, this->is_directed);
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalGraphCPU<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const {
    if (this->edges->empty()) return 0;

    const auto it = std::lower_bound(this->edges->unique_timestamps.begin(), this->edges->unique_timestamps.end(), timestamp);
    return it - this->edges->unique_timestamps.begin();
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalGraphCPU<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const {
    if (this->edges->empty()) return 0;

    auto it = std::upper_bound(this->edges->unique_timestamps.begin(), this->edges->unique_timestamps.end(), timestamp);
    return this->edges->unique_timestamps.end() - it;
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalGraphCPU<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = this->node_mapping->to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->is_directed ? this->node_index->inbound_timestamp_group_offsets : this->node_index->outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->is_directed ? this->node_index->inbound_timestamp_group_indices : this->node_index->outbound_timestamp_group_indices;
    const auto& edge_indices = this->is_directed ? this->node_index->inbound_indices : this->node_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::lower_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](size_t group_pos, int64_t ts)
        {
            return this->edges->timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices.begin() + static_cast<int>(group_start), it);
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalGraphCPU<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = this->node_mapping->to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->node_index->outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->node_index->outbound_timestamp_group_indices;
    const auto& edge_indices = this->node_index->outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    const auto it = std::upper_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](int64_t ts, size_t group_pos)
        {
            return ts < this->edges->timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, timestamp_group_indices.begin() + static_cast<int>(group_end));
}

template<GPUUsageMode GPUUsage>
HOST Edge TemporalGraphCPU<GPUUsage>::get_edge_at_host(
    RandomPicker<GPUUsage>* picker,
    int64_t timestamp,
    const bool forward) const {

    if (this->edges->empty()) return Edge{-1, -1, -1};

    const size_t num_groups = this->edges->get_timestamp_group_count();
    if (num_groups == 0) return Edge{-1, -1, -1};

    size_t group_idx;
    if (timestamp != -1) {
        if (forward) {
            const size_t first_group = this->edges->find_group_after_timestamp(timestamp);
            const size_t available_groups = num_groups - first_group;
            if (available_groups == 0) return Edge{-1, -1, -1};

            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const size_t index = index_picker->pick_random_host(0, static_cast<int>(available_groups), false);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = first_group + index;
            }
            else {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_idx = weight_picker->pick_random_host(
                    this->edges->forward_cumulative_weights_exponential,
                    static_cast<int>(first_group),
                    static_cast<int>(num_groups));
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        } else {
            const size_t last_group = this->edges->find_group_before_timestamp(timestamp);
            if (last_group == static_cast<size_t>(-1)) return Edge{-1, -1, -1};

            const size_t available_groups = last_group + 1;
            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const size_t index = index_picker->pick_random_host(0, static_cast<int>(available_groups), true);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available_groups) return Edge{-1, -1, -1};
                group_idx = last_group - (available_groups - index - 1);
            }
            else {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_idx = weight_picker->pick_random_host(
                    this->edges->backward_cumulative_weights_exponential,
                    0,
                    static_cast<int>(last_group + 1));
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
            auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
            const size_t index = index_picker->pick_random_host(0, static_cast<int>(num_groups), !forward);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_idx = index;
        } else {
            auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
            if (forward) {
                group_idx = weight_picker->pick_random_host(
                    this->edges->forward_cumulative_weights_exponential,
                    0,
                    static_cast<int>(num_groups));
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
            else {
                group_idx = weight_picker->pick_random_host(
                    this->edges->backward_cumulative_weights_exponential,
                    0,
                    static_cast<int>(num_groups));
                if (group_idx == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get selected group's boundaries
    auto [group_start, group_end] = this->edges->get_timestamp_group_range(group_idx);
    if (group_start == group_end) {
        return Edge{-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_start + generate_random_number_bounded_by_host(static_cast<int>(group_end - group_start));
    return Edge {
        this->edges->sources[random_idx],
        this->edges->targets[random_idx],
        this->edges->timestamps[random_idx]
    };
}

template<GPUUsageMode GPUUsage>
HOST Edge TemporalGraphCPU<GPUUsage>::get_node_edge_at_host(
    const int node_id,
    RandomPicker<GPUUsage>* picker,
    const int64_t timestamp,
    const bool forward) const {

    const int dense_idx = this->node_mapping->to_dense(node_id);
    if (dense_idx < 0) return Edge{-1, -1, -1};

    // Get appropriate node indices based on direction and graph type
    const auto& timestamp_group_offsets = forward
        ? this->node_index->outbound_timestamp_group_offsets
        : (this->is_directed ? this->node_index->inbound_timestamp_group_offsets : this->node_index->outbound_timestamp_group_offsets);

    const auto& timestamp_group_indices = forward
        ? this->node_index->outbound_timestamp_group_indices
        : (this->is_directed ? this->node_index->inbound_timestamp_group_indices : this->node_index->outbound_timestamp_group_indices);

    const auto& edge_indices = forward
        ? this->node_index->outbound_indices
        : (this->is_directed ? this->node_index->inbound_indices : this->node_index->outbound_indices);

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return Edge{-1, -1, -1};

    size_t group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            auto it = std::upper_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](int64_t ts, size_t pos) {
                    return ts < this->edges->timestamps[edge_indices[pos]];
                });

            // Count available groups after timestamp
            const size_t available = std::distance(
                it,
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset));
            if (available == 0) return Edge{-1, -1, -1};

            const size_t start_pos = it - timestamp_group_indices.begin();
            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const size_t index = index_picker->pick_random_host(0, static_cast<int>(available), false);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = start_pos + index;
            }
            else
            {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_pos = weight_picker->pick_random_host(
                    this->node_index->outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(start_pos),
                    static_cast<int>(group_end_offset));
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        } else {
            // Find first group >= timestamp
            auto it = std::lower_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](size_t pos, int64_t ts) {
                    return this->edges->timestamps[edge_indices[pos]] < ts;
                });

            const size_t available = std::distance(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                it);
            if (available == 0) return Edge{-1, -1, -1};

            if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
                auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
                const size_t index = index_picker->pick_random_host(0, static_cast<int>(available), true);
                if (index == -1) return Edge{-1, -1, -1};

                if (index >= available) return Edge{-1, -1, -1};
                group_pos = (it - timestamp_group_indices.begin()) - 1 - (available - index - 1);
            }
            else
            {
                auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
                group_pos = weight_picker->pick_random_host(
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential
                        : this->node_index->outbound_backward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset), // start from node's first group
                    static_cast<int>(it - timestamp_group_indices.begin()) // up to and excluding first group >= timestamp
                );
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return Edge{-1, -1, -1};

        if (picker->get_picker_type() == INDEX_BASED_PICKER_TYPE) {
            auto* index_picker = static_cast<IndexBasedRandomPicker<GPUUsage>*>(picker);
            const size_t index = index_picker->pick_random_host(0, static_cast<int>(num_groups), !forward);
            if (index == -1) return Edge{-1, -1, -1};

            if (index >= num_groups) return Edge{-1, -1, -1};
            group_pos = forward
                ? group_start_offset + index
                : group_end_offset - 1 - (num_groups - index - 1);
        }
        else
        {
            auto* weight_picker = static_cast<WeightBasedRandomPicker<GPUUsage>*>(picker);
            if (forward)
            {
                group_pos = weight_picker->pick_random_host(
                    this->node_index->outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset));
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
            else
            {
                group_pos = weight_picker->pick_random_host(
                    this->is_directed
                        ? this->node_index->inbound_backward_cumulative_weights_exponential
                        : this->node_index->outbound_backward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset));
                if (group_pos == -1) return Edge{-1, -1, -1};
            }
        }
    }

    // Get edge range for selected group
    const size_t edge_start = timestamp_group_indices[group_pos];
    const size_t edge_end = (group_pos + 1 < group_end_offset)
        ? timestamp_group_indices[group_pos + 1]
        : (forward ? this->node_index->outbound_offsets[dense_idx + 1]
                  : (this->is_directed ? this->node_index->inbound_offsets[dense_idx + 1]
                                : this->node_index->outbound_offsets[dense_idx + 1]));

    // Validate range before random selection
    if (edge_start >= edge_end || edge_start >= edge_indices.size() || edge_end > edge_indices.size()) {
        return Edge{-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by_host(static_cast<int>(edge_end - edge_start))];

    return Edge {
        this->edges->sources[edge_idx],
        this->edges->targets[edge_idx],
        this->edges->timestamps[edge_idx]
    };
}

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCPU<GPUUsage>* TemporalGraphCPU<GPUUsage>::to_device_ptr() {
    throw std::runtime_error("Unsupported operation");
}

template class TemporalGraphCPU<GPUUsageMode::ON_CPU>;
