#include "edge_data.cuh"

#include <cmath>
#include <omp.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <atomic>
#include <cstddef>

#include "../utils/omp_utils.cuh"

/**
 * Common Functions
 */

HOST DEVICE size_t edge_data::size(const EdgeDataStore *edge_data) {
    return edge_data->timestamps_size;
}

HOST void edge_data::set_size(EdgeDataStore *edge_data, const size_t size) {
    edge_data->sources_size = size;
    edge_data->targets_size = size;
    edge_data->timestamps_size = size;
}

HOST void edge_data::add_edges(EdgeDataStore *edge_data, const int *sources, const int *targets, const int64_t *timestamps,
                           const size_t size) {
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
        const thrust::device_ptr<Edge> d_result(result.data);

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
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
    } else
    #endif
    {
        for (size_t i = 0; i < edge_data->timestamps_size; i++) {
            result.data[i] = Edge{edge_data->sources[i], edge_data->targets[i], edge_data->timestamps[i]};
        }
    }

    return result;
}

HOST DataBlock<int> edge_data::get_active_node_ids(const EdgeDataStore *edge_data) {
    size_t active_count = 0;

    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        const thrust::device_ptr<int> d_active_nodes(edge_data->active_node_ids);
        active_count = thrust::count(
            DEVICE_EXECUTION_POLICY,
            d_active_nodes,
            d_active_nodes + static_cast<long>(edge_data->active_node_ids_size),
            1
        );
        CUDA_KERNEL_CHECK("After thrust count in get_active_node_ids");
    } else
    #endif
    {
        for (int i = 0; i < edge_data->active_node_ids_size; i++) {
            if (edge_data->active_node_ids[i] == 1) {
                active_count++;
            }
        }
    }

    DataBlock<int> result(active_count, edge_data->use_gpu);
    if (active_count == 0) {
        return result;
    }

    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        const thrust::device_ptr<int> d_active_nodes(edge_data->active_node_ids);
        const thrust::device_ptr<int> d_result(result.data);

        thrust::copy_if(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(static_cast<int>(edge_data->active_node_ids_size)),
            d_active_nodes,
            d_result,
            [] __device__ (const int val) { return val == 1; }
        );
        CUDA_KERNEL_CHECK("After thrust copy_if in get_active_node_ids");
    } else
    #endif
    {
        size_t index = 0;
        for (int i = 0; i < edge_data->active_node_ids_size; i++) {
            if (edge_data->active_node_ids[i] == 1) {
                result.data[index++] = i;
            }
        }
    }

    return result;
}

HOST size_t edge_data::active_node_count(const EdgeDataStore *edge_data) {
    size_t count = 0;

    #ifdef HAS_CUDA
    if (edge_data->use_gpu) {
        const thrust::device_ptr<int> d_active_nodes(edge_data->active_node_ids);
        count = thrust::count(
            DEVICE_EXECUTION_POLICY,
            d_active_nodes,
            d_active_nodes + static_cast<long>(edge_data->active_node_ids_size),
            1
        );
        CUDA_KERNEL_CHECK("After thrust count in active_node_count");
    } else
    #endif
    {
        for (size_t i = 0; i < edge_data->active_node_ids_size; i++) {
            if (edge_data->active_node_ids[i] == 1) {
                count++;
            }
        }
    }

    return count;
}

HOST void edge_data::populate_active_nodes_std(EdgeDataStore *edge_data) {
    const size_t num_edges = size(edge_data);
    if (num_edges == 0) {
        return;
    }

    int max_node_id = -1;

    // Parallel reduction to find the max node id
    #pragma omp parallel for reduction(max:max_node_id)
    for (size_t i = 0; i < edge_data->sources_size; i++) {
        int src_node = edge_data->sources[i];
        int tgt_node = edge_data->targets[i];
        max_node_id = std::max({max_node_id, src_node, tgt_node});
    }

    allocate_memory(&edge_data->active_node_ids, max_node_id + 1, edge_data->use_gpu);
    edge_data->active_node_ids_size = max_node_id + 1;

    fill_memory(edge_data->active_node_ids, max_node_id + 1, 0, edge_data->use_gpu);

    // Parallel setting of active node flags
    #pragma omp parallel for
    for (size_t i = 0; i < size(edge_data); i++) {
        const int src = edge_data->sources[i];
        const int tgt = edge_data->targets[i];

        edge_data->active_node_ids[src] = 1;
        edge_data->active_node_ids[tgt] = 1;
    }
}

#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <atomic>
#include <cstddef>

HOST void edge_data::build_node_adjacency_csr_std(EdgeDataStore *edge_data) {
    const size_t n = active_node_count(edge_data);
    const size_t m = size(edge_data);

    // Allocate CSR arrays (host)
    resize_memory(&edge_data->node_adj_offsets,
                  edge_data->node_adj_offsets_size,
                  n + 1,
                  edge_data->use_gpu);
    edge_data->node_adj_offsets_size = n + 1;

    resize_memory(&edge_data->node_adj_neighbors,
                  edge_data->node_adj_neighbors_size,
                  2 * m,
                  edge_data->use_gpu);
    edge_data->node_adj_neighbors_size = 2 * m;

    // Handle empty graph
    if (n == 0 || m == 0) {
        // Parallel fill offsets with 0
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n + 1),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    edge_data->node_adj_offsets[i] = 0;
                }
            }
        );
        edge_data->node_adj_neighbors_size = 0;
        return;
    }

    const int* sources = edge_data->sources;
    const int* targets = edge_data->targets;

    // ---------------------------------------------------------------------
    // 1) Degree counting (parallel): degree[u]++, degree[v]++
    // ---------------------------------------------------------------------
    std::vector<std::atomic<size_t>> degree(n);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                degree[i].store(0, std::memory_order_relaxed);
            }
        }
    );

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, m),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const int u = sources[i];
                const int v = targets[i];
                // Assumes u and v are valid dense node IDs in [0, n).
                degree[static_cast<size_t>(u)].fetch_add(1, std::memory_order_relaxed);
                degree[static_cast<size_t>(v)].fetch_add(1, std::memory_order_relaxed);
            }
        }
    );

    // ---------------------------------------------------------------------
    // 2) Offsets via parallel prefix sum (TBB parallel_scan)
    //    offsets[0] = 0, offsets[i+1] = offsets[i] + degree[i]
    // ---------------------------------------------------------------------
    edge_data->node_adj_offsets[0] = 0; // O(1) scalar assignment

    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, n),
        static_cast<size_t>(0),
        [&](const tbb::blocked_range<size_t>& r, size_t sum, bool is_final) -> size_t {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const size_t d = degree[i].load(std::memory_order_relaxed);
                if (is_final) {
                    edge_data->node_adj_offsets[i + 1] = sum + d;
                }
                sum += d;
            }
            return sum;
        },
        std::plus<size_t>()
    );

    // ---------------------------------------------------------------------
    // 3) Cursor init from offsets (parallel)
    // ---------------------------------------------------------------------
    std::vector<std::atomic<size_t>> cursor(n);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                cursor[i].store(edge_data->node_adj_offsets[i], std::memory_order_relaxed);
            }
        }
    );

    // ---------------------------------------------------------------------
    // 4) Fill neighbors using cursor atomics (parallel)
    // ---------------------------------------------------------------------
    int* neighbors = edge_data->node_adj_neighbors;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, m),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const int u = sources[i];
                const int v = targets[i];

                const size_t u_pos =
                    cursor[static_cast<size_t>(u)].fetch_add(1, std::memory_order_relaxed);
                const size_t v_pos =
                    cursor[static_cast<size_t>(v)].fetch_add(1, std::memory_order_relaxed);

                neighbors[u_pos] = v;
                neighbors[v_pos] = u;
            }
        }
    );

    // ---------------------------------------------------------------------
    // 5) Sort adjacency list per node (parallel over nodes + tbb::parallel_sort)
    // ---------------------------------------------------------------------
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t node = r.begin(); node != r.end(); ++node) {
                const size_t start = edge_data->node_adj_offsets[node];
                const size_t end   = edge_data->node_adj_offsets[node + 1];
                if (end > start + 1) {
                    tbb::parallel_sort(neighbors + start, neighbors + end);
                }
            }
        }
    );
}

HOST void edge_data::update_timestamp_groups_std(EdgeDataStore* edge_data) {
    if (edge_data->timestamps_size == 0) {
        clear_memory(&edge_data->timestamp_group_offsets, edge_data->use_gpu);
        edge_data->timestamp_group_offsets_size = 0;

        clear_memory(&edge_data->unique_timestamps, edge_data->use_gpu);
        edge_data->unique_timestamps_size = 0;
        return;
    }

    const size_t n = edge_data->timestamps_size;

    // Step 1: Flag where timestamps change
    std::vector<int> flags(n, 0);
    flags[0] = 1;

    #pragma omp parallel for
    for (size_t i = 1; i < n; ++i) {
        flags[i] = (edge_data->timestamps[i] != edge_data->timestamps[i - 1]) ? 1 : 0;
    }

    // Step 2: Compute prefix sum into raw buffer (exclusive scan)
    std::vector<int> prefix_sum(n);
    parallel_prefix_sum(flags.data(), prefix_sum.data(), n);

    const int num_groups = prefix_sum[n - 1] + flags[n - 1];

    // Step 3: Resize output arrays
    resize_memory(&edge_data->timestamp_group_offsets,
                  edge_data->timestamp_group_offsets_size,
                  num_groups + 1,
                  edge_data->use_gpu);
    edge_data->timestamp_group_offsets_size = num_groups + 1;

    resize_memory(&edge_data->unique_timestamps,
                  edge_data->unique_timestamps_size,
                  num_groups,
                  edge_data->use_gpu);
    edge_data->unique_timestamps_size = num_groups;

    // Step 4: Write group offsets and unique timestamps
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if (flags[i]) {
            int idx = prefix_sum[i];
            edge_data->timestamp_group_offsets[idx] = i;
            edge_data->unique_timestamps[idx] = edge_data->timestamps[i];
        }
    }

    // Step 5: Final group offset (end marker)
    edge_data->timestamp_group_offsets[num_groups] = n;

    // Step 6: Activate nodes
    populate_active_nodes_std(edge_data);
}

HOST void edge_data::update_temporal_weights_std(EdgeDataStore* edge_data, const double timescale_bound) {
    if (edge_data->timestamps_size == 0) {
        clear_memory(&edge_data->forward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->forward_cumulative_weights_exponential_size = 0;

        clear_memory(&edge_data->backward_cumulative_weights_exponential, edge_data->use_gpu);
        edge_data->backward_cumulative_weights_exponential_size = 0;
        return;
    }

    const int64_t min_timestamp = edge_data->timestamps[0];
    const int64_t max_timestamp = edge_data->timestamps[edge_data->timestamps_size - 1];
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count(edge_data);

    // Resize output arrays
    resize_memory(&edge_data->forward_cumulative_weights_exponential,
                  edge_data->forward_cumulative_weights_exponential_size,
                  num_groups,
                  edge_data->use_gpu);
    edge_data->forward_cumulative_weights_exponential_size = num_groups;

    resize_memory(&edge_data->backward_cumulative_weights_exponential,
                  edge_data->backward_cumulative_weights_exponential_size,
                  num_groups,
                  edge_data->use_gpu);
    edge_data->backward_cumulative_weights_exponential_size = num_groups;

    auto* forward = edge_data->forward_cumulative_weights_exponential;
    auto* backward = edge_data->backward_cumulative_weights_exponential;
    const auto* timestamps = edge_data->timestamps;
    const auto* offsets = edge_data->timestamp_group_offsets;

    double forward_sum = 0.0, backward_sum = 0.0;

    // Step 1: Compute unnormalized weights and sums
    #pragma omp parallel for reduction(+:forward_sum, backward_sum)
    for (size_t group = 0; group < num_groups; ++group) {
        const size_t start = offsets[group];
        const int64_t ts = timestamps[start];

        const double t_fwd = static_cast<double>(max_timestamp - ts);
        const double t_bwd = static_cast<double>(ts - min_timestamp);

        const double fwd_scaled = (timescale_bound > 0) ? t_fwd * time_scale : t_fwd;
        const double bwd_scaled = (timescale_bound > 0) ? t_bwd * time_scale : t_bwd;

        const double f_weight = std::exp(fwd_scaled);
        const double b_weight = std::exp(bwd_scaled);

        forward[group] = f_weight;
        backward[group] = b_weight;

        forward_sum += f_weight;
        backward_sum += b_weight;
    }

    // Step 2: Normalize
    #pragma omp parallel for
    for (size_t group = 0; group < num_groups; ++group) {
        forward[group] /= forward_sum;
        backward[group] /= backward_sum;
    }

    // Step 3: Inclusive scan
    parallel_inclusive_scan(forward, num_groups);
    parallel_inclusive_scan(backward, num_groups);
}

#ifdef HAS_CUDA

HOST void edge_data::populate_active_nodes_cuda(EdgeDataStore *edge_data) {
    const size_t num_edges = size(edge_data);
    if (num_edges == 0) {
        return;
    }

    const thrust::device_ptr<int> d_sources(edge_data->sources);
    const thrust::device_ptr<int> d_targets(edge_data->targets);

    const int max_source = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(num_edges),
        0,
        thrust::maximum<int>()
    );

    const int max_target = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_targets,
        d_targets + static_cast<long>(num_edges),
        0,
        thrust::maximum<int>()
    );

    int max_node_id = std::max(max_source, max_target);

    allocate_memory(&edge_data->active_node_ids, max_node_id + 1, edge_data->use_gpu);
    edge_data->active_node_ids_size = max_node_id + 1;

    fill_memory(edge_data->active_node_ids, max_node_id + 1, 0, edge_data->use_gpu);

    thrust::device_ptr<int> d_active_nodes(edge_data->active_node_ids);

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(num_edges),
        [d_active_nodes] __device__ (const int source_id) {
            d_active_nodes[source_id] = 1;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each sources in populate_active_nodes_cuda");

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        d_targets,
        d_targets + static_cast<long>(num_edges),
        [d_active_nodes] __device__ (int target_id) {
            d_active_nodes[target_id] = 1;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each targets in populate_active_nodes_cuda");
}

HOST void edge_data::build_node_adjacency_csr_cuda(EdgeDataStore *edge_data) {
    const size_t n = active_node_count(edge_data);
    const size_t m = size(edge_data);

    // Allocate CSR arrays (device)
    resize_memory(&edge_data->node_adj_offsets,
                  edge_data->node_adj_offsets_size,
                  n + 1,
                  edge_data->use_gpu);
    edge_data->node_adj_offsets_size = n + 1;

    resize_memory(&edge_data->node_adj_neighbors,
                  edge_data->node_adj_neighbors_size,
                  2 * m,
                  edge_data->use_gpu);
    edge_data->node_adj_neighbors_size = 2 * m;

    // Handle empty graph
    if (n == 0 || m == 0) {
        // offsets[0] = 0 (and n==0 => size is 1)
        thrust::fill_n(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(edge_data->node_adj_offsets),
            static_cast<long>(n + 1),
            static_cast<size_t>(0)
        );
        CUDA_KERNEL_CHECK("After thrust fill_n zero offsets in build_node_adjacency_csr_cuda");
        return;
    }

    // Device pointers to input edges and output CSR arrays
    const auto d_sources   = thrust::device_pointer_cast(edge_data->sources);
    const auto d_targets   = thrust::device_pointer_cast(edge_data->targets);
    const auto d_offsets   = thrust::device_pointer_cast(edge_data->node_adj_offsets);
    const auto d_neighbors = thrust::device_pointer_cast(edge_data->node_adj_neighbors);

    // ---------------------------------------------------------------------
    // 1) Degree counting (device): degree[u]++, degree[v]++
    //    Use unsigned int for atomicAdd compatibility.
    // ---------------------------------------------------------------------
    thrust::device_vector<unsigned int> degree(n, 0u);

    unsigned int* d_degree_raw = thrust::raw_pointer_cast(degree.data());

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(m),
        [d_sources, d_targets, d_degree_raw] __device__ (const size_t i) {
            const int u = d_sources[static_cast<long>(i)];
            const int v = d_targets[static_cast<long>(i)];
            // Assumes u and v are valid dense node IDs in [0, n).
            atomicAdd(d_degree_raw + u, 1u);
            atomicAdd(d_degree_raw + v, 1u);
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each degree counting in build_node_adjacency_csr_cuda");

    // ---------------------------------------------------------------------
    // 2) Offsets: exclusive_scan(degree) -> offsets[0..n)
    //    Then explicitly set offsets[n] = 2*m on device (no host memcpy).
    // ---------------------------------------------------------------------
    thrust::exclusive_scan(
        DEVICE_EXECUTION_POLICY,
        degree.begin(),
        degree.end(),
        d_offsets,
        static_cast<size_t>(0)
    );
    CUDA_KERNEL_CHECK("After thrust exclusive_scan offsets in build_node_adjacency_csr_cuda");

    // offsets[n] = 2*m (device write)
    thrust::fill_n(
        DEVICE_EXECUTION_POLICY,
        d_offsets + static_cast<long>(n),
        1,
        static_cast<size_t>(2 * m)
    );
    CUDA_KERNEL_CHECK("After writing offsets[n] in build_node_adjacency_csr_cuda");

    // ---------------------------------------------------------------------
    // 3) Cursor: initialize from offsets[0..n)
    //    cursor[u] starts at offsets[u], then atomicAdd during fill.
    // ---------------------------------------------------------------------
    thrust::device_vector<unsigned int> cursor(n, 0u);

    // Copy offsets[0..n) into cursor as unsigned int.
    // Note: This assumes offsets fit into 32-bit (true if 2m fits in uint32).
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_offsets,
        d_offsets + static_cast<long>(n),
        cursor.begin(),
        [] __device__ (const size_t x) { return static_cast<unsigned int>(x); }
    );
    CUDA_KERNEL_CHECK("After cursor init transform in build_node_adjacency_csr_cuda");

    unsigned int* d_cursor_raw = thrust::raw_pointer_cast(cursor.data());

    // ---------------------------------------------------------------------
    // 4) Fill neighbors using cursor atomics
    // ---------------------------------------------------------------------
    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(m),
        [d_sources, d_targets, d_cursor_raw, d_neighbors] __device__ (const size_t i) {
            const int u = d_sources[static_cast<long>(i)];
            const int v = d_targets[static_cast<long>(i)];

            const unsigned int u_pos = atomicAdd(d_cursor_raw + u, 1u);
            const unsigned int v_pos = atomicAdd(d_cursor_raw + v, 1u);

            d_neighbors[static_cast<long>(static_cast<size_t>(u_pos))] = v;
            d_neighbors[static_cast<long>(static_cast<size_t>(v_pos))] = u;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each neighbor fill in build_node_adjacency_csr_cuda");

    // ---------------------------------------------------------------------
    // 5) GPU segmented sort (thrust-only, no loops)
    //    Build key array node_ids[i] = owning node of neighbors[i],
    //    then sort (node_id, neighbor) pairs lexicographically.
    // ---------------------------------------------------------------------
    const size_t nnz = static_cast<size_t>(2 * m);

    thrust::device_vector<int> node_ids(nnz);

    // node_ids[pos] = upper_bound(offsets, pos) - 1
    // Search range must be offsets[0..n] inclusive => n+1 elements.
    thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        d_offsets,
        d_offsets + static_cast<long>(n + 1),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(nnz),
        node_ids.begin()
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in build_node_adjacency_csr_cuda");

    // upper_bound returns (node+1); subtract 1 to get node id
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        node_ids.begin(),
        node_ids.end(),
        node_ids.begin(),
        [] __device__ (int x) { return x - 1; }
    );
    CUDA_KERNEL_CHECK("After node_ids subtract-one transform in build_node_adjacency_csr_cuda");

    // Sort pairs (node_id, neighbor)
    auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            node_ids.begin(),
            d_neighbors
        )
    );

    thrust::sort(
        DEVICE_EXECUTION_POLICY,
        zipped_begin,
        zipped_begin + static_cast<long>(nnz)
    );
    CUDA_KERNEL_CHECK("After thrust segmented sort (key,neighbor) in build_node_adjacency_csr_cuda");
}


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
    int *d_flags = nullptr;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_flags, n * sizeof(int)));

    // Wrap raw pointers with thrust device pointers for algorithm use
    const thrust::device_ptr<int64_t> d_timestamps(edge_data->timestamps);
    const thrust::device_ptr<int> d_flags_ptr(d_flags);

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
    const thrust::device_ptr<size_t> d_group_offsets(edge_data->timestamp_group_offsets);
    const thrust::device_ptr<int64_t> d_unique_timestamps(edge_data->unique_timestamps);

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

    populate_active_nodes_cuda(edge_data);
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
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(&min_timestamp, edge_data->timestamps, sizeof(int64_t), cudaMemcpyDeviceToHost))
    ;
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&max_timestamp, edge_data->timestamps + (edge_data->timestamps_size - 1), sizeof(int64_t),
            cudaMemcpyDeviceToHost));

    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

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
    double *d_forward_weights = nullptr;
    double *d_backward_weights = nullptr;
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

            const double forward_scaled = timescale_bound > 0 ? time_diff_forward * time_scale : time_diff_forward;
            const double backward_scaled = timescale_bound > 0
                                               ? time_diff_backward * time_scale
                                               : time_diff_backward;

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

HOST EdgeDataStore* edge_data::to_device_ptr(const EdgeDataStore *edge_data) {
    // Create a new EdgeData object on the device
    EdgeDataStore *device_edge_data;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_edge_data, sizeof(EdgeDataStore)));

    // Create a temporary copy to modify for device pointers
    EdgeDataStore temp_edge_data = *edge_data;
    temp_edge_data.owns_data = false;

    // If already using GPU, just copy the struct with its pointers
    if (!edge_data->use_gpu) {
        temp_edge_data.owns_data = true;

        // Copy each array to device if it exists
        if (edge_data->sources) {
            int *d_sources;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_sources, edge_data->sources_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_sources, edge_data->sources, edge_data->sources_size * sizeof(int),
                    cudaMemcpyHostToDevice));
            temp_edge_data.sources = d_sources;
        }

        if (edge_data->targets) {
            int *d_targets;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_targets, edge_data->targets_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_targets, edge_data->targets, edge_data->targets_size * sizeof(int),
                    cudaMemcpyHostToDevice));
            temp_edge_data.targets = d_targets;
        }

        if (edge_data->timestamps) {
            int64_t *d_timestamps;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_timestamps, edge_data->timestamps_size * sizeof(int64_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_timestamps, edge_data->timestamps, edge_data->timestamps_size * sizeof(int64_t),
                    cudaMemcpyHostToDevice));
            temp_edge_data.timestamps = d_timestamps;
        }

        if (edge_data->active_node_ids) {
            int *d_active_node_ids;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_active_node_ids, edge_data->active_node_ids_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_active_node_ids, edge_data->active_node_ids, edge_data->active_node_ids_size * sizeof(
                    int), cudaMemcpyHostToDevice));
            temp_edge_data.active_node_ids = d_active_node_ids;
        }

        if (edge_data->node_adj_offsets) {
            size_t *d_node_adj_offsets;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_node_adj_offsets, edge_data->node_adj_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_node_adj_offsets,
                           edge_data->node_adj_offsets,
                           edge_data->node_adj_offsets_size * sizeof(size_t),
                           cudaMemcpyHostToDevice));
            temp_edge_data.node_adj_offsets = d_node_adj_offsets;
        }

        if (edge_data->node_adj_neighbors) {
            int *d_node_adj_neighbors;
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_node_adj_neighbors, edge_data->node_adj_neighbors_size * sizeof(int)));
            CUDA_CHECK_AND_CLEAR(
                cudaMemcpy(d_node_adj_neighbors,
                           edge_data->node_adj_neighbors,
                           edge_data->node_adj_neighbors_size * sizeof(int),
                           cudaMemcpyHostToDevice));
            temp_edge_data.node_adj_neighbors = d_node_adj_neighbors;
        }

        if (edge_data->timestamp_group_offsets) {
            size_t *d_timestamp_group_offsets;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_timestamp_group_offsets, edge_data->timestamp_group_offsets_size * sizeof(size_t)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_timestamp_group_offsets, edge_data->timestamp_group_offsets,
                edge_data->timestamp_group_offsets_size * sizeof(size_t), cudaMemcpyHostToDevice));
            temp_edge_data.timestamp_group_offsets = d_timestamp_group_offsets;
        }

        if (edge_data->unique_timestamps) {
            int64_t *d_unique_timestamps;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_unique_timestamps, edge_data->unique_timestamps_size * sizeof(int64_t)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_unique_timestamps, edge_data->unique_timestamps,
                edge_data->unique_timestamps_size * sizeof(int64_t), cudaMemcpyHostToDevice));
            temp_edge_data.unique_timestamps = d_unique_timestamps;
        }

        if (edge_data->forward_cumulative_weights_exponential) {
            double *d_forward_weights;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_forward_weights, edge_data->forward_cumulative_weights_exponential_size * sizeof(
                    double)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_forward_weights, edge_data->forward_cumulative_weights_exponential,
                edge_data->forward_cumulative_weights_exponential_size * sizeof(double), cudaMemcpyHostToDevice));
            temp_edge_data.forward_cumulative_weights_exponential = d_forward_weights;
        }

        if (edge_data->backward_cumulative_weights_exponential) {
            double *d_backward_weights;
            CUDA_CHECK_AND_CLEAR(
                cudaMalloc(&d_backward_weights, edge_data->backward_cumulative_weights_exponential_size * sizeof(
                    double)));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_backward_weights, edge_data->backward_cumulative_weights_exponential,
                edge_data->backward_cumulative_weights_exponential_size * sizeof(double), cudaMemcpyHostToDevice));
            temp_edge_data.backward_cumulative_weights_exponential = d_backward_weights;
        }

        // Make sure use_gpu is set to true
        temp_edge_data.use_gpu = true;
    }

    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(device_edge_data, &temp_edge_data, sizeof(EdgeDataStore), cudaMemcpyHostToDevice));

    temp_edge_data.owns_data = false;

    return device_edge_data;
}

#endif

HOST size_t edge_data::get_memory_used(EdgeDataStore* edge_data) {
    size_t total_memory = 0;

    // Basic edge data arrays
    total_memory += edge_data->sources_size * sizeof(int);
    total_memory += edge_data->targets_size * sizeof(int);
    total_memory += edge_data->timestamps_size * sizeof(int64_t);

    // Active nodes array
    total_memory += edge_data->active_node_ids_size * sizeof(int);

    // Node adjacency CSR arrays
    total_memory += edge_data->node_adj_offsets_size * sizeof(size_t);
    total_memory += edge_data->node_adj_neighbors_size * sizeof(int);

    // Timestamp grouping arrays
    total_memory += edge_data->timestamp_group_offsets_size * sizeof(size_t);
    total_memory += edge_data->unique_timestamps_size * sizeof(int64_t);

    // Weight computation arrays (if allocated)
    total_memory += edge_data->forward_cumulative_weights_exponential_size * sizeof(double);
    total_memory += edge_data->backward_cumulative_weights_exponential_size * sizeof(double);

    return total_memory;
}
