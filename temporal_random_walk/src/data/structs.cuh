#ifndef STRUCTS_H
#define STRUCTS_H

#include "../cuda_common/macros.cuh"
#include <cstddef>
#include <cstdint>
#include "../cuda_common/types.cuh"

#include "enums.h"


struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(size_t f, size_t t) : from(f), to(t) {}

    HOST DEVICE SizeRange& operator=(const SizeRange& other)
    {
        if (this != &other)
        {
            from = other.from;
            to = other.to;
        }
        return *this;
    }
};

template <typename T, typename U>
struct IndexValuePair {
    T index;
    U value;

    HOST DEVICE IndexValuePair() : index(), value() {}
    HOST DEVICE IndexValuePair(const T& idx, const U& val) : index(idx), value(val) {}

    // Optional: Add assignment operator for full compatibility
    HOST DEVICE IndexValuePair& operator=(const IndexValuePair& other) {
        if (this != &other) {
            index = other.index;
            value = other.value;
        }
        return *this;
    }
};

struct Edge {
    int u;
    int i;
    int64_t ts;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1) {}

    HOST DEVICE explicit Edge(int u, int i, int64_t ts) : u(u), i(i), ts(ts) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
        }
        return *this;
    }
};

struct NodeWithTime {
    int node;
    int64_t timestamp;

    HOST DEVICE NodeWithTime(): node(-1), timestamp(-1) {}

    HOST DEVICE NodeWithTime(int node, int64_t timestamp): node(node), timestamp(timestamp) {}

    HOST DEVICE NodeWithTime& operator=(const NodeWithTime& other)
    {
        if (this != &other)
        {
            node = other.node;
            timestamp = other.timestamp;
        }

        return *this;
    }
};

struct EdgeWithEndpointType {
    long edge_id;
    bool is_source;

    HOST DEVICE EdgeWithEndpointType(): edge_id(-1), is_source(true) {}

    HOST DEVICE EdgeWithEndpointType(long edge_id, bool is_source): edge_id(edge_id), is_source(is_source) {}

    HOST DEVICE EdgeWithEndpointType& operator=(const EdgeWithEndpointType& other)
    {
        if (this != &other)
        {
            edge_id = other.edge_id;
            is_source = other.is_source;
        }

        return *this;
    }
};

template<GPUUsageMode GPUUsage>
struct WalkSet
{
    size_t num_walks;
    size_t max_len;

    typename SelectVectorType<int, GPUUsage>::type nodes;
    typename SelectVectorType<int64_t, GPUUsage>::type timestamps;
    typename SelectVectorType<size_t, GPUUsage>::type walk_lens;

    // CPU vectors for host access
    std::vector<int> nodes_cpu;
    std::vector<int64_t> timestamps_cpu;
    std::vector<size_t> walk_lens_cpu;

    int* nodes_ptr = nullptr;
    int64_t* timestamps_ptr = nullptr;
    size_t* walk_lens_ptr = nullptr;

    size_t total_len;

    HOST WalkSet(): num_walks(0), max_len(0), nodes({}), timestamps({}), walk_lens({}), total_len(0) {}

    HOST WalkSet(size_t num_walks, const size_t max_len)
        : num_walks(num_walks), max_len(max_len), nodes({}), timestamps({}), walk_lens({})
    {
        total_len = num_walks * max_len;

        nodes.resize(total_len);
        timestamps.resize(total_len);
        walk_lens.resize(num_walks);

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            nodes_ptr = thrust::raw_pointer_cast(nodes.data());
            timestamps_ptr = thrust::raw_pointer_cast(timestamps.data());
            walk_lens_ptr = thrust::raw_pointer_cast(walk_lens.data());
        }
        else
        #endif
        {
            nodes_ptr = nodes.data();
            timestamps_ptr = timestamps.data();
            walk_lens_ptr = walk_lens.data();
        }
    }

    // Copy constructor
    HOST WalkSet(const WalkSet& other)
        : num_walks(other.num_walks), max_len(other.max_len),
          nodes(other.nodes), timestamps(other.timestamps), walk_lens(other.walk_lens),
          nodes_cpu(other.nodes_cpu), timestamps_cpu(other.timestamps_cpu), walk_lens_cpu(other.walk_lens_cpu),
          total_len(other.total_len)
    {
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            nodes_ptr = thrust::raw_pointer_cast(nodes.data());
            timestamps_ptr = thrust::raw_pointer_cast(timestamps.data());
            walk_lens_ptr = thrust::raw_pointer_cast(walk_lens.data());
        }
        else
        #endif
        {
            nodes_ptr = nodes.data();
            timestamps_ptr = timestamps.data();
            walk_lens_ptr = walk_lens.data();
        }
    }

    // Move constructor
    HOST WalkSet(WalkSet&& other) noexcept
        : num_walks(other.num_walks), max_len(other.max_len),
          nodes(std::move(other.nodes)), timestamps(std::move(other.timestamps)),
          walk_lens(std::move(other.walk_lens)),
          nodes_cpu(std::move(other.nodes_cpu)), timestamps_cpu(std::move(other.timestamps_cpu)),
          walk_lens_cpu(std::move(other.walk_lens_cpu)),
          nodes_ptr(other.nodes_ptr), timestamps_ptr(other.timestamps_ptr),
          walk_lens_ptr(other.walk_lens_ptr),
          total_len(other.total_len)
    {
        // Reset the source object to prevent double-free issues
        other.num_walks = 0;
        other.max_len = 0;
        other.total_len = 0;
        other.nodes_ptr = nullptr;
        other.timestamps_ptr = nullptr;
        other.walk_lens_ptr = nullptr;
    }

    // Copy assignment operator
    HOST WalkSet& operator=(const WalkSet& other)
    {
        if (this != &other) {
            num_walks = other.num_walks;
            max_len = other.max_len;
            total_len = other.total_len;

            nodes = other.nodes;
            timestamps = other.timestamps;
            walk_lens = other.walk_lens;

            nodes_cpu = other.nodes_cpu;
            timestamps_cpu = other.timestamps_cpu;
            walk_lens_cpu = other.walk_lens_cpu;

            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                nodes_ptr = thrust::raw_pointer_cast(nodes.data());
                timestamps_ptr = thrust::raw_pointer_cast(timestamps.data());
                walk_lens_ptr = thrust::raw_pointer_cast(walk_lens.data());
            }
            else
            #endif
            {
                nodes_ptr = nodes.data();
                timestamps_ptr = timestamps.data();
                walk_lens_ptr = walk_lens.data();
            }
        }
        return *this;
    }

    // Move assignment operator
    HOST WalkSet& operator=(WalkSet&& other) noexcept
    {
        if (this != &other) {
            num_walks = other.num_walks;
            max_len = other.max_len;
            total_len = other.total_len;

            nodes = std::move(other.nodes);
            timestamps = std::move(other.timestamps);
            walk_lens = std::move(other.walk_lens);

            nodes_cpu = std::move(other.nodes_cpu);
            timestamps_cpu = std::move(other.timestamps_cpu);
            walk_lens_cpu = std::move(other.walk_lens_cpu);

            nodes_ptr = other.nodes_ptr;
            timestamps_ptr = other.timestamps_ptr;
            walk_lens_ptr = other.walk_lens_ptr;

            // Reset the source object
            other.num_walks = 0;
            other.max_len = 0;
            other.total_len = 0;
            other.nodes_ptr = nullptr;
            other.timestamps_ptr = nullptr;
            other.walk_lens_ptr = nullptr;
        }
        return *this;
    }

    HOST WalkSet* to_device_ptr() {
        WalkSet* device_walk_set;
        cudaMalloc(&device_walk_set, sizeof(WalkSet));
        cudaMemcpy(device_walk_set, this, sizeof(WalkSet), cudaMemcpyHostToDevice);
        return device_walk_set;
    }

    // Method to copy data from a device WalkSet to this host WalkSet
    HOST void copy_from_device(const WalkSet* d_walk_set) {
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // Allocate a host-side WalkSet
            WalkSet<GPUUsage> h_walk_set;

            // Copy the entire struct from device to host
            cudaMemcpy(&h_walk_set, d_walk_set, sizeof(WalkSet<GPUUsage>), cudaMemcpyDeviceToHost);

            // Ensure host vectors are properly sized
            h_walk_set.nodes_cpu.resize(h_walk_set.total_len);
            h_walk_set.timestamps_cpu.resize(h_walk_set.total_len);
            h_walk_set.walk_lens_cpu.resize(h_walk_set.num_walks);

            // Copy data from device memory to CPU vectors
            cudaMemcpy(
                h_walk_set.nodes_cpu.data(),
                h_walk_set.nodes_ptr,
                sizeof(int) * h_walk_set.total_len,
                cudaMemcpyDeviceToHost);

            cudaMemcpy(
                h_walk_set.timestamps_cpu.data(),
                h_walk_set.timestamps_ptr,
                sizeof(int64_t) * h_walk_set.total_len,
                cudaMemcpyDeviceToHost);

            cudaMemcpy(
                h_walk_set.walk_lens_cpu.data(),
                h_walk_set.walk_lens_ptr,
                sizeof(size_t) * h_walk_set.num_walks,
                cudaMemcpyDeviceToHost);
        }
        #endif
    }

    HOST DEVICE void add_hop(int walk_number, int node, int64_t timestamp)
    {
        const size_t offset = walk_number * max_len + walk_lens_ptr[walk_number];
        nodes_ptr[offset] = node;
        timestamps_ptr[offset] = timestamp;
        walk_lens_ptr[walk_number] += 1;
    }

    HOST size_t get_walk_len(int walk_number) {
        #ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_GPU) {
            return walk_lens_cpu[walk_number];
        }
        else
        #endif
        {
            return walk_lens_ptr[walk_number];
        }
    }

    DEVICE size_t get_walk_len_device(int walk_number) {
        return walk_lens_ptr[walk_number];
    }

    HOST NodeWithTime get_walk_hop(int walk_number, int hop_number) {
        #ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_GPU) {
            size_t walk_length = walk_lens_cpu[walk_number];
            if (hop_number < 0 || hop_number >= walk_length) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            size_t offset = walk_number * max_len + hop_number;
            return NodeWithTime{nodes_cpu[offset], timestamps_cpu[offset]};
        }
        else
        #endif
        {
            size_t walk_length = walk_lens_ptr[walk_number];
            if (hop_number < 0 || hop_number >= walk_length) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            size_t offset = walk_number * max_len + hop_number;
            return NodeWithTime{nodes_ptr[offset], timestamps_ptr[offset]};
        }
    }

    HOST DEVICE void reverse_walk(int walk_number)
    {
        const size_t walk_length = walk_lens_ptr[walk_number];
        if (walk_length <= 1) return; // No need to reverse if walk is empty or has one hop

        const size_t start = walk_number * max_len;
        const size_t end = start + walk_length - 1;

        for (size_t i = 0; i < walk_length / 2; ++i) {
            // Swap nodes
            int temp_node = nodes_ptr[start + i];
            nodes_ptr[start + i] = nodes_ptr[end - i];
            nodes_ptr[end - i] = temp_node;

            // Swap timestamps
            int64_t temp_time = timestamps_ptr[start + i];
            timestamps_ptr[start + i] = timestamps_ptr[end - i];
            timestamps_ptr[end - i] = temp_time;
        }
    }
};

template <typename T, GPUUsageMode GPUUsage>
struct DividedVector {
    typename SelectVectorType<IndexValuePair<int, T>, GPUUsage>::type elements;
    typename SelectVectorType<size_t, GPUUsage>::type group_offsets;
    size_t num_groups;

    IndexValuePair<int, T>* elements_ptr = nullptr;
    size_t* group_offsets_ptr = nullptr;
    size_t total_len;

    // Constructor - divides input vector into n groups
    HOST DividedVector(const typename SelectVectorType<T, GPUUsage>::type& input, int n)
        : num_groups(n)
    {
        const int total_size = static_cast<int>(input.size());
        const int base_size = total_size / n;
        const int remainder = total_size % n;

        // Reserve space for group offsets (n+1 offsets for n groups)
        group_offsets.resize(n + 1);

        // Calculate and store group offsets
        size_t current_offset = 0;
        group_offsets[0] = current_offset;

        for (int i = 0; i < n; i++) {
            const int group_size = base_size + (i < remainder ? 1 : 0);
            current_offset += group_size;
            group_offsets[i + 1] = current_offset;
        }

        // Allocate space for all elements
        elements.resize(total_size);

        // Populate the elements array
        for (int i = 0; i < n; i++) {
            const size_t start_idx = group_offsets[i];
            const size_t end_idx = group_offsets[i + 1];

            for (size_t j = start_idx; j < end_idx; ++j) {
                elements[j] = IndexValuePair<int, T>(j, input[j]);
            }
        }

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            elements_ptr = thrust::raw_pointer_cast(elements.data());
            group_offsets_ptr = thrust::raw_pointer_cast(group_offsets.data());
        }
        else
        #endif
        {
            elements_ptr = elements.data();
            group_offsets_ptr = group_offsets.data();
        }
        total_len = elements.size();
    }

    // Get begin iterator for a specific group
    HOST DEVICE IndexValuePair<int, T>* group_begin(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }
        return elements_ptr + group_offsets_ptr[group_idx];
    }

    // Get end iterator for a specific group
    HOST DEVICE IndexValuePair<int, T>* group_end(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }
        return elements_ptr + group_offsets_ptr[group_idx + 1];
    }

    // Get size of a specific group
    HOST DEVICE [[nodiscard]] size_t group_size(size_t group_idx) const {
        if (group_idx >= num_groups) {
            return 0;
        }
        return group_offsets_ptr[group_idx + 1] - group_offsets_ptr[group_idx];
    }

    // Helper class for iterating over a group
    struct GroupIterator {
        DividedVector& divided_vector;
        size_t group_idx;

        HOST DEVICE GroupIterator(DividedVector& dv, size_t idx)
            : divided_vector(dv), group_idx(idx) {}

        HOST DEVICE IndexValuePair<int, T>* begin() const {
            return divided_vector.group_begin(group_idx);
        }

        HOST DEVICE IndexValuePair<int, T>* end() const {
            return divided_vector.group_end(group_idx);
        }

        HOST DEVICE [[nodiscard]] size_t size() const {
            return divided_vector.group_size(group_idx);
        }
    };

    // Get an iterator for a specific group
    HOST DEVICE GroupIterator get_group(size_t group_idx) {
        return GroupIterator(*this, group_idx);
    }
};

#endif // STRUCTS_H
