#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstddef>
#include "../common/memory.cuh"
#include "../common/macros.cuh"

struct Edge {
    int u;
    int i;
    int64_t ts;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts) : u(u), i(i), ts(ts) {}

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

struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(const size_t f, const size_t t) : from(f), to(t) {}

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

template <typename T>
struct DataBlock {
    T* data;
    size_t size;
    bool use_gpu;

    // Constructor allocates memory internally
    DataBlock(const size_t size, const bool use_gpu) : size(size), use_gpu(use_gpu) {
        if (size == 0) {
            data = nullptr;
        } else if (use_gpu) {
            cudaMalloc(&data, size * sizeof(T));
        } else {
            data = new T[size];  // CPU allocation
        }
    }

    ~DataBlock() {
        if (data) {
            if (use_gpu) {
                cudaFree(data);
            } else {
                delete[] data;
            }
        }
    }
};

template <typename T>
struct MemoryView {
    T* data;
    size_t size;
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

struct WalkSet {
    size_t num_walks;
    size_t max_len;
    bool use_gpu;

    int* nodes = nullptr;
    int64_t* timestamps = nullptr;
    size_t* walk_lens = nullptr;

    size_t nodes_size = 0;
    size_t timestamps_size = 0;
    size_t walk_lens_size = 0;

    size_t total_len = 0;

    HOST WalkSet(): num_walks(0), max_len(0), use_gpu(false) {}

    HOST WalkSet(const size_t num_walks, const size_t max_len, const bool use_gpu):
        num_walks(num_walks), max_len(max_len), use_gpu(use_gpu) {

        total_len = num_walks * max_len;

        allocate_memory(&nodes, total_len, use_gpu);
        nodes_size = total_len;

        allocate_memory(&timestamps, total_len, use_gpu);
        timestamps_size = total_len;

        allocate_memory(&walk_lens, num_walks, use_gpu);
        walk_lens_size = num_walks;

        // Initialize walk_lens to zero
        fill_memory(walk_lens, num_walks, static_cast<size_t>(0), use_gpu);
    }

    // Copy constructor
    HOST WalkSet(const WalkSet& other)
        : num_walks(other.num_walks), max_len(other.max_len), use_gpu(other.use_gpu),
          total_len(other.total_len) {

        // Allocate and copy nodes
        allocate_memory(&nodes, other.nodes_size, use_gpu);
        nodes_size = other.nodes_size;
        copy_memory(nodes, other.nodes, nodes_size, use_gpu, other.use_gpu);

        // Allocate and copy timestamps
        allocate_memory(&timestamps, other.timestamps_size, use_gpu);
        timestamps_size = other.timestamps_size;
        copy_memory(timestamps, other.timestamps, timestamps_size, use_gpu, other.use_gpu);

        // Allocate and copy walk_lens
        allocate_memory(&walk_lens, other.walk_lens_size, use_gpu);
        walk_lens_size = other.walk_lens_size;
        copy_memory(walk_lens, other.walk_lens, walk_lens_size, use_gpu, other.use_gpu);
    }

    // Move constructor
    HOST WalkSet(WalkSet&& other) noexcept
        : num_walks(other.num_walks), max_len(other.max_len), use_gpu(other.use_gpu),
          nodes(other.nodes), timestamps(other.timestamps), walk_lens(other.walk_lens),
          nodes_size(other.nodes_size), timestamps_size(other.timestamps_size),
          walk_lens_size(other.walk_lens_size), total_len(other.total_len) {

        // Reset the source object to prevent double-free issues
        other.num_walks = 0;
        other.max_len = 0;
        other.total_len = 0;
        other.nodes = nullptr;
        other.timestamps = nullptr;
        other.walk_lens = nullptr;
        other.nodes_size = 0;
        other.timestamps_size = 0;
        other.walk_lens_size = 0;
    }

    // Copy assignment operator
    HOST WalkSet& operator=(const WalkSet& other) {
        if (this != &other) {
            // Free existing memory
            clear_memory(&nodes, use_gpu);
            clear_memory(&timestamps, use_gpu);
            clear_memory(&walk_lens, use_gpu);

            num_walks = other.num_walks;
            max_len = other.max_len;
            use_gpu = other.use_gpu;
            total_len = other.total_len;

            // Allocate and copy nodes
            allocate_memory(&nodes, other.nodes_size, use_gpu);
            nodes_size = other.nodes_size;
            copy_memory(nodes, other.nodes, nodes_size, use_gpu, other.use_gpu);

            // Allocate and copy timestamps
            allocate_memory(&timestamps, other.timestamps_size, use_gpu);
            timestamps_size = other.timestamps_size;
            copy_memory(timestamps, other.timestamps, timestamps_size, use_gpu, other.use_gpu);

            // Allocate and copy walk_lens
            allocate_memory(&walk_lens, other.walk_lens_size, use_gpu);
            walk_lens_size = other.walk_lens_size;
            copy_memory(walk_lens, other.walk_lens, walk_lens_size, use_gpu, other.use_gpu);
        }
        return *this;
    }

    // Move assignment operator
    HOST WalkSet& operator=(WalkSet&& other) noexcept {
        if (this != &other) {
            // Free existing memory
            clear_memory(&nodes, use_gpu);
            clear_memory(&timestamps, use_gpu);
            clear_memory(&walk_lens, use_gpu);

            num_walks = other.num_walks;
            max_len = other.max_len;
            use_gpu = other.use_gpu;
            total_len = other.total_len;

            // Move pointers
            nodes = other.nodes;
            timestamps = other.timestamps;
            walk_lens = other.walk_lens;

            nodes_size = other.nodes_size;
            timestamps_size = other.timestamps_size;
            walk_lens_size = other.walk_lens_size;

            // Reset the source object
            other.num_walks = 0;
            other.max_len = 0;
            other.total_len = 0;
            other.nodes = nullptr;
            other.timestamps = nullptr;
            other.walk_lens = nullptr;
            other.nodes_size = 0;
            other.timestamps_size = 0;
            other.walk_lens_size = 0;
        }
        return *this;
    }

    // Destructor
    HOST ~WalkSet() {
        clear_memory(&nodes, use_gpu);
        clear_memory(&timestamps, use_gpu);
        clear_memory(&walk_lens, use_gpu);
    }

    HOST WalkSet* to_device_ptr() {
        WalkSet* device_walk_set;
        cudaMalloc(&device_walk_set, sizeof(WalkSet));
        cudaMemcpy(device_walk_set, this, sizeof(WalkSet), cudaMemcpyHostToDevice);
        return device_walk_set;
    }

    // Method to copy data from a device WalkSet to a host-allocated memory
    HOST void copy_from_device(const WalkSet* d_walk_set) {
        // First copy the metadata structure
        WalkSet temp_walk_set;
        cudaMemcpy(&temp_walk_set, d_walk_set, sizeof(WalkSet), cudaMemcpyDeviceToHost);

        // Ensure host memory is properly sized
        if (nodes_size < temp_walk_set.nodes_size) {
            clear_memory(&nodes, use_gpu);
            allocate_memory(&nodes, temp_walk_set.nodes_size, false); // Allocate on host
            nodes_size = temp_walk_set.nodes_size;
        }

        if (timestamps_size < temp_walk_set.timestamps_size) {
            clear_memory(&timestamps, use_gpu);
            allocate_memory(&timestamps, temp_walk_set.timestamps_size, false); // Allocate on host
            timestamps_size = temp_walk_set.timestamps_size;
        }

        if (walk_lens_size < temp_walk_set.walk_lens_size) {
            clear_memory(&walk_lens, use_gpu);
            allocate_memory(&walk_lens, temp_walk_set.walk_lens_size, false); // Allocate on host
            walk_lens_size = temp_walk_set.walk_lens_size;
        }

        // Copy data from device memory to host memory
        cudaMemcpy(nodes, temp_walk_set.nodes,
                   sizeof(int) * temp_walk_set.nodes_size, cudaMemcpyDeviceToHost);

        cudaMemcpy(timestamps, temp_walk_set.timestamps,
                   sizeof(int64_t) * temp_walk_set.timestamps_size, cudaMemcpyDeviceToHost);

        cudaMemcpy(walk_lens, temp_walk_set.walk_lens,
                   sizeof(size_t) * temp_walk_set.walk_lens_size, cudaMemcpyDeviceToHost);

        // Update metadata
        num_walks = temp_walk_set.num_walks;
        max_len = temp_walk_set.max_len;
        total_len = temp_walk_set.total_len;
        use_gpu = false; // We copied to host memory
    }

    HOST DEVICE void add_hop(const int walk_number, const int node, const int64_t timestamp) {
        const size_t offset = walk_number * max_len + walk_lens[walk_number];
        nodes[offset] = node;
        timestamps[offset] = timestamp;
        walk_lens[walk_number] += 1;
    }

    HOST size_t get_walk_len(const int walk_number) {
        if (use_gpu) {
            // Need to copy from device to host
            size_t walk_len;
            cudaMemcpy(&walk_len, &walk_lens[walk_number], sizeof(size_t), cudaMemcpyDeviceToHost);
            return walk_len;
        } else {
            return walk_lens[walk_number];
        }
    }

    DEVICE size_t get_walk_len_device(const int walk_number) {
        return walk_lens[walk_number];
    }

    HOST NodeWithTime get_walk_hop(const int walk_number, const int hop_number) {
        if (use_gpu) {
            // Need to copy from device to host
            size_t walk_len;
            cudaMemcpy(&walk_len, &walk_lens[walk_number], sizeof(size_t), cudaMemcpyDeviceToHost);

            if (hop_number < 0 || hop_number >= walk_len) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            const size_t offset = walk_number * max_len + hop_number;
            int node;
            int64_t timestamp;
            cudaMemcpy(&node, &nodes[offset], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&timestamp, &timestamps[offset], sizeof(int64_t), cudaMemcpyDeviceToHost);

            return NodeWithTime{node, timestamp};
        } else {
            size_t walk_len = walk_lens[walk_number];
            if (hop_number < 0 || hop_number >= walk_len) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            const size_t offset = walk_number * max_len + hop_number;
            return NodeWithTime{nodes[offset], timestamps[offset]};
        }
    }

    HOST DEVICE void reverse_walk(const int walk_number) {
        const size_t walk_length = walk_lens[walk_number];
        if (walk_length <= 1) return; // No need to reverse if walk is empty or has one hop

        const size_t start = walk_number * max_len;
        const size_t end = start + walk_length - 1;

        for (size_t i = 0; i < walk_length / 2; ++i) {
            // Swap nodes
            const int temp_node = nodes[start + i];
            nodes[start + i] = nodes[end - i];
            nodes[end - i] = temp_node;

            // Swap timestamps
            const int64_t temp_time = timestamps[start + i];
            timestamps[start + i] = timestamps[end - i];
            timestamps[end - i] = temp_time;
        }
    }
};

#endif // STRUCTS_H