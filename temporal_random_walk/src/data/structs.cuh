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

template <typename T, typename U>
struct IndexValuePair {
    T index;
    U value;

    HOST DEVICE IndexValuePair() : index(), value() {}
    HOST DEVICE IndexValuePair(const T& idx, const U& val) : index(idx), value(val) {}

    HOST DEVICE IndexValuePair& operator=(const IndexValuePair& other) {
        if (this != &other) {
            index = other.index;
            value = other.value;
        }
        return *this;
    }
};


struct NodeWithTime {
    int node;
    int64_t timestamp;

    HOST DEVICE NodeWithTime(): node(-1), timestamp(-1) {}

    HOST DEVICE NodeWithTime(const int node, const int64_t timestamp): node(node), timestamp(timestamp) {}

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
    T* data = nullptr;
    size_t size;
    bool use_gpu;

    // Constructor allocates memory internally
    HOST DataBlock(const size_t size, const bool use_gpu) : size(size), use_gpu(use_gpu) {
        if (size == 0) {
            data = nullptr;
        }
        #ifdef HAS_CUDA
        else if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&data, size * sizeof(T)));
        }
        #endif
        else {
            data = static_cast<T *>(malloc(sizeof(T) * size));  // CPU allocation
        }
    }

    HOST ~DataBlock() {
        if (data) {
            #ifdef HAS_CUDA
            if (use_gpu) {
                CUDA_CHECK_AND_CLEAR(cudaFree(data));
            }
            else
            #endif
            {
                free(data);
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

    HOST DEVICE EdgeWithEndpointType(const long edge_id, const bool is_source): edge_id(edge_id), is_source(is_source) {}

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

template <typename T>
struct DividedVector {
    IndexValuePair<int, T>* elements = nullptr;
    size_t* group_offsets = nullptr;
    size_t num_groups = 0;
    bool use_gpu = false;

    size_t elements_size = 0;
    size_t group_offsets_size = 0;
    size_t total_len = 0;

    // Default constructor
    HOST DividedVector(): num_groups(0), use_gpu(false) {}

    // Constructor - divides input array into n groups
    HOST DividedVector(const T* input, const size_t input_size, const int n, bool use_gpu)
        : num_groups(n), use_gpu(use_gpu)
    {
        const int total_size = static_cast<int>(input_size);
        const int base_size = total_size / n;
        const int remainder = total_size % n;

        // Allocate space for group offsets (n+1 offsets for n groups)
        allocate_memory(&group_offsets, n + 1, use_gpu);
        group_offsets_size = n + 1;

        // Calculate and store group offsets (on host first)
        size_t* host_group_offsets = use_gpu ? new size_t[n + 1] : group_offsets;

        size_t current_offset = 0;
        host_group_offsets[0] = current_offset;

        for (int i = 0; i < n; i++) {
            const int group_size = base_size + (i < remainder ? 1 : 0);
            current_offset += group_size;
            host_group_offsets[i + 1] = current_offset;
        }

        // Allocate space for all elements
        allocate_memory(&elements, total_size, use_gpu);
        elements_size = total_size;

        // Populate the elements array
        #ifdef HAS_CUDA
        if (use_gpu) {
            // Create temporary host array
            IndexValuePair<int, T>* host_elements = new IndexValuePair<int, T>[total_size];

            for (int i = 0; i < n; i++) {
                const size_t start_idx = (i == 0) ? 0 : host_group_offsets[i];
                const size_t end_idx = host_group_offsets[i + 1];

                for (size_t j = start_idx; j < end_idx; ++j) {
                    host_elements[j] = IndexValuePair<int, T>(j, input[j]);
                }
            }

            // Copy to device
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(elements, host_elements, total_size * sizeof(IndexValuePair<int, T>), cudaMemcpyHostToDevice));
            delete[] host_elements;
        }
        else
        #endif
        {
            // Directly fill the host array
            for (int i = 0; i < n; i++) {
                const size_t start_idx = host_group_offsets[i];
                const size_t end_idx = host_group_offsets[i + 1];

                for (size_t j = start_idx; j < end_idx; ++j) {
                    elements[j] = IndexValuePair<int, T>(j, input[j]);
                }
            }
        }

        #ifdef HAS_CUDA
        // If using GPU, copy offsets to device
        if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(group_offsets, host_group_offsets, (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
            delete[] host_group_offsets;
        }
        #endif

        total_len = total_size;
    }

    // Copy constructor
    HOST DividedVector(const DividedVector& other)
        : num_groups(other.num_groups), use_gpu(other.use_gpu),
          elements_size(other.elements_size), group_offsets_size(other.group_offsets_size),
          total_len(other.total_len)
    {
        // Allocate and copy elements
        allocate_memory(&elements, elements_size, use_gpu);
        copy_memory(elements, other.elements, elements_size, use_gpu, other.use_gpu);

        // Allocate and copy group offsets
        allocate_memory(&group_offsets, group_offsets_size, use_gpu);
        copy_memory(group_offsets, other.group_offsets, group_offsets_size, use_gpu, other.use_gpu);
    }

    // Move constructor
    HOST DividedVector(DividedVector&& other) noexcept
        : elements(other.elements), group_offsets(other.group_offsets),
          num_groups(other.num_groups), use_gpu(other.use_gpu),
          elements_size(other.elements_size), group_offsets_size(other.group_offsets_size),
          total_len(other.total_len)
    {
        // Reset the source object
        other.elements = nullptr;
        other.group_offsets = nullptr;
        other.num_groups = 0;
        other.elements_size = 0;
        other.group_offsets_size = 0;
        other.total_len = 0;
    }

    // Copy assignment operator
    HOST DividedVector& operator=(const DividedVector& other) {
        if (this != &other) {
            // Free existing memory
            clear_memory(&elements, use_gpu);
            clear_memory(&group_offsets, use_gpu);

            // Copy properties
            num_groups = other.num_groups;
            use_gpu = other.use_gpu;
            elements_size = other.elements_size;
            group_offsets_size = other.group_offsets_size;
            total_len = other.total_len;

            // Allocate and copy elements
            allocate_memory(&elements, elements_size, use_gpu);
            copy_memory(elements, other.elements, elements_size, use_gpu, other.use_gpu);

            // Allocate and copy group offsets
            allocate_memory(&group_offsets, group_offsets_size, use_gpu);
            copy_memory(group_offsets, other.group_offsets, group_offsets_size, use_gpu, other.use_gpu);
        }
        return *this;
    }

    // Move assignment operator
    HOST DividedVector& operator=(DividedVector&& other) noexcept {
        if (this != &other) {
            // Free existing memory
            clear_memory(&elements, use_gpu);
            clear_memory(&group_offsets, use_gpu);

            // Move properties and pointers
            elements = other.elements;
            group_offsets = other.group_offsets;
            num_groups = other.num_groups;
            use_gpu = other.use_gpu;
            elements_size = other.elements_size;
            group_offsets_size = other.group_offsets_size;
            total_len = other.total_len;

            // Reset the source object
            other.elements = nullptr;
            other.group_offsets = nullptr;
            other.num_groups = 0;
            other.elements_size = 0;
            other.group_offsets_size = 0;
            other.total_len = 0;
        }
        return *this;
    }

    // Destructor
    HOST ~DividedVector() {
        clear_memory(&elements, use_gpu);
        clear_memory(&group_offsets, use_gpu);
    }

    // Get begin iterator for a specific group
    HOST IndexValuePair<int, T>* group_begin(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }

        size_t offset;

        #ifdef HAS_CUDA
        if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&offset, &group_offsets[group_idx], sizeof(size_t), cudaMemcpyDeviceToHost));
        }
        else
        #endif
        {
            offset = group_offsets[group_idx];
        }

        return elements + offset;
    }

    // Get end iterator for a specific group
    HOST IndexValuePair<int, T>* group_end(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }

        size_t offset;

        #ifdef HAS_CUDA
        if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&offset, &group_offsets[group_idx + 1], sizeof(size_t), cudaMemcpyDeviceToHost));
        }
        else
        #endif
        {
            offset = group_offsets[group_idx + 1];
        }

        return elements + offset;
    }

    // Get size of a specific group
    HOST [[nodiscard]] size_t group_size(size_t group_idx) const {
        if (group_idx >= num_groups) {
            return 0;
        }

        size_t start, end;

        #ifdef HAS_CUDA
        if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&start, &group_offsets[group_idx], sizeof(size_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&end, &group_offsets[group_idx + 1], sizeof(size_t), cudaMemcpyDeviceToHost));
        }
        else
        #endif
        {
            start = group_offsets[group_idx];
            end = group_offsets[group_idx + 1];
        }

        return end - start;
    }


    // Helper class for iterating over a group
    struct GroupIterator {
        DividedVector& divided_vector;
        size_t group_idx;

        HOST GroupIterator(DividedVector& dv, size_t idx)
            : divided_vector(dv), group_idx(idx) {}

        HOST IndexValuePair<int, T>* begin() const {
            return divided_vector.group_begin(group_idx);
        }

        HOST IndexValuePair<int, T>* end() const {
            return divided_vector.group_end(group_idx);
        }

        HOST [[nodiscard]] size_t size() const {
            return divided_vector.group_size(group_idx);
        }
    };

    // Get an iterator for a specific group
    HOST GroupIterator get_group(size_t group_idx) {
        return GroupIterator(*this, group_idx);
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

    #ifdef HAS_CUDA

    HOST WalkSet* to_device_ptr() const {
        WalkSet* device_walk_set;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_walk_set, sizeof(WalkSet)));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_walk_set, this, sizeof(WalkSet), cudaMemcpyHostToDevice));
        return device_walk_set;
    }

    // Method to copy data from a device WalkSet to a host-allocated memory
    HOST void copy_from_device(const WalkSet* d_walk_set) {
        // First copy the metadata structure
        WalkSet temp_walk_set;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&temp_walk_set, d_walk_set, sizeof(WalkSet), cudaMemcpyDeviceToHost));

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
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(nodes, temp_walk_set.nodes,
                   sizeof(int) * temp_walk_set.nodes_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(timestamps, temp_walk_set.timestamps,
                   sizeof(int64_t) * temp_walk_set.timestamps_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(walk_lens, temp_walk_set.walk_lens,
                   sizeof(size_t) * temp_walk_set.walk_lens_size, cudaMemcpyDeviceToHost));

        // Update metadata
        num_walks = temp_walk_set.num_walks;
        max_len = temp_walk_set.max_len;
        total_len = temp_walk_set.total_len;
        use_gpu = false; // We copied to host memory
    }

    #endif

    HOST DEVICE void add_hop(const int walk_number, const int node, const int64_t timestamp) const {
        const size_t offset = walk_number * max_len + walk_lens[walk_number];
        nodes[offset] = node;
        timestamps[offset] = timestamp;
        walk_lens[walk_number] += 1;
    }

    HOST size_t get_walk_len(const int walk_number) const {
        #ifdef HAS_CUDA
        if (use_gpu) {
            // Need to copy from device to host
            size_t walk_len;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&walk_len, &walk_lens[walk_number], sizeof(size_t), cudaMemcpyDeviceToHost));
            return walk_len;
        }
        else
        #endif
        {
            return walk_lens[walk_number];
        }
    }

    DEVICE size_t get_walk_len_device(const int walk_number) const {
        return walk_lens[walk_number];
    }

    HOST NodeWithTime get_walk_hop(const int walk_number, const int hop_number, const std::unordered_map<int, int>* node_index=nullptr) const {
        #ifdef HAS_CUDA
        if (use_gpu) {
            // Need to copy from device to host
            size_t walk_len;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&walk_len, &walk_lens[walk_number], sizeof(size_t), cudaMemcpyDeviceToHost));

            if (hop_number < 0 || hop_number >= walk_len) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            const size_t offset = walk_number * max_len + hop_number;
            int node;
            int64_t timestamp;
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&node, &nodes[offset], sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(&timestamp, &timestamps[offset], sizeof(int64_t), cudaMemcpyDeviceToHost));

            if (!node_index) {
                return NodeWithTime{node, timestamp};
            } else {
                return NodeWithTime{node_index->at(node), timestamp};
            }
        }
        else
        #endif
        {
            size_t walk_len = walk_lens[walk_number];
            if (hop_number < 0 || hop_number >= walk_len) {
                return NodeWithTime{-1, -1};  // Return invalid entry
            }

            const size_t offset = walk_number * max_len + hop_number;
            return NodeWithTime{nodes[offset], timestamps[offset]};
        }
    }

    HOST DEVICE void reverse_walk(const int walk_number) const {
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