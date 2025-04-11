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

#endif // STRUCTS_H