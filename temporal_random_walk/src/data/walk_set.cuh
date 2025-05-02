#ifndef WALK_SET_CUH
#define WALK_SET_CUH

#include "../common/macros.cuh"
#include <cstddef>
#include <stdexcept>

#ifdef HAS_CUDA
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#endif

#include "../common/const.cuh"

#include "../common/cuda_config.cuh"
#include "../common/memory.cuh"

// Forward declaration of WalkSet
struct WalkSet;

// Step representation for a node and timestamp pair
struct Step {
    int node;
    int64_t timestamp;
};

// Forward declarations
class Walk;
class WalkIterator;
class WalksIterator;

// Iterator for a single walk
class WalkIterator {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    size_t index_;
    size_t length_;

public:
    // Constructor
    HOST WalkIterator(const int* nodes, const int64_t* timestamps, const size_t index, const size_t length)
    : nodes_(nodes), timestamps_(timestamps), index_(index), length_(length) {}

    // Iterator operations
    Step operator*() const {
        return {nodes_[index_], timestamps_[index_]};
    }

    WalkIterator& operator++() {
        ++index_;
        return *this;
    }

    WalkIterator operator++(int) {
        const WalkIterator temp = *this;
        ++index_;
        return temp;
    }

    bool operator==(const WalkIterator& other) const {
        return index_ == other.index_;
    }

    bool operator!=(const WalkIterator& other) const {
        return index_ != other.index_;
    }
};

// Class representing a single walk (non-owning view)
class Walk {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    size_t length_;

public:
    // Constructor
    HOST Walk(const int* nodes, const int64_t* timestamps, const size_t length)
        : nodes_(nodes), timestamps_(timestamps), length_(length) {}

    // Access operations
    Step operator[](const size_t index) const {
        if (index >= length_) {
            throw std::out_of_range("Walk index out of range");
        }
        return {nodes_[index], timestamps_[index]};
    }

    size_t size() const {
        return length_;
    }

    bool empty() const {
        return length_ == 0;
    }

    // Iterator support
    WalkIterator begin() const {
        return {nodes_, timestamps_, 0, length_};
    }

    WalkIterator end() const {
        return {nodes_, timestamps_, length_, length_};
    }

    // Direct access
    const int* nodes() const {
        return nodes_;
    }

    const int64_t* timestamps() const {
        return timestamps_;
    }

    HOST Step back() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        return {nodes_[length_ - 1], timestamps_[length_ - 1]};
    }

    HOST Step front() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        return {nodes_[0], timestamps_[0]};
    }
};

// Iterator for all non-empty walks in the WalkSet
class WalksIterator {
private:
    const WalkSet* walk_set_;
    size_t walk_index_;
    size_t max_walks_;

    // Helper function
    HOST void find_next_non_empty();

public:
    // Constructor
    explicit WalksIterator(const WalkSet* walk_set, const size_t start_index = 0);

    // Iterator operations
    Walk operator*() const;

    WalksIterator& operator++();

    WalksIterator operator++(int);

    bool operator==(const WalksIterator& other) const;

    HOST bool operator!=(const WalksIterator& other) const;
};

// Main WalkSet structure definition
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
    bool owns_data = true;

    // Constructors and assignment operators
    HOST WalkSet(): num_walks(0), max_len(0), use_gpu(false) {}

    HOST WalkSet(const size_t num_walks, const size_t max_len, const bool use_gpu): num_walks(num_walks),
                                                                                        max_len(max_len),
                                                                                        use_gpu(use_gpu) {
        total_len = num_walks * max_len;

        allocate_memory(&nodes, total_len, use_gpu);
        fill_memory(nodes, total_len, EMPTY_NODE_VALUE, use_gpu);
        nodes_size = total_len;

        allocate_memory(&timestamps, total_len, use_gpu);
        fill_memory(timestamps, total_len, EMPTY_TIMESTAMP_VALUE, use_gpu);
        timestamps_size = total_len;

        allocate_memory(&walk_lens, num_walks, use_gpu);
        fill_memory(walk_lens, num_walks, static_cast<size_t>(0), use_gpu);
        walk_lens_size = num_walks;
    }

    HOST WalkSet(const WalkSet &other)
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

    HOST WalkSet(WalkSet &&other) noexcept
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

    HOST ~WalkSet() {
        if (owns_data) {
            clear_memory(&nodes, use_gpu);
            clear_memory(&timestamps, use_gpu);
            clear_memory(&walk_lens, use_gpu);
        } else {
            nodes = nullptr;
            timestamps = nullptr;
            walk_lens = nullptr;
        }
    }

    // CUDA specific methods
    #ifdef HAS_CUDA
    HOST WalkSet* to_device_ptr() const {
        WalkSet *device_walk_set;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_walk_set, sizeof(WalkSet)));
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(device_walk_set, this, sizeof(WalkSet), cudaMemcpyHostToDevice));
        return device_walk_set;
    }

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

    // Walk operations
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
        } else
        #endif
        {
            return walk_lens[walk_number];
        }
    }

    DEVICE size_t get_walk_len_device(const int walk_number) const {
        return walk_lens[walk_number];
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

    // Walk properties
    HOST size_t size() const {
        size_t non_empty_count = 0;

        #ifdef HAS_CUDA
        if (use_gpu) {
            const thrust::device_ptr<size_t> d_walk_lens(walk_lens);
            non_empty_count = thrust::count_if(
                DEVICE_EXECUTION_POLICY,
                d_walk_lens,
                d_walk_lens + static_cast<long>(num_walks),
                [] __device__ (const size_t len) { return len > 0; }
            );
            CUDA_KERNEL_CHECK("After thrust count_if in WalkSet::size");
        }
        else
        #endif
        {
            for (size_t i = 0; i < num_walks; i++) {
                if (walk_lens[i] > 0) {
                    non_empty_count++;
                }
            }
        }

        return non_empty_count;
    }

    // Iterator support
    HOST Walk get_walk(const int walk_number) const {
        const size_t walk_len = get_walk_len(walk_number);
        const size_t offset = walk_number * max_len;

        #ifdef HAS_CUDA
        if (use_gpu) {
            // WARNING: This returns a Walk with device pointers
            // It can only be used safely in device code
        }
        #endif

        return {nodes + offset, timestamps + offset, walk_len};
    }

    HOST WalksIterator walks_begin() const {
        return WalksIterator{this};
    }

    HOST WalksIterator walks_end() const {
        return WalksIterator{this, num_walks};
    }
};

HOST inline void WalksIterator::find_next_non_empty() {
	while (walk_index_ < max_walks_ && walk_set_->walk_lens[walk_index_] == 0) {
    	++walk_index_;
    }
}

inline WalksIterator::WalksIterator(const WalkSet* walk_set, const size_t start_index)
    : walk_set_(walk_set), walk_index_(start_index), max_walks_(walk_set->num_walks) {
  	find_next_non_empty();
}

// Iterator operations
inline Walk WalksIterator::operator*() const {
  	size_t len = walk_set_->walk_lens[walk_index_];
    const size_t offset = walk_index_ * walk_set_->max_len;
    return {walk_set_->nodes + offset, walk_set_->timestamps + offset, len};
}

inline WalksIterator& WalksIterator::operator++() {
	++walk_index_;
    find_next_non_empty();
    return *this;
}

inline WalksIterator WalksIterator::operator++(int) {
	const WalksIterator temp = *this;
    ++(*this);
    return temp;
}

inline bool WalksIterator::operator==(const WalksIterator& other) const {
	return walk_index_ == other.walk_index_;
}

HOST inline bool WalksIterator::operator!=(const WalksIterator& other) const {
	return walk_index_ != other.walk_index_;
}

#endif // WALK_SET_CUH
