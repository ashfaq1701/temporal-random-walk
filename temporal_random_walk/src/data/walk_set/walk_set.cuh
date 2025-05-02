#ifndef WALK_SET_CUH
#define WALK_SET_CUH

#include "common.cuh"
#include "walk.cuh"
#include "walks_iterator.cuh"

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

    // Constructors
    HOST WalkSet();
    HOST WalkSet(const size_t num_walks, const size_t max_len, const bool use_gpu);

    // Copy/move constructors and assignment operators
    HOST WalkSet(const WalkSet &other);
    HOST WalkSet(WalkSet &&other) noexcept;
    HOST WalkSet& operator=(const WalkSet& other);
    HOST WalkSet& operator=(WalkSet&& other) noexcept;

    // Destructor
    HOST ~WalkSet();

    // CUDA specific methods
    #ifdef HAS_CUDA
    HOST WalkSet* to_device_ptr() const;
    HOST void copy_from_device(const WalkSet* d_walk_set);
    #endif

    // Walk operations
    HOST DEVICE void add_hop(const int walk_number, const int node, const int64_t timestamp) const;
    HOST size_t get_walk_len(const int walk_number) const;
    DEVICE size_t get_walk_len_device(const int walk_number) const;
    HOST DEVICE void reverse_walk(const int walk_number) const;

    // Walk properties
    HOST size_t size() const;

    // Iterator support
    HOST Walk get_walk(const int walk_number) const;
    HOST WalksIterator walks_begin() const;
    HOST WalksIterator walks_end() const;
};

// Include implementation. DO NOT REMOVE.
#include "walk_set_impl.cuh"

#endif // WALK_SET_CUH
