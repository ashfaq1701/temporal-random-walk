#ifndef WALK_SET_CUH
#define WALK_SET_CUH

#include "../common/macros.cuh"
#include <cstddef>
#include <stdexcept>

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
    WalkIterator(const int* nodes, const int64_t* timestamps, size_t index, size_t length);

    // Iterator operations
    Step operator*() const;
    WalkIterator& operator++();
    WalkIterator operator++(int);
    bool operator==(const WalkIterator& other) const;
    bool operator!=(const WalkIterator& other) const;
};

// Class representing a single walk (non-owning view)
class Walk {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    size_t length_;

public:
    // Constructor
    Walk(const int* nodes, const int64_t* timestamps, size_t length);

    // Access operations
    Step operator[](size_t index) const;
    size_t size() const;
    bool empty() const;

    // Iterator support
    WalkIterator begin() const;
    WalkIterator end() const;

    // Direct access
    const int* nodes() const;
    const int64_t* timestamps() const;
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
    explicit WalksIterator(const WalkSet* walk_set, size_t start_index = 0);

    // Iterator operations
    Walk operator*() const;
    WalksIterator& operator++();
    WalksIterator operator++(int);
    bool operator==(const WalksIterator& other) const;
    bool operator!=(const WalksIterator& other) const;
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
    HOST WalkSet();
    HOST WalkSet(const size_t num_walks, const size_t max_len, const bool use_gpu);
    HOST WalkSet(const WalkSet& other);
    HOST WalkSet(WalkSet&& other) noexcept;
    HOST WalkSet& operator=(const WalkSet& other);
    HOST WalkSet& operator=(WalkSet&& other) noexcept;
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

#endif // WALK_SET_CUH
