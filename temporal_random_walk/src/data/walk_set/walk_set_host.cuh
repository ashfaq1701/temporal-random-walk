#ifndef DATA_WALK_SET_WALK_SET_HOST_CUH
#define DATA_WALK_SET_WALK_SET_HOST_CUH

#include <cstddef>
#include <cstdint>

#include "../../common/const.cuh"
#include "../../common/macros.cuh"
#include "../buffer.cuh"
#include "walk_set_view.cuh"
#include "walks_iterator.cuh"

/**
 * WalkSetHost — host-resident container of finished walks. Sibling of
 * WalkSetDevice (GPU-resident scratch) and WalkSetView (POD kernel ABI).
 * Designed for two consumers:
 *   1. C++ users iterating walks via Walk / WalksIterator (non-owning
 *      views over the four arrays).
 *   2. The pybind11 layer handing the arrays to numpy via py::capsule.
 *      Each buffer is a single Buffer<T>-owned block whose ownership is
 *      transferred to Python by release_*() — these return HostRelease
 *      POD, and the pybind layer wraps it with make_capsule() which
 *      knows to free via std::free or cudaFreeHost depending on whether
 *      the block was pinned.
 *
 * Move-only.
 */
class WalkSetHost {
public:
    WalkSetHost() = default;

    WalkSetHost(size_t num_walks, size_t max_len, int walk_padding_value,
                bool pinned_host = false);

    WalkSetHost(const WalkSetHost&)            = delete;
    WalkSetHost& operator=(const WalkSetHost&) = delete;
    WalkSetHost(WalkSetHost&&) noexcept        = default;
    WalkSetHost& operator=(WalkSetHost&&) noexcept = default;

    size_t num_walks()     const noexcept { return num_walks_; }
    size_t max_len()       const noexcept { return max_len_; }
    int    padding_value() const noexcept { return walk_padding_value_; }

    const int*     nodes_ptr()      const { return nodes_.data(); }
    const int64_t* timestamps_ptr() const { return timestamps_.data(); }
    const size_t*  walk_lens_ptr()  const { return walk_lens_.data(); }
    const int64_t* edge_ids_ptr()   const { return edge_ids_.data(); }

    size_t nodes_size()      const { return nodes_.size(); }
    size_t timestamps_size() const { return timestamps_.size(); }
    size_t walk_lens_size()  const { return walk_lens_.size(); }
    size_t edge_ids_size()   const { return edge_ids_.size(); }

    size_t get_memory_used() const noexcept {
        return nodes_.size() * sizeof(int)
             + timestamps_.size() * sizeof(int64_t)
             + walk_lens_.size() * sizeof(size_t)
             + edge_ids_.size() * sizeof(int64_t);
    }

    // Count of walks with length > 0. Called size() for parity with the
    // container-idiom usage in tests.
    size_t non_empty_count() const noexcept;
    size_t size() const noexcept { return non_empty_count(); }

    bool is_pinned_host() const noexcept { return nodes_.is_pinned_host(); }

    // Zero-copy ownership transfer. After each call the corresponding
    // internal Buffer is empty. The returned HostRelease carries the
    // (ptr, bytes, pinned) needed to free via the correct allocator.
    HostRelease release_nodes()      { return nodes_.release_host(); }
    HostRelease release_timestamps() { return timestamps_.release_host(); }
    HostRelease release_walk_lens()  { return walk_lens_.release_host(); }
    HostRelease release_edge_ids()   { return edge_ids_.release_host(); }

    // Non-owning iteration over the host-resident walks.
    WalksIterator walks_begin() const {
        return WalksIterator{
            nodes_.data(), timestamps_.data(),
            walk_lens_.data(), edge_ids_.data(),
            num_walks_, max_len_, 0};
    }
    WalksIterator walks_end() const {
        return WalksIterator{
            nodes_.data(), timestamps_.data(),
            walk_lens_.data(), edge_ids_.data(),
            num_walks_, max_len_, num_walks_};
    }

    void overwrite_from_device_buffers(
        const Buffer<int>&     d_nodes,
        const Buffer<int64_t>& d_timestamps,
        const Buffer<size_t>&  d_walk_lens,
        const Buffer<int64_t>& d_edge_ids);

    // Build a mutable POD view for host-side walk kernels (CPU path).
    WalkSetView make_host_view() noexcept {
        WalkSetView v{};
        v.nodes              = nodes_.data();
        v.timestamps         = timestamps_.data();
        v.walk_lens          = walk_lens_.data();
        v.edge_ids           = edge_ids_.data();
        v.num_walks          = num_walks_;
        v.max_len            = max_len_;
        v.walk_padding_value = walk_padding_value_;
        return v;
    }

private:
    Buffer<int>     nodes_{false};
    Buffer<int64_t> timestamps_{false};
    Buffer<size_t>  walk_lens_{false};
    Buffer<int64_t> edge_ids_{false};

    size_t num_walks_          = 0;
    size_t max_len_            = 0;
    int    walk_padding_value_ = EMPTY_NODE_VALUE;
};

#include "walks_iterator_impl.cuh"

#endif // DATA_WALK_SET_WALK_SET_HOST_CUH
