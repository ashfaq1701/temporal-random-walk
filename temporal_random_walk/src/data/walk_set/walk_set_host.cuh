#ifndef DATA_WALK_SET_WALK_SET_HOST_CUH
#define DATA_WALK_SET_WALK_SET_HOST_CUH

#include <cstddef>
#include <cstdint>

#include "../../common/const.cuh"
#include "../../common/macros.cuh"
#include "../buffer.cuh"
#include "walk_set_view.cuh"

/**
 * WalkSetHost — host-resident container of finished walks. Designed for
 * two consumers:
 *   1. C++ users iterating walks via Walk / WalksIterator (existing
 *      classes; they are non-owning views over the four arrays and do
 *      not need to change).
 *   2. The pybind11 layer handing the arrays to numpy via py::capsule.
 *      Each buffer is a single malloc-allocated block whose ownership
 *      is transferred to Python by release_*() — after release, the
 *      pointer is nulled and the host destructor is a no-op on that
 *      field.
 *
 * Move-only.
 *
 * Note on allocator: pybind11 capsule deleters here call std::free
 * because the buffers are allocated via malloc (Buffer<T> with
 * use_gpu=false). If you change Buffer's host allocator, update the
 * capsule deleter in py_interface/ accordingly.
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

    size_t non_empty_count() const noexcept;

    bool is_pinned_host() const noexcept { return nodes_.is_pinned_host(); }

    int*     release_nodes_as_raw()      noexcept;
    int64_t* release_timestamps_as_raw() noexcept;
    size_t*  release_walk_lens_as_raw()  noexcept;
    int64_t* release_edge_ids_as_raw()   noexcept;

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

#endif // DATA_WALK_SET_WALK_SET_HOST_CUH
