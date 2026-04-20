#include "walk_set_host.cuh"

#include <cstring>
#include <cstdlib>
#include <new>
#include <stdexcept>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "../../common/error_handlers.cuh"
#endif

WalkSetHost::WalkSetHost(const size_t num_walks,
                         const size_t max_len,
                         const int walk_padding_value,
                         const bool pinned_host)
    : nodes_(false, pinned_host),
      timestamps_(false, pinned_host),
      walk_lens_(false, pinned_host),
      edge_ids_(false, pinned_host),
      num_walks_(num_walks), max_len_(max_len),
      walk_padding_value_(walk_padding_value) {

    const size_t total      = num_walks * max_len;
    const size_t edge_total = num_walks * (max_len == 0 ? 0 : max_len - 1);

    nodes_.resize(total);
    timestamps_.resize(total);
    walk_lens_.resize(num_walks);
    edge_ids_.resize(edge_total);

    nodes_.fill(walk_padding_value);
    timestamps_.fill(EMPTY_TIMESTAMP_VALUE);
    walk_lens_.fill(static_cast<size_t>(0));
    edge_ids_.fill(EMPTY_EDGE_ID);
}

size_t WalkSetHost::non_empty_count() const noexcept {
    size_t count = 0;
    const size_t* lens = walk_lens_.data();
    if (!lens) return 0;
    for (size_t i = 0; i < num_walks_; ++i) {
        if (lens[i] > 0) ++count;
    }
    return count;
}

namespace {
template <typename T>
T* copy_out_and_clear(Buffer<T>& buf) {
    const size_t n = buf.size();
    if (n == 0) {
        buf.shrink_to_fit_empty();
        return nullptr;
    }
    T* out = static_cast<T*>(std::malloc(n * sizeof(T)));
    if (!out) throw std::bad_alloc();
    std::memcpy(out, buf.data(), n * sizeof(T));
    buf.shrink_to_fit_empty();
    return out;
}
} // namespace

int*     WalkSetHost::release_nodes_as_raw()      noexcept {
    try { return copy_out_and_clear(nodes_); } catch (...) { return nullptr; }
}
int64_t* WalkSetHost::release_timestamps_as_raw() noexcept {
    try { return copy_out_and_clear(timestamps_); } catch (...) { return nullptr; }
}
size_t*  WalkSetHost::release_walk_lens_as_raw()  noexcept {
    try { return copy_out_and_clear(walk_lens_); } catch (...) { return nullptr; }
}
int64_t* WalkSetHost::release_edge_ids_as_raw()   noexcept {
    try { return copy_out_and_clear(edge_ids_); } catch (...) { return nullptr; }
}

void WalkSetHost::overwrite_from_device_buffers(
    const Buffer<int>&     d_nodes,
    const Buffer<int64_t>& d_timestamps,
    const Buffer<size_t>&  d_walk_lens,
    const Buffer<int64_t>& d_edge_ids) {

    nodes_.resize(d_nodes.size());
    timestamps_.resize(d_timestamps.size());
    walk_lens_.resize(d_walk_lens.size());
    edge_ids_.resize(d_edge_ids.size());

#ifdef HAS_CUDA
    if (d_nodes.size() > 0) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            nodes_.data(), d_nodes.data(),
            d_nodes.size() * sizeof(int),
            cudaMemcpyDeviceToHost));
    }
    if (d_timestamps.size() > 0) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            timestamps_.data(), d_timestamps.data(),
            d_timestamps.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost));
    }
    if (d_walk_lens.size() > 0) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            walk_lens_.data(), d_walk_lens.data(),
            d_walk_lens.size() * sizeof(size_t),
            cudaMemcpyDeviceToHost));
    }
    if (d_edge_ids.size() > 0) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            edge_ids_.data(), d_edge_ids.data(),
            d_edge_ids.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost));
    }
#else
    if (d_nodes.size() > 0) std::memcpy(nodes_.data(), d_nodes.data(), d_nodes.size() * sizeof(int));
    if (d_timestamps.size() > 0) std::memcpy(timestamps_.data(), d_timestamps.data(), d_timestamps.size() * sizeof(int64_t));
    if (d_walk_lens.size() > 0) std::memcpy(walk_lens_.data(), d_walk_lens.data(), d_walk_lens.size() * sizeof(size_t));
    if (d_edge_ids.size() > 0) std::memcpy(edge_ids_.data(), d_edge_ids.data(), d_edge_ids.size() * sizeof(int64_t));
#endif
}
