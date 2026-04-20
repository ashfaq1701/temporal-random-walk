#ifndef DATA_WALK_SET_WALK_SET_DEVICE_CUH
#define DATA_WALK_SET_WALK_SET_DEVICE_CUH

#include <cstddef>
#include <cstdint>

#include "../../common/const.cuh"
#include "../../common/macros.cuh"
#include "../buffer.cuh"
#include "walk_set_view.cuh"

class WalkSetHost;

/**
 * WalkSetDevice — RAII owner of the four device-side buffers that hold a
 * batch of in-progress walks.
 *
 * Construction allocates and fills each buffer with its padding value.
 * make_view() produces a POD WalkSetView suitable for kernel arguments.
 * download_to_host() (&&) consumes *this and returns a WalkSetHost with
 * freshly-allocated host buffers populated from the device buffers.
 *
 * Move-only.
 */
class WalkSetDevice {
public:
    WalkSetDevice() = default;

    WalkSetDevice(size_t num_walks,
                  size_t max_len,
                  int walk_padding_value);

    WalkSetDevice(const WalkSetDevice&)            = delete;
    WalkSetDevice& operator=(const WalkSetDevice&) = delete;
    WalkSetDevice(WalkSetDevice&&) noexcept        = default;
    WalkSetDevice& operator=(WalkSetDevice&&) noexcept = default;

    WalkSetView make_view();

    WalkSetHost download_to_host() &&;

    size_t num_walks()     const noexcept { return num_walks_; }
    size_t max_len()       const noexcept { return max_len_; }
    int    padding_value() const noexcept { return walk_padding_value_; }

private:
    Buffer<int>     nodes_{true};
    Buffer<int64_t> timestamps_{true};
    Buffer<size_t>  walk_lens_{true};
    Buffer<int64_t> edge_ids_{true};

    size_t num_walks_          = 0;
    size_t max_len_            = 0;
    int    walk_padding_value_ = EMPTY_NODE_VALUE;
};

#endif // DATA_WALK_SET_WALK_SET_DEVICE_CUH
