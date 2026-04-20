#ifndef PY_INTERFACE_BUFFER_TO_NUMPY_CUH
#define PY_INTERFACE_BUFFER_TO_NUMPY_CUH

#include <cstdlib>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../src/data/buffer.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

// Build a py::capsule whose destructor frees a HostRelease'd block via
// the correct allocator. Context is a small heap-allocated HostRelease
// so the destructor — which must be a plain C function pointer — can
// recover the (bytes, pinned) state without captures.
inline py::capsule make_capsule(HostRelease r) {
    auto* ctx = new HostRelease(r);
    return py::capsule(ctx, +[](void* vp) noexcept {
        auto* c = static_cast<HostRelease*>(vp);
        if (c->ptr) {
#ifdef HAS_CUDA
            if (c->pinned) {
                cudaFreeHost(c->ptr);
                cudaGetLastError();
                g_total_pinned_host_bytes.fetch_sub(
                    c->bytes, std::memory_order_relaxed);
            } else
#endif
            {
                std::free(c->ptr);
            }
        }
        delete c;
    });
}

#endif // PY_INTERFACE_BUFFER_TO_NUMPY_CUH
