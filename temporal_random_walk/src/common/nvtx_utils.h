#pragma once
#ifdef HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

// ------------------------------
// NVTX helper (no-op on CPU)
// ------------------------------
struct NvtxRange {
#ifdef HAS_CUDA
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char*) {}
#endif
};
