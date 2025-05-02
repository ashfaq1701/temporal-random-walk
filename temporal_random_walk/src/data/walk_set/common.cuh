#ifndef WALK_SET_COMMON_CUH
#define WALK_SET_COMMON_CUH

// ** These includes are used by files that include this header. DO NOT REMOVE. **

#include "../../common/macros.cuh"
#include "../../common/const.cuh"
#include "../../common/cuda_config.cuh"
#include "../../common/memory.cuh"
#include <cstddef>
#include <stdexcept>

#ifdef HAS_CUDA
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#endif

// Forward declarations
struct WalkSet;
struct Step;
class Walk;
class WalkIterator;
class WalksIterator;

#endif // WALK_SET_COMMON_CUH
