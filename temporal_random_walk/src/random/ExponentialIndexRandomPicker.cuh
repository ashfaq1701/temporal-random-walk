#ifndef EXPONENTIALINDEXRANDOMPICKER_H
#define EXPONENTIALINDEXRANDOMPICKER_H

#include "../data/enums.h"
#include "../cuda_common/macros.cuh"
#include "IndexBasedRandomPicker.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
class ExponentialIndexRandomPicker final : public IndexBasedRandomPicker<GPUUsage> {
public:
    HOST int pick_random_host(int start, int end, bool prioritize_end) override;

    #ifdef HAS_CUDA
    DEVICE int pick_random_device(int start, int end, bool prioritize_end, curandState* rand_state) override;
    #endif

    #ifdef HAS_CUDA
    RandomPicker<GPUUsage>* to_device_ptr() override;
    #endif
};

#endif //EXPONENTIALINDEXRANDOMPICKER_H
