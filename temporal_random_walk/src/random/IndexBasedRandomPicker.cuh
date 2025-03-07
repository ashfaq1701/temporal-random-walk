#ifndef INDEXBASEDRANDOMPICKER_H
#define INDEXBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../cuda_common/macros.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
class IndexBasedRandomPicker : public RandomPicker<GPUUsage> {

public:
    ~IndexBasedRandomPicker() override = default;

    virtual int pick_random(const int start, const int end, const bool prioritize_end);

    virtual HOST int pick_random_host(int start, int end, bool prioritize_end) { return -1; }

    #ifdef HAS_CUDA
    virtual DEVICE int pick_random_device(int start, int end, bool prioritize_end, curandState* rand_state) { return -1; };
    #endif

    int get_picker_type() override
    {
        return INDEX_BASED_PICKER_TYPE;
    }
};

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
__global__ void pick_random_kernel(
    IndexBasedRandomPicker<GPUUsage>* random_picker,
    int start,
    int end,
    bool prioritize_end,
    int* picked_value, curandState* rand_states);
#endif

#endif //INDEXBASEDRANDOMPICKER_H
