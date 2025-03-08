#ifndef RANDOMPICKER_H
#define RANDOMPICKER_H

#include "../data/enums.h"
#include "../cuda_common/macros.cuh"

constexpr int INDEX_BASED_PICKER_TYPE = 1;
constexpr int WEIGHT_BASED_PICKER_TYPE = 2;

template<GPUUsageMode GPUUsage>
class RandomPicker
{
public:
    virtual ~RandomPicker() = default;

    virtual HOST DEVICE int get_picker_type() = 0;

    #ifdef HAS_CUDA
    virtual RandomPicker<GPUUsage>* to_device_ptr() = 0;
    #endif
};

#endif //RANDOMPICKER_H
