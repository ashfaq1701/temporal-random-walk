#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../data/enums.h"
#include "../cuda_common/types.cuh"
#include "../cuda_common/macros.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPicker final : public RandomPicker<GPUUsage>
{
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);

    [[nodiscard]] HOST int pick_random_host(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);

    #ifdef HAS_CUDA
    [[nodiscard]] DEVICE int pick_random_device(
        const double* cumulative_weights_ptr,
        size_t weights_size,
        int group_start,
        int group_end,
        curandState* rand_state);
    #endif

    int HOST DEVICE get_picker_type() override
    {
        return WEIGHT_BASED_PICKER_TYPE;
    }

    #ifdef HAS_CUDA
    RandomPicker<GPUUsage>* to_device_ptr() override;
    #endif
};

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
__global__ void pick_random_kernel(
    WeightBasedRandomPicker<GPUUsage>* random_picker,
    const double* cumulative_weights_ptr,
    size_t weights_size,
    int group_start,
    int group_end,
    int* picked_value,
    curandState* rand_states);
#endif

#endif //WEIGHTBASEDRANDOMPICKER_H
