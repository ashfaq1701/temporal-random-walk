// WeightBasedRandomPickerGPU.cuh
#ifndef WEIGHTBASEDRANDOMPICKERGPU_CUH
#define WEIGHTBASEDRANDOMPICKERGPU_CUH

#include "RandomPicker.h"

#include "../data/enums.h"
#include "../cuda_common/types.cuh"


template<GPUUsageMode GPUUsage>
class WeightBasedRandomPickerGPU final : public RandomPicker {
private:
    double* d_random_val{};  // Persistent device memory

public:
    WeightBasedRandomPickerGPU();

    ~WeightBasedRandomPickerGPU() override;

    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);

    int get_picker_type() override
    {
        return WEIGHT_BASED_PICKER_TYPE;
    }
};

#endif //WEIGHTBASEDRANDOMPICKERGPU_CUH
