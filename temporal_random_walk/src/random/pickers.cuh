#ifndef PICKERS_H
#define PICKERS_H

#include <cstddef>
#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif
#include "../common/macros.cuh"
#include "../data/enums.cuh"

namespace random_pickers {

    HOST DEVICE int pick_random_linear(int start, int end, bool prioritize_end, double rand_number);

    HOST DEVICE int pick_random_exponential_index(int start, int end, bool prioritize_end, double rand_number);

    HOST DEVICE int pick_random_uniform(int start, int end, double rand_number);

    HOST int pick_random_exponential_weights_host(double* weights, size_t weights_size, size_t group_start, size_t group_end, double rand_number);

    #ifdef HAS_CUDA

    DEVICE int pick_random_exponential_weights_device(double* weights, size_t weights_size, size_t group_start, size_t group_end, double rand_number);

    #endif

    HOST DEVICE bool is_index_based_picker(RandomPickerType picker_type);

    HOST DEVICE int pick_using_index_based_picker(RandomPickerType random_picker, int start, int end, bool prioritize_end, double rand_number);

    HOST int pick_using_weight_based_picker_host(RandomPickerType random_picker, double* weights, size_t weights_size, size_t group_start, size_t group_end, double rand_number);

    #ifdef HAS_CUDA

    DEVICE int pick_using_weight_based_picker_device(RandomPickerType random_picker, double* weights, size_t weights_size, size_t group_start, size_t group_end, double rand_number);

    #endif
}

#endif // PICKERS_H
