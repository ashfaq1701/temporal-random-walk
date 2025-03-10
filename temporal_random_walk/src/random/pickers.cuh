#ifndef PICKERS_H
#define PICKERS_H

#include <cstddef>
#include <curand_kernel.h>
#include "../common/macros.cuh"

HOST int pick_random_linear_host(int start, int end, bool prioritize_end);
DEVICE int pick_random_linear_device(int start, int end, bool prioritize_end, curandState* rand_state);

HOST int pick_random_exponential_index_host(int start, int end, bool prioritize_end);
DEVICE int pick_random_exponential_index_device(int start, int end, bool prioritize_end, curandState* rand_state);

HOST int pick_random_uniform_host(int start, int end);
DEVICE int pick_random_uniform_device(int start, int end, curandState* rand_state);

HOST int pick_random_exponential_weights_host(double* weights, size_t weights_size, size_t group_start, size_t group_end);
DEVICE int pick_random_exponential_weights_device(double* weights, size_t weights_size, size_t group_start, size_t group_end, curandState* rand_state);

#endif // PICKERS_H
