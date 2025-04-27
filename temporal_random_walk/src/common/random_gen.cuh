#ifndef RANDOM_GEN_H
#define RANDOM_GEN_H

#include <cstddef>

double* generate_n_random_numbers_cpu(size_t n);

#ifdef HAS_CUDA
double* generate_n_random_numbers_gpu(size_t n);
#endif

#endif // RANDOM_GEN_H
