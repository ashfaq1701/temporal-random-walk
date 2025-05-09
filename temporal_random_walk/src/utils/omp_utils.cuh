#ifndef UTILS_OMP_H
#define UTILS_OMP_H

#include <omp.h>
#include <vector>

template <typename T>
void parallel_prefix_sum(const T* input, T* output, size_t n) {
    static_assert(std::is_arithmetic_v<T>, "parallel_prefix_sum_ptr requires a numeric type");

    if (n == 0) return;

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));  // thread_sums[0] = 0
        }

        T local_sum = T(0);

        // First pass: local inclusive scan
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += input[i];
            output[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier

        // Scan over thread partial sums
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        // Convert to exclusive and apply offset
        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            output[i] -= input[i];  // convert inclusive to exclusive
            output[i] += offset;
        }
    }
}

template <typename T>
void parallel_exclusive_scan(const T* input, T* output_with_offset, const size_t n) {
    static_assert(std::is_arithmetic_v<T>, "T must be numeric");
    if (n == 0) {
        output_with_offset[0] = 0;
        return;
    }

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));  // thread_sums[0] = 0
        }

        T local_sum = 0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += input[i];
            output_with_offset[i + 1] = local_sum; // write to output[1..n]
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            output_with_offset[i + 1] += offset;
        }
    }

    output_with_offset[0] = 0; // Set base offset
}

template <typename T>
void parallel_inclusive_scan(T* data, size_t n) {
    static_assert(std::is_arithmetic_v<T>, "parallel_inclusive_scan_ptr requires a numeric type");
    if (n == 0) return;

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));
        }

        T local_sum = T(0);

        // First pass: local inclusive scan per thread
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += data[i];
            data[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier

        // Prefix sum over thread_sums (serial, small)
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        // Second pass: apply offset to each thread’s data
        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            data[i] += offset;
        }
    }
}

#endif // UTILS_OMP_H
