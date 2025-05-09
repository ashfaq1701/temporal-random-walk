#ifndef UTILS_OMP_H
#define UTILS_OMP_H

#include <omp.h>
#include <vector>

inline void parallel_prefix_sum(const std::vector<int>& input, std::vector<int>& output) {
    const size_t n = input.size();
    int num_threads = 0;
    std::vector<int> thread_sums;

    output.resize(n);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, 0);
        }

        int local_sum = 0;
        size_t start, end;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += input[i];
            output[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            output[i] -= input[i]; // convert to exclusive
            output[i] += thread_sums[tid];
        }
    }
}

template <typename T>
void parallel_inclusive_scan(std::vector<T>& data) {
    static_assert(std::is_arithmetic<T>::value, "parallel_inclusive_scan requires a numeric type");

    const size_t n = data.size();
    if (n == 0) return;

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));
        }

        T local_sum = T(0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += data[i];
            data[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            data[i] += thread_sums[tid];
        }
    }
}

#endif // UTILS_OMP_H
