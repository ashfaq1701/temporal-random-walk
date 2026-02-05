#ifndef PICKERS_H
#define PICKERS_H

#include <cstddef>
#include <cmath>
#ifdef HAS_CUDA
#include <curand_kernel.h>
#include <cuda/std/__algorithm/lower_bound.h>
#endif

#include "../common/macros.cuh"
#include "../data/enums.cuh"

template <RandomPickerType T>
struct is_index_based_picker_trait {
    static constexpr bool value =
        T == RandomPickerType::Linear ||
        T == RandomPickerType::Uniform ||
        T == RandomPickerType::ExponentialIndex ||
        T == RandomPickerType::TEST_FIRST ||
        T == RandomPickerType::TEST_LAST;
};

namespace random_pickers {

    HOST DEVICE inline int pick_random_linear(
        const int start,
        const int end,
        const bool prioritize_end,
        const double rand_number) {
        if (start >= end) {
            return -1;
        }

        const int len_seq = end - start;

        // For a sequence of length n, weights form an arithmetic sequence
        // When prioritizing end: weights are 1, 2, 3, ..., n
        // When prioritizing start: weights are n, n-1, n-2, ..., 1
        // Sum of arithmetic sequence = n(a1 + an)/2 = n(n+1)/2
        const double total_weight = static_cast<double>(len_seq) *
                                       (static_cast<double>(len_seq) + 1.0) / 2.0;

        // Generate random value in [0, total_weight)
        const double scaled_random_value = total_weight * rand_number;

        // For both cases, we solve quadratic equation i² + i - 2r = 0
        // where r is our random value (or transformed random value)
        // Using quadratic formula: (-1 ± √(1 + 8r))/2
        const double discriminant = 1.0 + 8.0 * scaled_random_value;
        const double root = (-1.0 + std::sqrt(discriminant)) / 2.0;
        const int index = static_cast<int>(std::floor(root));

        if (prioritize_end) {
            // For prioritize_end=true, larger indices should have higher probability
            return start + std::min(index, len_seq - 1);
        } else {
            // For prioritize_end=false, we reverse the index to give
            // higher probability to smaller indices
            const int revered_index = len_seq - 1 - index;
            return start + std::max(0, revered_index);
        }
    }

    HOST DEVICE inline int pick_random_exponential_index(
        const int start,
        const int end,
        const bool prioritize_end,
        const double rand_number) {
        if (start >= end) {
            return -1;
        }

        const int len_seq = end - start;

        double k;
        if (len_seq < 710) {
            // Inverse CDF formula,
            // k = ln(1 + u * (e^len seq − 1)) − 1
            k = log1p(rand_number * expm1(len_seq)) - 1;
        } else {
            // Inverse CDF approximation for large len_seq,
            // k = len_seq + ln(u) − 1
            k = len_seq + std::log(rand_number) - 1;
        }

        // Due to rounding, the trailing "-1" in the inverse CDF formula causes error.
        // To compensate for this we add 1 with k.
        // And bound the results within limits.
        const int rounded_index = std::max(0, std::min(static_cast<int>(k + 1), len_seq - 1));

        if (prioritize_end) {
            return start + rounded_index;
        } else {
            return start + (len_seq - 1 - rounded_index);
        }
    }

    HOST DEVICE inline int pick_random_uniform(const int start, const int end, const double rand_number) {
        if (start >= end) {
            return -1;
        }

        return start + static_cast<int>(rand_number * (end - start));
    }

    HOST inline int pick_random_exponential_weights_host(
        double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number) {
        if (group_start >= group_end || group_end > weights_size) {
            return -1;
        }

        // Get start and end sums
        double start_sum = 0.0;
        if (group_start > 0) {
            start_sum = weights[group_start - 1];
        }
        const double end_sum = weights[group_end - 1];

        if (end_sum < start_sum) {
            return -1;
        }

        const double random_val = start_sum + random_number * (end_sum - start_sum);

        return static_cast<int>(std::lower_bound(
                weights + group_start,
                weights + group_end,
                random_val) - weights);
    }

    #ifdef HAS_CUDA

    DEVICE inline int pick_random_exponential_weights_device(
        double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number) {
        if (group_start >= group_end || group_end > weights_size) {
            return -1;
        }

        // Get start and end sums
        double start_sum = 0.0;
        if (group_start > 0) {
            start_sum = weights[group_start - 1];
        }
        const double end_sum = weights[group_end - 1];

        if (end_sum < start_sum) {
            return -1;
        }

        const double random_val = start_sum + random_number * (end_sum - start_sum);

        return static_cast<int>(cuda::std::lower_bound(
                weights + group_start,
                weights + group_end,
                random_val) - weights);
    }

    #endif

    HOST DEVICE inline bool is_index_based_picker(const RandomPickerType picker_type) {
        switch (picker_type) {
            case RandomPickerType::Linear:
            case RandomPickerType::Uniform:
            case RandomPickerType::ExponentialIndex:
            case RandomPickerType::TEST_FIRST:
            case RandomPickerType::TEST_LAST:
                return true;
            default:
                return false;
        }
    }

    template <RandomPickerType T>
    inline constexpr bool is_index_based_picker_v = is_index_based_picker_trait<T>::value;

    HOST DEVICE inline int pick_using_index_based_picker(
        const RandomPickerType random_picker,
        const int start,
        const int end,
        const bool prioritize_end,
        const double random_number) {
        switch (random_picker) {
            case RandomPickerType::Linear:
                return pick_random_linear(start, end, prioritize_end, random_number);
            case RandomPickerType::ExponentialIndex:
                return pick_random_exponential_index(start, end, prioritize_end, random_number);
            case RandomPickerType::Uniform:
                return pick_random_uniform(start, end, random_number);
            // ONLY FOR TEST
            case RandomPickerType::TEST_FIRST:
                return start;
            case RandomPickerType::TEST_LAST:
                return end - 1;
            default:
                return -1;
        }
    }

    HOST inline int pick_using_weight_based_picker_host(
        const RandomPickerType random_picker,
        double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number) {
        if (random_picker != RandomPickerType::ExponentialWeight &&
            random_picker != RandomPickerType::TemporalNode2Vec) {
            return -1;
        }

        return pick_random_exponential_weights_host(weights, weights_size, group_start, group_end, random_number);
    }

    #ifdef HAS_CUDA

    DEVICE inline int pick_using_weight_based_picker_device(
        const RandomPickerType random_picker,
        double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number) {
        if (random_picker != RandomPickerType::ExponentialWeight &&
            random_picker != RandomPickerType::TemporalNode2Vec) {
            return -1;
        }

        return pick_random_exponential_weights_device(weights, weights_size, group_start, group_end, random_number);
    }

    #endif
}

#endif // PICKERS_H
