#include "pickers.cuh"

#include <cmath>
#include <stdexcept>

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#endif

#include "../utils/random.cuh"

HOST DEVICE int random_pickers::pick_random_linear(const int start, const int end, const bool prioritize_end, const double rand_number) {
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

HOST DEVICE int random_pickers::pick_random_exponential_index(const int start, const int end, const bool prioritize_end, const double rand_number) {
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

HOST DEVICE int random_pickers::pick_random_uniform(const int start, const int end, const double rand_number) {
    if (start >= end) {
        return -1;
    }

    return start + static_cast<int>(rand_number * (end - start));
}

HOST int random_pickers::pick_random_exponential_weights_host(double* weights, const size_t weights_size, const size_t group_start, const size_t group_end, const double random_number) {
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

DEVICE int random_pickers::pick_random_exponential_weights_device(double* weights, const size_t weights_size, const size_t group_start, const size_t group_end, const double random_number) {
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

HOST DEVICE bool random_pickers::is_index_based_picker(const RandomPickerType picker_type) {
    return picker_type == RandomPickerType::Linear || picker_type == RandomPickerType::Uniform ||
        picker_type == RandomPickerType::ExponentialIndex ||
            // ONLY FOR TESTS
            picker_type == RandomPickerType::TEST_FIRST || picker_type == RandomPickerType::TEST_LAST;
}

HOST DEVICE int random_pickers::pick_using_index_based_picker(const RandomPickerType random_picker, const int start, const int end, const bool prioritize_end, const double random_number) {
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

HOST int random_pickers::pick_using_weight_based_picker_host(const RandomPickerType random_picker, double* weights, const size_t weights_size, const size_t group_start, const size_t group_end, const double random_number) {
    if (random_picker != RandomPickerType::ExponentialWeight) {
        return -1;
    }

    return pick_random_exponential_weights_host(weights, weights_size, group_start, group_end, random_number);
}

#ifdef HAS_CUDA

DEVICE int random_pickers::pick_using_weight_based_picker_device(const RandomPickerType random_picker, double* weights, const size_t weights_size, const size_t group_start, const size_t group_end, const double random_number) {
    if (random_picker != RandomPickerType::ExponentialWeight) {
        return -1;
    }

    return pick_random_exponential_weights_device(weights, weights_size, group_start, group_end, random_number);
}

#endif
