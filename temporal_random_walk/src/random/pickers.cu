#include "pickers.cuh"

#include <cmath>
#include <stdexcept>
#include <cuda/std/__algorithm/lower_bound.h>

#include "../utils/random.cuh"

HOST int random_pickers::pick_random_linear_host(int start, int end, bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // For a sequence of length n, weights form an arithmetic sequence
    // When prioritizing end: weights are 1, 2, 3, ..., n
    // When prioritizing start: weights are n, n-1, n-2, ..., 1
    // Sum of arithmetic sequence = n(a1 + an)/2 = n(n+1)/2
    const double total_weight = static_cast<double>(len_seq) *
                                   (static_cast<double>(len_seq) + 1.0) / 2.0;

    // Generate random value in [0, total_weight)
    const double random_value = generate_random_value_host(0.0, total_weight);

    // For both cases, we solve quadratic equation i² + i - 2r = 0
    // where r is our random value (or transformed random value)
    // Using quadratic formula: (-1 ± √(1 + 8r))/2
    const double discriminant = 1.0 + 8.0 * random_value;
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

HOST int random_pickers::pick_random_exponential_index_host(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    const double u = generate_random_value_host(0.0, 1.0);

    double k;
    if (len_seq < 710) {
        // Inverse CDF formula,
        // k = ln(1 + u * (e^len seq − 1)) − 1
        k = log1p(u * expm1(len_seq)) - 1;
    } else {
        // Inverse CDF approximation for large len_seq,
        // k = len_seq + ln(u) − 1
        k = len_seq + std::log(u) - 1;
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

HOST int random_pickers::pick_random_uniform_host(int start, int end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    return generate_random_int_host(start, end - 1);
}

HOST int random_pickers::pick_random_exponential_weights_host(double* weights, size_t weights_size, size_t group_start, size_t group_end) {
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

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value_host(start_sum, end_sum);
    return static_cast<int>(std::lower_bound(
            weights + group_start,
            weights + group_end,
            random_val) - weights);
}

DEVICE int random_pickers::pick_random_linear_device(int start, int end, bool prioritize_end, curandState* rand_state) {
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
    const double random_value = generate_random_value_device(0.0, total_weight, rand_state);

    // For both cases, we solve quadratic equation i² + i - 2r = 0
    // where r is our random value (or transformed random value)
    // Using quadratic formula: (-1 ± √(1 + 8r))/2
    const double discriminant = 1.0 + 8.0 * random_value;
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

DEVICE int random_pickers::pick_random_exponential_index_device(int start, int end, bool prioritize_end, curandState* rand_state) {
    if (start >= end) {
        return -1;
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    const double u = generate_random_value_device(0.0, 1.0, rand_state);

    double k;
    if (len_seq < 710) {
        // Inverse CDF formula,
        // k = ln(1 + u * (e^len seq − 1)) − 1
        k = log1p(u * expm1(len_seq)) - 1;
    } else {
        // Inverse CDF approximation for large len_seq,
        // k = len_seq + ln(u) − 1
        k = len_seq + std::log(u) - 1;
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

DEVICE int random_pickers::pick_random_uniform_device(int start, int end, curandState* rand_state) {
    if (start >= end) {
        return -1;
    }

    return generate_random_int_device(start, end - 1, rand_state);
}

DEVICE int random_pickers::pick_random_exponential_weights_device(double* weights, size_t weights_size, size_t group_start, size_t group_end, curandState* rand_state) {
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

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value_device(start_sum, end_sum, rand_state);
    return static_cast<int>(cuda::std::lower_bound(
            weights + group_start,
            weights + group_end,
            random_val) - weights);
}
