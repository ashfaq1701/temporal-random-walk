#ifndef TEST_RUN_UTILS_H
#define TEST_RUN_UTILS_H

#include <iostream>

inline void print_temporal_random_walks_with_times(
    const WalkSet& walk_set,
    const size_t n = 10)
{
    size_t count = 0;
    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end() && count < n; ++it, ++count) {
        const auto& walk = *it;

        std::cout << "Length: " << walk.size() << ", Walk: ";
        for (const auto& step : walk) {
            std::cout << "(" << step.node << ", " << step.timestamp << "), ";
        }
        std::cout << std::endl;
    }
}

inline double get_average_walk_length(const WalkSet& walk_set) {
    size_t total_walks = 0;
    size_t total_length = 0;

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        total_length += walk.size();
        total_walks++;
    }

    return total_walks > 0 ?
        static_cast<double>(total_length) / static_cast<double>(total_walks) : 0.0;
}

#endif //TEST_RUN_UTILS_H
