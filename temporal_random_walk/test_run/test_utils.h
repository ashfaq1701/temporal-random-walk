#ifndef TEST_RUN_UTILS_H
#define TEST_RUN_UTILS_H

#include <iostream>

inline void print_temporal_random_walks_with_times(
    const std::vector<std::tuple<std::vector<int>, std::vector<int64_t>>>& walks_with_times,
    size_t n = 10)
{
    size_t count = 0;
    for (const auto& walk : walks_with_times) {
        if (count++ >= n) break;

        const auto& nodes = std::get<0>(walk);
        const auto& timestamps = std::get<1>(walk);

        std::cout << "Length: " << nodes.size() << ", Walk: ";
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::cout << "(" << nodes[i] << ", " << timestamps[i] << "), ";
        }
        std::cout << std::endl;
    }
}

inline double get_average_walk_length(const std::vector<std::tuple<std::vector<int>, std::vector<int64_t>>>& walks) {
    if (walks.empty()) return 0.0;

    size_t total_length = 0;

    for (const auto& walk : walks) {
        const auto& nodes = std::get<0>(walk);
        total_length += nodes.size();
    }

    return static_cast<double>(total_length) / static_cast<double>(walks.size());
}

#endif //TEST_RUN_UTILS_H
