#ifndef TEST_RUN_UTILS_H
#define TEST_RUN_UTILS_H

#include <iostream>
#include <filesystem>
#include <regex>
#include <algorithm>

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

inline std::tuple<std::vector<int>, std::vector<int>, std::vector<int64_t>>
convert_edge_tuples_to_components(const std::vector<std::tuple<int, int, int64_t>>& edges) {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    // Reserve space to avoid reallocations
    sources.reserve(edges.size());
    targets.reserve(edges.size());
    timestamps.reserve(edges.size());

    // Extract components from each tuple
    for (const auto& edge : edges) {
        sources.push_back(std::get<0>(edge));
        targets.push_back(std::get<1>(edge));
        timestamps.push_back(std::get<2>(edge));
    }

    return std::make_tuple(sources, targets, timestamps);
}

std::vector<std::string> get_sorted_data_files(const std::string& base_path) {
    std::vector<std::string> file_paths;
    std::regex pattern("processed_data_(\\d+)\\.csv");

    try {
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            std::string filename = entry.path().filename().string();
            std::smatch matches;

            // Check if the filename matches our pattern
            if (std::regex_match(filename, matches, pattern)) {
                file_paths.push_back(entry.path().string());
            }
        }

        // Sort files based on the index number
        std::sort(file_paths.begin(), file_paths.end(),
            [&pattern](const std::string& a, const std::string& b) {
                std::smatch matches_a, matches_b;
                std::regex_search(a, matches_a, pattern);
                std::regex_search(b, matches_b, pattern);

                const int index_a = std::stoi(matches_a[1].str());
                const int index_b = std::stoi(matches_b[1].str());

                return index_a < index_b;
            });

    } catch (const std::exception& e) {
        std::cerr << "Error finding files: " << e.what() << std::endl;
    }

    return file_paths;
}

#endif //TEST_RUN_UTILS_H
