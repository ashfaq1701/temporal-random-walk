#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <sstream>
#include <string>
#include <fstream>

inline std::vector<std::tuple<int, int, int64_t>> read_edges_from_csv(const std::string& filename, const int row_count = -1, char delim=',') {
    std::ifstream file(filename);
    std::vector<std::tuple<int, int, int64_t>> edges;
    std::string line;

    std::getline(file, line);

    int current_row_count = 0;

    while (std::getline(file, line)) {
        if (row_count != -1 && current_row_count == row_count) {
            break;
        }

        std::stringstream ss(line);
        std::string u_str, i_str, t_str;

        std::getline(ss, u_str, delim);
        std::getline(ss, i_str, delim);
        std::getline(ss, t_str, delim);
        edges.emplace_back(std::stoi(u_str), std::stoi(i_str), std::stoll(t_str));

        current_row_count += 1;
    }

    return edges;
}

#endif // TEST_UTILS_H
