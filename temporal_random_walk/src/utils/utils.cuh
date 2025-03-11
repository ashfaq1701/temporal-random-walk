#ifndef UTILS_H
#define UTILS_H

#include "../common/macros.cuh"

inline HOST DEVICE int pick_other_number(const int first, const int second, const int picked_number) {
    return (picked_number == first) ? second : first;
}

#endif // UTILS_H
