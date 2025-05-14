#ifndef COMPARATORS_H
#define COMPARATORS_H

#include "../data/structs.cuh"
#include "macros.cuh"

struct TimestampComparator {
    int64_t* timestamps;

    // Constructor to initialize the timestamps pointer
    explicit TimestampComparator(int64_t* ts) : timestamps(ts) {}

    // Comparison operator
    DEVICE bool operator()(const int a, const int b) const {
        return timestamps[a] < timestamps[b];
    }
};

#endif // COMPARATORS_H
