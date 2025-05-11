#ifndef FUNCTORS_H
#define FUNCTORS_H

#include "macros.cuh"

struct TimestampCompare {
    const int64_t* timestamps;

    HOST DEVICE explicit TimestampCompare(const int64_t* ts) : timestamps(ts) {}

    HOST DEVICE bool operator()(size_t i, size_t j) const {
        return timestamps[i] < timestamps[j];
    }
};

#endif // FUNCTORS_H
