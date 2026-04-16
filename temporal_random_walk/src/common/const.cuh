#ifndef CONST_H
#define CONST_H

#include <cstdint>

constexpr double DEFAULT_TIMESCALE_BOUND = -1;
constexpr double DEFAULT_NODE2VEC_P = 1.0;
constexpr double DEFAULT_NODE2VEC_Q = 1.0;

constexpr int EMPTY_NODE_VALUE = -1;
constexpr int64_t EMPTY_TIMESTAMP_VALUE = -1;
constexpr int64_t EMPTY_EDGE_ID = -1;
constexpr uint64_t EMPTY_GLOBAL_SEED = UINT64_MAX;

constexpr bool DEFAULT_SHUFFLE_WALK_ORDER = true;

#endif // CONST_H
