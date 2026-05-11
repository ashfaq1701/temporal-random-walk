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

// Temporal Node2Vec rejection-sampling retry cap.  Per-hop proposal is
// drawn from the static-exp distribution and accepted with prob β/β_max;
// expected retries ≤ β_max / E[β] (typically 1–2 for benign p, q).  On
// overflow the last proposal is accepted defensively, so degenerate
// (p ≪ 1 or q ≫ 1) configurations stay bounded.
constexpr int NODE2VEC_MAX_RETRIES = 16;

#endif // CONST_H
