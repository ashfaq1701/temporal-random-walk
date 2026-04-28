#ifndef CONST_H
#define CONST_H

#include <cstdint>

constexpr double DEFAULT_TIMESCALE_BOUND = -1;
constexpr double DEFAULT_NODE2VEC_P = 1.0;
constexpr double DEFAULT_NODE2VEC_Q = 1.0;

// TemporalNode2Vec: per-(node, ts-group) candidate-set cap. Groups with
// edge-count <= K stay EXACT (loop visits every edge). Groups larger than
// K are approximated by K independent uniform-with-replacement samples;
// the resulting β-weighted selection is an unbiased Monte Carlo estimator
// of exact Node2Vec selection (variance scales as 1/K). This caps the
// per-step Node2Vec cost at O(K) regardless of hub size — required for
// dense graphs (e.g. delicious) where un-capped per-step cost makes
// FULL_WALK Node2Vec infeasible.
constexpr int K_NODE2VEC = 64;

constexpr int EMPTY_NODE_VALUE = -1;
constexpr int64_t EMPTY_TIMESTAMP_VALUE = -1;
constexpr int64_t EMPTY_EDGE_ID = -1;
constexpr uint64_t EMPTY_GLOBAL_SEED = UINT64_MAX;

constexpr bool DEFAULT_SHUFFLE_WALK_ORDER = true;

#endif // CONST_H
