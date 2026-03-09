#include <gtest/gtest.h>
#include <cmath>

#include "test_utils.h"
#include "../src/proxies/TemporalRandomWalk.cuh"


template<typename T>
class TimescaleBoundedTemporalRandomWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_random_walk = std::make_unique<TemporalRandomWalk>(true, T::value, -1, true, false, 10.0);

        // Note: Each directional edge has occured once, for checking unambigous edge features.
        temporal_random_walk->add_multiple_edges({
            {1, 2, 100},
            {2, 3, 101},
            {3, 5, 101},
            {1, 5, 110},
            {2, 5, 120},
            {2, 4, 110},
            {3, 4, 130},
            {4, 5, 130},
            {4, 1, 140},
            {5, 1, 140},
            {3, 1, 150},
            {2, 1, 150},
            {5, 3, 160},
            {3, 2, 160},
            {4, 2, 170}
        });

        // Note: Each feature set is unique.
        std::vector<float> edge_feats = {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
            0.7f, 0.8f, 0.9f,
            1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f,
            1.6f, 1.7f, 1.8f,
            1.9f, 2.0f, 2.1f,
            2.2f, 2.3f, 2.4f,
            2.5f, 2.6f, 2.7f,
            2.8f, 2.9f, 3.0f,
            3.1f, 3.2f, 3.3f,
            3.4f, 3.5f, 3.6f,
            3.7f, 3.8f, 3.9f,
            4.0f, 4.1f, 4.2f,
            4.3f, 4.4f, 4.5f
        };

        size_t feature_dim = 3;
    }

    std::unique_ptr<TemporalRandomWalk> temporal_random_walk;
};