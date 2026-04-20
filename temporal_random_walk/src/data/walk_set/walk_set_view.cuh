#ifndef DATA_WALK_SET_WALK_SET_VIEW_CUH
#define DATA_WALK_SET_WALK_SET_VIEW_CUH

#include <cstddef>
#include <cstdint>

#include "../../common/macros.cuh"
#include "../../common/const.cuh"

/**
 * WalkSetView — POD view of a WalkSetDevice's underlying buffers, passed
 * to kernels by value. Kernels write hops via add_hop and optionally
 * reverse walks via reverse_walk. The view itself owns nothing.
 *
 * Replaces the old to_device_ptr path (which allocated a device mirror of
 * the struct). A kernel now takes `WalkSetView view` as a regular by-value
 * parameter — no cudaMalloc, no cudaMemcpy.
 */
struct WalkSetView {
    int*     nodes;
    int64_t* timestamps;
    size_t*  walk_lens;
    int64_t* edge_ids;

    size_t num_walks;
    size_t max_len;
    int    walk_padding_value;

    /**
     * Append a hop to walk `walk_idx`. hop_pos is the current length of the
     * walk, which becomes the index of the newly written hop.
     *
     * edge_id corresponds to the transition INTO this hop:
     *   - hop 0 has no incoming edge (edge_id field skipped).
     *   - hop i (i >= 1) stores edge_id at edge_ids[walk_idx * (max_len-1) + (i-1)].
     */
    HOST DEVICE void add_hop(
        const int walk_idx,
        const int node,
        const int64_t timestamp,
        const int64_t edge_id = EMPTY_EDGE_ID) const {

        const size_t hop_pos = walk_lens[walk_idx];
        const size_t hop_offset =
            static_cast<size_t>(walk_idx) * max_len + hop_pos;

        nodes[hop_offset] = node;
        timestamps[hop_offset] = timestamp;

        if (edge_id != EMPTY_EDGE_ID && hop_pos > 0) {
            const size_t edge_base =
                static_cast<size_t>(walk_idx) * (max_len - 1);
            edge_ids[edge_base + (hop_pos - 1)] = edge_id;
        }

        walk_lens[walk_idx] = hop_pos + 1;
    }

    /**
     * Reverse the walk at walk_idx in place (nodes + timestamps + edge_ids).
     * Used for backward-in-time walks so the caller always sees a
     * chronologically forward sequence.
     */
    HOST DEVICE void reverse_walk(const int walk_idx) const {
        const size_t walk_length = walk_lens[walk_idx];
        if (walk_length <= 1) return;

        const size_t node_start = static_cast<size_t>(walk_idx) * max_len;
        const size_t node_end   = node_start + walk_length - 1;

        for (size_t i = 0; i < walk_length / 2; ++i) {
            const size_t l = node_start + i;
            const size_t r = node_end   - i;

            const int tmp_n = nodes[l]; nodes[l] = nodes[r]; nodes[r] = tmp_n;
            const int64_t tmp_t = timestamps[l];
            timestamps[l] = timestamps[r];
            timestamps[r] = tmp_t;
        }

        const size_t edge_count = walk_length - 1;
        if (edge_count > 1) {
            const size_t edge_start =
                static_cast<size_t>(walk_idx) * (max_len - 1);
            const size_t edge_end = edge_start + edge_count - 1;
            for (size_t i = 0; i < edge_count / 2; ++i) {
                const size_t l = edge_start + i;
                const size_t r = edge_end   - i;
                const int64_t tmp_e = edge_ids[l];
                edge_ids[l] = edge_ids[r];
                edge_ids[r] = tmp_e;
            }
        }
    }
};

#endif // DATA_WALK_SET_WALK_SET_VIEW_CUH
