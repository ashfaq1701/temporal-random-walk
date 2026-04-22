#ifndef NODE_GROUPED_KERNELS_CUH
#define NODE_GROUPED_KERNELS_CUH

// Umbrella header for the NODE_GROUPED kernel family. Callers that need
// every kernel (e.g. the dispatcher) include this one; callers that want
// a narrower subset (e.g. a future test that exercises only the block
// tier) can include the specific kernels/*.cuh directly.

#include "kernels/per_walk.cuh"   // step_philox, pick_start_edges, advance_one_walk,
                                   // node_grouped_solo_kernel, reverse_walks_kernel.
#include "kernels/coop_warp.cuh"   // warp_smem (real body, task 10) + warp_global
                                   // (real body, task 11).
#include "kernels/coop_block.cuh"  // block_smem (real body, task 8) + block_global
                                   // (real body, task 9).

#endif // NODE_GROUPED_KERNELS_CUH
