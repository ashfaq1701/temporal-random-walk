#ifndef NODE_GROUPED_KERNELS_CUH
#define NODE_GROUPED_KERNELS_CUH

// Umbrella header for the NODE_GROUPED kernel family. Callers that need
// every kernel (e.g. the dispatcher) include this one; callers that want
// a narrower subset (e.g. a future test that exercises only the block
// tier) can include the specific kernels/*.cuh directly.

#include "kernels/per_walk.cuh"   // step philox helper, pick_start_edges_kernel,
                                   // advance_one_walk, node_grouped_solo_kernel,
                                   // per_walk_step_kernel, reverse_walks_kernel.
#include "kernels/coop_warp.cuh"   // node_grouped_warp_smem_kernel,
                                   //   node_grouped_warp_global_kernel.
#include "kernels/coop_block.cuh"  // node_grouped_block_smem_kernel,
                                   //   node_grouped_block_global_kernel.

#endif // NODE_GROUPED_KERNELS_CUH
