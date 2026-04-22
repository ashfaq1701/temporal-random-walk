# NODE_GROUPED walk kernel — design dossier

Scoped to the Phase-2 walk-sampling refactor on `feature/warp-collaboration`.
Durable context for the node-grouped pipeline: what it is, why it exists, and
what remains open.

## 1. Problem

The `FULL_WALK` kernel assigns one CUDA thread per walk for the walk's entire
life. Each thread, at every step, re-reads the adjacency metadata of *its own*
current node, runs a temporal-cutoff binary search, picks a group, samples an
edge, and advances. Three pathologies on skewed temporal graphs:

1. **No reuse across co-located walks.** Thousands of walks can share one hub
   node at a given step; each thread re-fetches the hub's per-node arrays from
   global memory.
2. **3-deep dependent-load chain in the binary search.** The comparator reads
   `node_ts_groups_offsets[probe] → sorted_idx`, then
   `node_ts_sorted_indices[sorted_idx] → edge_idx`, then
   `view.timestamps[edge_idx] → compared_ts`. Each probe cannot start until the
   prior probe's global load returns. Binary search does `log₂(G)` probes, all
   serialized.
3. **Warp divergence grows with step count.** Threads in a warp terminate at
   wildly different times; late steps run with most lanes idle.

The `NODE_GROUPED` path groups walks by current node per step and processes
each group cooperatively. Shared metadata is loaded once into shared memory;
the binary-search key sequence is pre-materialized, killing the dependent
chain.

## 2. Overall plan

Per step (intermediate steps 1..max-1, and constrained step 0) we run:

1. **Filter terminated walks** (`walk.nodes[walk_idx][last] == walk_padding_value`)
2. **Gather** surviving walks' current nodes
3. **Argsort** `(current_node, walk_idx)` pairs by current_node
4. **Run-length encode** to produce unique_nodes + per-node walk counts + per-node offsets
5. **Partition unique nodes by W** into three tiers: solo (W=1), warp
   (W ∈ [2, 255]), block (W ≥ 256)
6. **Partition each cooperative tier by G** into smem-fit vs global-fallback
7. **Expand block tier**: nodes with W > 8192 split into ⌈W/8192⌉ block-tasks
   of disjoint walk slices, each preloading the same panel independently
8. **Launch five kernels in sequence** over their pre-partitioned task lists:
   solo, warp-smem, warp-global, block-smem, block-global

Step 0 unconstrained (`start_node_ids[i] == -1` for all i) short-circuits to
the existing start-edge kernel — no current node exists to group on. Step 0
constrained flows through the same scheduler as intermediate steps.

Walks preserve their original `walk_idx` end-to-end. Sort, compaction, and RLE
carry `walk_idx` as the payload; every kernel writes `walk_set[walk_idx]`
directly without dense renumbering. Philox is keyed on `(base_seed, walk_idx,
step_philox_offset(step))` so tier routing is RNG-invariant.

## 3. Scheduling strategy & thresholds

Tier boundaries come from NextDoor (EuroSys 2021) and FlowWalker (VLDB 2024)
conventions adapted for walks-per-node clustering. Values in
`src/common/warp_coop_config.cuh`.

| Tier       | W range           | Cooperation unit | Basis |
|------------|-------------------|------------------|-------|
| solo       | W = 1             | 1 thread         | No reuse possible |
| warp-coop  | W ∈ [2, 255]      | 1 warp, 8 warps/block | Intra-warp stride loop over up to ⌈255/32⌉ = 8 rounds |
| block-coop | W ≥ 256           | 1 block, 256 threads | Stride loop in chunks of 256, up to 8192 walks per block |

**Per-block walk cap.** Power-law temporal graphs at Tempest scale (tens of
millions of walks) produce mega-hubs with W in the hundreds of thousands. A
single block processing all such walks would monopolize one SM while others
idle. Cap: `TRW_NODE_GROUPED_BLOCK_WALK_CAP = 8192`. Nodes with W > cap split
into ⌈W/cap⌉ block-tasks, each processing a disjoint walk slice. Each block
independently preloads the same panel; L2 absorbs the redundancy after the
first block pays DRAM.

Not a grid-cooperative mechanism. Each split block is fully independent — no
cross-block coordination, writes go to disjoint walk indices.

**Dispatch key.** `W` per unique node — the walk count in the current step's
(node, step) group. Not graph degree, not neighbor-list size. NextDoor's
"total neighbors to sample per transit" is the closest prior analogue.

**Pre-partitioned task lists, not guard-and-exit.** The scheduler emits five
disjoint task lists; each kernel is launched only over its own list. Kernels
do not receive walks they skip. This is deliberate — `guard-and-exit` wastes
launches and defeats the point of partitioning.

## 4. Shared-memory plan

### What goes in smem

Per unique node, keyed on **G** (distinct-timestamp-group count), not E (edge
count). The binary-search bottleneck scales with G, not E.

For index-based pickers (Uniform, Linear, ExponentialIndex), two arrays:
- `s_group_offsets[G]` — per-group offsets into the sorted-edge array (8 B/group)
- `s_first_ts[G]` — timestamp of the first edge of each group, pre-gathered
  through `node_ts_sorted_indices` to collapse the 3-deep dependent chain
  (8 B/group)

**Total: 16 B/group.**

For weighted pickers (ExponentialWeight, TemporalNode2Vec), add one array:
- `s_cum_weights[G]` — per-group cumulative weights for the weighted
  `lower_bound` (8 B/group)

**Total: 24 B/group.**

Not preloaded: the full edge list, per-edge timestamps, the final-edge
endpoint fetch. Final edge pick is one uncoalesced global read per walk —
unavoidable, not worth preloading.

### Smem budget

- **Static envelope:** 48 KB per block on every arch sm_70+ without opt-in.
- **Reserved:** ~4 KB for Philox state, control fields, alignment padding.
- **Usable panel:** **44 KB per block** (`TRW_NODE_GROUPED_SMEM_PANEL_BYTES`).
- **Per-warp slice:** 44 KB / 8 warps per block = 5.5 KB per warp.

### Derived G caps

| Tier  | Picker class | G cap |
|-------|--------------|-------|
| Block | Index        | **2800** |
| Block | Weighted     | **1800** |
| Warp  | Index        | **340**  |
| Warp  | Weighted     | **220**  |

### Fit test

Per unique node in a cooperative tier, compare G to the tier's cap:
- `G ≤ cap` → smem kernel variant (panel preload + stride loop over walks)
- `G > cap` → global kernel variant (no panel, direct binary search against
  global arrays, still cooperative stride loop over walks)

Binary. No tiling, no demotion to solo. The global-fallback path is a
cooperative kernel — all co-located walks share L2 residency of the per-node
arrays, which is cheaper than the baseline per-thread path that scatters
across nodes.

### Why 44 KB and not more

Opt-in dynamic smem via `cudaFuncSetAttribute` could push the block budget to
163 KB on A100 or 227 KB on H100, raising the G caps proportionally. Deferred
because:

1. **Occupancy cliff.** A100 drops from 2 blocks/SM to 1 at ~81 KB per block.
   H100 same transition at ~113 KB. Doubling the G cap rarely pays for half
   the occupancy.
2. **Portability.** 48 KB is the static cap on every target arch (sm_70
   through sm_90+). Static envelope is a single code path.
3. **Escape hatch remains.** A single `cudaFuncSetAttribute` call plus
   updated constants is enough to opt in later if measurement shows the
   overflow path hot (>20% of hub time).

### Why G, not E

The binary search runs over G-length arrays. The weighted `lower_bound` runs
over G-length cumulative weights. The only E-scale work is the single final
edge load per walk, which is not a search and not a candidate for smem.

On temporal graphs at Tempest scale, G is typically O(thousands) on hubs
(distinct timestamps per node), while E can be O(millions) (every appearance
of the node across the stream). Keying on G keeps the smem budget tractable
on hub nodes; keying on E would blow the budget on the first hub.

This is, to the best of the prior-art review, **not how any existing GPU
temporal-graph system indexes its smem**. TGL, TGLite, TGOpt, TEA all use
temporal-CSR treating neighbor lists as undifferentiated time-sorted runs.
The G-keyed panel is the mechanism-level novelty of this work.

## 5. Kernel matrix

| Kernel                                         | Tier        | smem panel | Dispatch list         |
|------------------------------------------------|-------------|------------|----------------------|
| `pick_start_edges_kernel`                      | unconstrained step 0 | n/a | global short-circuit |
| `node_grouped_solo_kernel`                     | solo        | n/a        | solo_walks            |
| `node_grouped_warp_smem_kernel`                | warp        | yes        | warp_smem_nodes       |
| `node_grouped_warp_global_kernel`              | warp        | no         | warp_global_nodes     |
| `node_grouped_block_smem_kernel`               | block       | yes        | block_smem_tasks      |
| `node_grouped_block_global_kernel`             | block       | no         | block_global_tasks    |
| `reverse_walks_kernel`                         | —           | n/a        | backward walks only   |

Every cooperative kernel is template-specialized on `<IsDirected, Forward,
EdgePickerType>` via the existing `dispatch_bool` / `dispatch_picker_type`
tag-dispatch helpers. No runtime branching inside kernel hot paths.

## 6. Where we are

Accurate snapshot of the tree on `feature/warp-collaboration`. This section is
the progress log; §5's kernel matrix is the target state.

**Done**
- Rename: `STEP_BASED` → `NODE_GROUPED` across enum, kernel file, dispatcher,
  tests, ablations.
- `src/common/warp_coop_config.cuh` — single source of truth for tier
  boundaries (`T_WARP=1`, `T_BLOCK=255`), launch shape (8 warps/block,
  256 threads/block), per-block walk cap (`BLOCK_WALK_CAP=8192`), smem panel
  budget (`SMEM_PANEL_BYTES=45056`, 44 KB), and the four derived G caps
  (block/index 2800, block/weighted 1800, warp/index 340, warp/weighted 220).
- Solo kernel (`node_grouped_solo_kernel`) functional for every picker ×
  directed/undirected × forward/backward combination. Start-edges kernel
  (`pick_start_edges_kernel`) keeps its name per §5 and handles both
  unconstrained and constrained step 0 today — the constrained case moves
  into the solo_walks list in task 5. Since the cooperative stubs were
  torn out in task 2 and solo now services every active walk, the
  previously vestigial `walk_to_group_size` group-size guards have been
  removed from both kernels; re-introduced as pre-partitioned task lists
  in task 5.
- Per-step pipeline extracted into
  `temporal_random_walk_node_grouped_scheduler.{cuh,cu}` as
  `NodeGroupedScheduler`. Stages: filter (`walk_alive_flags_kernel`) →
  compact (`cub_partition_flagged`) → `num_active` D2H readback → gather
  (`gather_last_nodes_kernel`) → sort (`cub_sort_pairs`) → RLE
  (`cub_run_length_encode`) → exclusive-scan → scatter
  (`scatter_walk_group_sizes_kernel`). Scheduler-internal helper kernels
  (iota, alive-flags, gather, scatter, zero) live in the scheduler's
  anonymous namespace — not exported from the per-walk kernel header.
  Dispatcher now just calls `scheduler.run_step(...)` per step and feeds
  the returned sorted-walk-idx list + device num_active pointer into the
  solo pick kernel.
- `DeviceArena` backs per-step scratch: `scheduler.run_step()` resets the
  arena at the top and acquires 11 slots (sort buffers, RLE buffers, device
  counters). Reset is a host-side offset reset; stream ordering keeps
  prior-step kernels safe on the same memory. Batch-persistent state
  (`iota_src_`, `walk_to_group_size_`) stays as `Buffer<int>` members.
  Growth happens at most once, during the first step, before any queued
  kernel reads arena memory.
- Pipeline extents driven by `num_active`: sort / RLE / scan / gather /
  scatter / pick all run on `host_num_active` read once per step via a D2H
  copy + stream sync on `trw->stream()` (CUB's `num_items` is host-side, no
  device-pointer overload exists). Filter and the `step_group_size`
  zero-init deliberately retain `num_walks` extent — the former reads every
  walk by design; the latter is indexed by original `walk_idx` and must clear
  slots from walks that terminated in prior steps.
- `walk_idx` preserved end-to-end: iota → `cub_partition_flagged` → sort-by-key
  values → scatter back by walk_idx. No dense renumbering.
- Unconstrained step-0 short-circuit: `if (!all_starts_unconstrained) { … }`
  skips sort/RLE/scatter for the all-`-1` case; solo start kernel services
  every walk through `get_edge_at_device` (global edge stream). All three
  entry points (`_for_all_nodes_cuda`, `_for_last_batch_cuda`, `_cuda`)
  thread the flag correctly.
- `TemporalNode2Vec` bypasses the node-grouped pipeline entirely: the
  dispatcher gates on picker type at entry and routes Node2Vec through
  `per_walk_step_kernel` (one thread per walk per step, no scheduler,
  no coop tiers). Per-walk `prev_node` bias makes a shared cooperative
  panel impossible by construction, so sort/RLE/W-partition/G-partition/
  expansion are all skipped. Dead walks no-op inside `advance_one_walk`
  via `is_node_active`, so the filter/compact stages aren't needed
  either. The four coop kernels now assume non-Node2Vec and don't carry
  any Node2Vec branches or `prev_node` reads.
- NVTX ranges on every pipeline stage (`NG step0 setup`, `NG step`,
  `NG filter alive`, `NG compact`, `NG num_active readback`, `NG gather`,
  `NG sort`, `NG RLE+scan`, `NG scatter`, `NG pick`, `NG reverse`) for
  nsys legibility.
- Structural parity harness in `test_node_grouped_parity.cpp` —
  `FULL_WALK` vs `NODE_GROUPED` across forward-constrained, unconstrained,
  and backward cases. Not bit-exact (Philox counter offsets differ between
  the two paths); asserts walk validity, slot-0 agreement, and mean-length
  ratio band.
- `WalkSetView.walk_padding_value` consumed in the filter kernel — no
  hardcoded `-1`.

**Open**
- Cooperative kernel scaffolds (`node_grouped_warp_smem_kernel`,
  `_warp_global_kernel`, `_block_smem_kernel`, `_block_global_kernel`)
  exist, each consuming a node-level task list
  (sorted_walk_idx + walk_starts + walk_counts). Task 5 wires solo / warp-smem
  / block-smem into their own task lists; tasks 8–11 replace each body with
  its tier-specific cooperative implementation. The shared per-walk helper
  is `advance_one_walk` (takes a concrete walk_idx_int; no list indexing).
- G-partition is live (task 6): each warp task splits into warp_smem or
  warp_global by `G <= TRW_NODE_GROUPED_G_CAP_WARP_{INDEX,WEIGHTED}`; same
  for block. The picker-class choice of cap is a runtime branch on
  `is_index_based_picker(edge_picker_type)`. G is read per-node as
  `count_ts_group_per_node[node+1] - count_ts_group_per_node[node]`; the
  caller resolves the directional variant of that array (forward →
  outbound; backward directed → inbound; backward undirected → outbound)
  before passing it to `run_step`.
- Block-task expansion is live (task 7): every `block_smem` /
  `block_global` entry from the G-partition is run through
  `expand_block_tasks_kernel`, which splits mega-hub nodes
  (`W > TRW_NODE_GROUPED_BLOCK_WALK_CAP` = 8192) into ⌈W/cap⌉ disjoint
  sub-tasks each carrying the same `node_id`. Each thread reserves its
  output range with a single `atomicAdd(counter, num_sub_tasks)` so no
  per-sub-task atomic is needed. Warp tier is skipped (W ≤ T_BLOCK = 255
  ≪ cap). The scaffold kernels iterate ≤ cap walks per thread now; real
  coop bodies (tasks 8/9) get at most 8192 walks per block-task.
- Block-smem and block-global cooperative bodies are live (tasks 8, 9).
  Both run one block per block-task, 256 threads per block. Block-smem
  cooperatively preloads s_group_offsets[G] and s_first_ts[G] into smem
  before the stride loop (kills the double-indirect load chain in the
  stage-1 binary search). Block-global — used when the tier's G exceeds
  TRW_NODE_GROUPED_G_CAP_BLOCK_* — skips the preload and binary-searches
  against global arrays via find_group_pos_slice's double-indirect
  fallback, but still cooperatively parallelizes walks across the 256
  threads of a block (L1/L2 residency shared across co-located walks).
  Both use a small static-smem header to broadcast node-level scalars.
  Node2Vec never reaches these kernels — the dispatcher gates it out.
- Directional-pointer resolution factored into
  core/node_grouped/kernels/common.cuh as `NodeDirPtrs` +
  `resolve_node_dir_ptrs<IsDirected, Forward>(view)`. Both block
  kernels call the helper once at task entry; tasks 10/11 reuse it.
- Warp-smem cooperative body is live (task 10). 8 warps per block, 256
  threads; each warp services one task against its own per-warp smem
  slice tiled into one flat dynamic-smem allocation. Lane 0 broadcasts
  the task header; lanes 0..31 cooperatively preload the G-sized panel
  (stride 32); lanes 0..31 stride through the task's walks. Only
  __syncwarp() is used — warps in a block run independent tasks and
  may diverge at the task-id guard in partial last blocks, so a
  block-wide barrier would deadlock. Binary search is per-lane against
  the warp's s_first_ts (fast-path comparator). Node2Vec never reaches
  this kernel — the dispatcher gates it out.
- Warp-global cooperative body is live (task 11). Same launch topology
  as warp-smem (8 warps/block, one task per warp), no panel preload —
  the tier services nodes whose G exceeds TRW_NODE_GROUPED_G_CAP_WARP_*
  and wouldn't fit in the per-warp slice. Binary search runs against
  the GLOBAL node_ts_groups_offsets slice via find_group_pos_slice's
  double-indirect fallback. A tiny per-warp static smem header
  (8 × (int[4] + 2 size_t) ≈ 256 B/block) lets lane 0 broadcast
  node-level scalars once so the other 31 lanes don't refetch from
  global on every walk. __syncwarp() only. Node2Vec never reaches this
  kernel — the dispatcher gates it out. Phase III (real cooperative
  bodies for all four tiers) is complete.

## 7. Architecture guardrails

Standing rules, imposed by the project owner, respected going forward.

- **Don't auto-kick builds or tests.** The owner signals when ready.
- **No half-finished implementations.** Either land a working kernel or a
  clearly-marked scaffold copy. No silently-broken code, no TODOs in landed
  mechanism code.
- **Preserve original `walk_idx` end-to-end.** Never renumber walks to a
  dense range mid-pipeline.
- **Read `walk_padding_value` from `WalkSetView`**, never hardcode -1.
- **Buffer-backed CUB scratch.** All temp storage is `Buffer<uint8_t>` or
  `DeviceArena`-allocated — no manual `cudaMalloc`/`cudaFree`.
- **Stream-aware everywhere.** All kernels, CUB calls, allocations on
  `trw->stream()` or the caller-provided stream.
- **Tag-dispatch for runtime booleans/enums.** Use `dispatch_bool` and
  `dispatch_picker_type`; do not branch inside kernel hot paths.
- **One responsibility per kernel.** Filter, gather, sort, RLE, scatter,
  partition, expand, pick — all separate. No mega-kernel.
- **Rename, don't shim.** No compatibility aliases.
- **`#ifdef HAS_CUDA` around CUDA-only code.** CPU build must compile and run
  full-walk paths.
- **No bloated functions.** If a function passes ~100 lines during this work,
  stop and decompose.
- **Don't weaken tests.** Distribution tests pass bit-for-bit against the
  baseline walk distribution. Fix code, not tests.
- **Commit per task boundary.** Each commit leaves the code buildable and
  testable.

## 8. Future tasks

Ordered so each task leaves the build green, the parity harness green, and
one acceptance boundary the owner can review in isolation. Phases I–II are
non-behavioral or distribution-preserving (bodies stay solo-copies); Phase
III introduces cooperative behavior; Phase IV validates.

### Phase I — Foundation (non-behavioral)

**Task 1 — `warp_coop_config.cuh`.** ✓ Done (commit `212d822`). Single source
of truth for tier constants, launch shape, smem budget, derived G caps, and
per-group byte costs. Misaligned constants removed from `cuda_config.cuh`.
Solo-kernel guard sites renamed to the new constants.

**Task 2 — Rename intermediate kernel to spec, tear out coop stubs.** ✓
Done. Renamed `pick_intermediate_edges_kernel` → `node_grouped_solo_kernel`
(and `dispatch_intermediate_edges_kernel` →
`dispatch_node_grouped_solo_kernel`). `pick_start_edges_kernel` keeps its
name per §5 — it owns the unconstrained step-0 short-circuit, and
temporarily also services constrained step 0 until task 5 moves that case
into the solo_walks list. Deleted both cooperative stubs
(`pick_start_edges_cooperative_kernel`,
`pick_intermediate_edges_cooperative_kernel`) and their dispatch wrappers;
the four scaffolds in task 3 supersede them. Removed the now-vestigial
`walk_to_group_size` parameter and its guard from both remaining kernels
(the guard existed only to defer to coop; with coop gone, solo services
every active walk). Dispatcher updates follow the same surgery.

**Task 3 — Five-kernel scaffold.** ✓ Done. Extracted the solo step body
into a shared `DEVICE __forceinline__` helper `node_grouped_solo_step_body`
and added four cooperative-tier kernels
(`node_grouped_warp_smem_kernel`, `_warp_global_kernel`,
`_block_smem_kernel`, `_block_global_kernel`), each a thin wrapper that
calls the helper verbatim — distribution is mathematically identical to
solo. Template specialization `<IsDirected, Forward, EdgePickerType>`
matches §5 spec. Each scaffold kernel gets its own
`dispatch_node_grouped_*_kernel` wrapper mirroring the solo dispatcher.
Dispatcher still launches only solo — scaffolds are declared but unused.
Tasks 8–11 replace each scaffold body with its tier-specific implementation;
until then any walk routed through any cooperative kernel produces the same
output as solo. Parity harness is the invariant gate for Phase II (tasks 5–7)
as launch topology rewires.

**Task 4 — Scheduler extraction.** ✓ Done. Moved the filter / compact /
gather / sort / RLE / exclusive-scan / scatter / `num_active` readback
stages (plus NVTX ranges and the per-stage helper kernels) out of
`dispatch_node_grouped_kernel` into `NodeGroupedScheduler`
(`temporal_random_walk_node_grouped_scheduler.{cuh,cu}`). Per-step scratch
is `DeviceArena`-allocated and reset at the top of every `run_step` call;
batch-persistent state (`iota_src_`, `walk_to_group_size_`) stays as
`Buffer<int>` members. The scheduler-internal helper kernels (iota,
alive-flags, gather, scatter, zero) live in the `.cu`'s anonymous
namespace — the per-walk kernel header is now focused on the kernels
that consume the scheduler's outputs. `setup_step0_constrained` is a
separate entrypoint for the per-batch start-node sort/RLE/scatter.
Dispatcher shrinks to ~60 lines: create scheduler, run step 0, loop over
`run_step`s and feed outputs to the solo pick kernel, reverse if backward.
Behavior-preserving — parity harness remains the acceptance gate.

### Phase II — Partition (behavior change, distribution unchanged)

**Task 5 — W partition.** ✓ Done. Scheduler's `run_step` now classifies
each unique node (RLE run) into one of three tiers via
`partition_by_w_kernel` (atomicAdd into `int[3]` counters):
- `W <= TRW_NODE_GROUPED_T_WARP` (==1) → **solo_walks** (one walk_idx per
  entry).
- `T_WARP < W <= TRW_NODE_GROUPED_T_BLOCK` (==255) → **warp_nodes** (tasks
  carry node_id, walk_start offset into sorted_walk_idx, walk_count).
- `W > T_BLOCK` → **block_nodes** (same shape as warp_nodes).

`StepOutputs` exposes the three lists + their device counters + host
counts (second D2H sync per step reads the three counters in one shot).
Dispatcher launches up to three kernels per step, each grid sized to its
own tier count. `warp_global` and `block_global` scaffolds are declared
but unused — task 6's G-partition is their first consumer.

The per-walk kernel helper was renamed from `node_grouped_solo_step_body`
to `advance_one_walk` and no longer does list indexing — it takes a
concrete `walk_idx_int`. Solo kernel is one thread per walk, indexing
`solo_walks[i]`. The four coop scaffolds consume
`(sorted_walk_idx, node_walk_starts, node_walk_counts, num_tasks_ptr)` —
one thread per node task, sequentially iterating walks via
`advance_one_walk`. Distribution still matches solo because each walk's
outcome depends only on `(walk_idx, step_number, base_seed)`.

Cleanup: `setup_step0_constrained`, `walk_to_group_size_`, the
`scatter_walk_group_sizes_kernel` helper, and the `zero_int_buffer_kernel`
are gone — all dead after task 2 removed their consumers. Parity harness
is still the acceptance gate.

**Task 6 — G partition.** ✓ Done. Scheduler runs two additional
`partition_by_g_kernel` passes after the W-partition — one per coop tier
— that split each node task into smem or global based on
`G <= TRW_NODE_GROUPED_G_CAP_{WARP,BLOCK}_{INDEX,WEIGHTED}`. G is read
from `count_ts_group_per_node[node+1] - count_ts_group_per_node[node]`;
direction resolution (outbound/inbound) happens at the dispatcher call
site from the template tags `kDir`/`kFwd`.

`StepOutputs` refactored: a nested `TierTaskList` struct carries
`{nodes, walk_starts, walk_counts, num_tasks_device, num_tasks_host}`,
and there are now four of them — `warp_smem`, `warp_global`,
`block_smem`, `block_global` — plus the standalone solo-walks tier.

Counter layout is a single `int[7]` in the arena (num_solo,
num_warp_intermediate, num_block_intermediate, num_warp_smem,
num_warp_global, num_block_smem, num_block_global). The two
intermediate W-partition counts stay on-device only; the host readback
grabs all seven in one D2H copy at end-of-step (still two syncs total:
one for num_active before the sort, one for tier counts at the end).

Dispatcher now launches five kernels per step: solo on `solo_walks`,
the four coop scaffolds on their respective `TierTaskList`s. Each
launch is skipped if its tier count is 0. Bodies are still solo-copies,
so distribution is identical to task 5 — parity harness stays green
and the W-partition test file still passes (it uses a trivial
all-zero `count_ts_group_per_node` that forces every coop task into
the smem variant; the G-partition tests live in a sibling file).

**Task 7 — Block-task expansion.** ✓ Done. `run_step` now runs
`expand_block_tasks_kernel` twice after the block G-partition — once for
`block_smem` and once for `block_global`. Each thread handles one source
task, computes `num_sub_tasks = ⌈count/BLOCK_WALK_CAP⌉`, reserves a
contiguous output range with a single `atomicAdd(out_counter, num_sub_tasks)`,
and writes every sub-task (same `node_id`, `walk_start + k*cap`,
`min(cap, remaining)`). No per-sub-task atomic.

Counter layout grew from int[7] to int[9]: `[5], [6]` hold the
G-partition intermediate counts (pre-expansion); `[7], [8]` hold the
final post-expansion counts that `StepOutputs.block_smem` and
`block_global` expose. Still one D2H readback at end of step.

Sub-task upper bound: sum of sub-tasks across a tier ≤ sum of source
walks ≤ num_active ≤ num_walks, so post-expansion arrays fit without
arena growth.

Scaffold kernels unchanged — their per-thread loop over `walk_count` is
now bounded by cap. Real coop bodies (tasks 8, 9) will rely on that
bound when sizing their block-wide stride loops. Warp tier is not
expanded (W ≤ T_BLOCK = 255 ≪ cap).

`StepOutputs` surface and dispatcher are unchanged — `block_smem` and
`block_global` still expose `TierTaskList`; their contents are the
expanded sub-tasks.

### Phase III — Kernel bodies (first real speedup)

**Task 8 — Block-smem body.** ✓ Done. `node_grouped_block_smem_kernel` now
runs the real cooperative body:

- One block per block-task, `TRW_NODE_GROUPED_COOP_BLOCK_THREADS` (256)
  threads. Block grid = num_block_smem_tasks (dispatcher changed from
  thread-per-task to block-per-task).
- Dynamic smem sized per picker class in the dispatch wrapper via
  `block_smem_dynamic_smem_bytes<EdgePickerType>()`:
    index pickers    -> 64 B header + 2800 × (size_t + int64_t) = 44864 B
    weighted pickers -> 64 B header + 1800 × (size_t + int64_t) = 28864 B
  Both fit inside the static 48 KB per-block smem envelope (no
  `cudaFuncSetAttribute` opt-in needed).
- Thread-0 broadcast: reads node_id / walk_start / walk_count from the
  task list plus node_group_begin/end and node_edge_offsets[node+1],
  writes to s_header[4] + s_node_edge_end. One `__syncthreads()`.
- Cooperative preload: threads stride by blockDim.x over `[0, G)`,
  each populating `s_group_offsets[p]` (global edge offset) and
  `s_first_ts[p]` (= `view.timestamps[node_ts_sorted_indices[offset]]`).
  One `__syncthreads()`.
- Stride loop: thread `t` handles walks at positions `t, t+256, t+512, …`
  up to `walk_count`. For each walk:
    * Read `last_ts` and `prev_node` from walk_set.
    * Philox init keyed on `(base_seed, walk_idx, step_philox_offset)`.
    * `find_group_pos_slice` with `first_ts = s_first_ts` — fast-path
      single-load comparator in the binary search.
    * Edge-range resolution uses `s_group_offsets[local_pos]` /
      `s_group_offsets[local_pos + 1]` (smem); last group falls back to
      `s_node_edge_end` (preloaded).
    * Final edge fetch (`node_ts_sorted_indices[...]` + `view.sources/
      targets/timestamps/edge_ids`) goes to global. One uncoalesced read
      per walk — unavoidable, not worth preloading.
- Node2Vec gating: originally handled with an `if constexpr (PickerType
  == TemporalNode2Vec)` short-circuit at kernel entry. That branch was
  later removed (see the §6 "Node2Vec bypasses the node-grouped pipeline
  entirely" entry) — the dispatcher now gates Node2Vec out before the
  scheduler + coop kernels ever run, so this kernel assumes non-Node2Vec.
- Weighted pickers still read `weights` (cumulative-weight array) from
  global. Preloading `s_cum_weights[G]` into smem would need the picker
  to accept a slice with an external "prior prefix" argument; deferred
  as a follow-up optimization since the first_ts smem preload already
  kills the dominant dependent-load chain.

All four coop kernel signatures now take `node_walk_nodes` as a parameter
so tasks 9–11 can consume it without further plumbing churn. The three
scaffolds (warp_smem, warp_global, block_global) accept the parameter
and ignore it with `(void)node_walk_nodes;`.

Regression: 289/289 full suite passes, parity harness green — the new
coop body produces the same walk distribution as the scaffold (and as
FULL_WALK) because each walk's Philox seed depends only on
`(walk_idx, step, base_seed)` and the sampling logic is mathematically
equivalent.

**Task 9 — Block-global body.** ✓ Done. Same launch topology as block-smem
(one block per block-task, 256 threads per block) and the same stride
loop over the task's walks. Differences from block-smem:

- No cooperative panel preload. The tier services nodes with
  G > TRW_NODE_GROUPED_G_CAP_BLOCK_{INDEX,WEIGHTED} whose metadata
  wouldn't fit in the block's smem budget anyway.
- `find_group_pos_slice` is called with `first_ts = nullptr`, which
  selects the double-indirect comparator (same access pattern as solo).
  The group-offsets slice pointer addresses global memory directly.
- Stage 2's edge-range resolution reads the same global slice; no smem
  offsets array to consult.

A small static-smem header (s_header[4] + 2 size_t scalars) broadcasts
`node_id`, `walk_start`, `walk_count`, `G`, `node_edge_end`, and
`node_group_begin` once per task so the 256 threads don't each refetch
task-level scalars from global on every walk.

Node2Vec: originally short-circuited inside this kernel; the branch was
later removed when the dispatcher began gating Node2Vec out of the
cooperative pipeline entirely. This kernel now assumes non-Node2Vec.

Dispatcher's block_global launch switches from the scaffold's
thread-per-task grid to block-per-task:
`grid = num_block_global_tasks`, `block = COOP_BLOCK_THREADS (256)`.

Still no dynamic smem at launch — all smem is statically declared inside
the kernel. Tests: 289/289 pass (distribution unchanged — the kernel
produces the same walk each walk_idx would have gotten in the solo
path since Philox keying is per-walk and the sampling math is
equivalent).

**Task 10 — Warp-smem body.** ✓ Done. `node_grouped_warp_smem_kernel`
runs the real cooperative body, warp equivalent of task 8:

- 8 warps per block (`TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK`), 256
  threads per block. Grid = `⌈num_warp_smem_tasks / 8⌉`. Each warp
  services one task; `task_id = blockIdx.x * 8 + warp_id`. Partial
  last block handled by the warp-uniform task_id guard — all 32
  lanes of an out-of-range warp return together.
- Sync discipline: `__syncwarp()` only, never `__syncthreads()`.
  Warps in a block run independent tasks; a block-wide barrier would
  deadlock against the idle warps in the partial last block.
- Per-warp smem slice tiled into one flat dynamic-smem allocation:
  `s_pool + warp_id * kPerWarpBytes`. Per-warp layout matches block-
  smem's shape (header[4], node_edge_end, padding, group_offsets[G],
  first_ts[G]) sized from the picker-class warp G cap (340 index /
  220 weighted). Byte budget: worst case 64 + 340*16 = 5504 B/warp
  × 8 = 44 032 B/block — fits the static 48 KB envelope.
- Lane 0 of each warp broadcasts the task header into its own slice,
  then all 32 lanes cooperatively preload s_group_offsets[G] and
  s_first_ts[G] via a stride-32 loop. At G=340 that's ⌈340/32⌉ = 11
  rounds per lane.
- Intra-warp stride over walks: lane l handles walks at l, l+32, … up
  to walk_count. At W=T_BLOCK (255) that's ⌈255/32⌉ = 8 rounds per
  lane.
- Binary search is per-lane against `s_first_ts` (non-null → fast-path
  comparator, single-load). Each of the 32 lanes runs its own
  independent search with its own walk's `last_ts`; no cross-lane
  cooperation in the search itself.
- Node2Vec: originally short-circuited inside this kernel; the branch
  was later removed once the dispatcher began gating Node2Vec out.
- Dispatcher's warp_smem launch flipped from thread-per-task grid to
  block-per-8-tasks (`grid = ⌈n/8⌉`, `block = 256`), dynamic smem
  sized by `warp_smem_dynamic_smem_bytes<PickerType>()`.

compute-sanitizer + ncu report due at phase IV.

**Task 11 — Warp-global body.** ✓ Done. `node_grouped_warp_global_kernel`
runs the real cooperative body — warp equivalent of task 9. Same
launch topology as warp-smem (8 warps/block, 256 threads, one task
per warp; grid = ⌈num_warp_global_tasks / 8⌉), no panel preload. The
tier services nodes with G > TRW_NODE_GROUPED_G_CAP_WARP_* whose
metadata wouldn't fit in the per-warp 5.5 KB slice.

- `find_group_pos_slice` is called with `first_ts = nullptr`, selecting
  the double-indirect comparator. The group-offsets slice pointer
  addresses global memory directly.
- A small static per-warp smem header (int[4] + 2 size_t per warp ×
  8 warps = 256 B/block) broadcasts node-level scalars once per task
  so the 32 lanes don't each refetch from global on every walk.
- Sync discipline: `__syncwarp()` only — mirrors warp-smem. Partial-
  last-block warps return all 32 lanes together at the warp-uniform
  task-id guard before hitting any sync.
- Node2Vec: originally short-circuited inside this kernel; the branch
  was later removed once the dispatcher began gating Node2Vec out.
- No dynamic smem at launch — all smem is static.

Dispatcher's warp_global launch flipped from thread-per-task grid to
block-per-8-tasks (grid = ⌈n/8⌉, block = 256).

Phase III (real cooperative bodies in all four tiers: block-smem,
block-global, warp-smem, warp-global) is complete. compute-sanitizer
+ ncu report due at Phase IV.

### Phase IV — Validation & polish

**Task 12 — Distribution & performance validation.** Run distribution tests
across every picker × directed/undirected × forward/backward. Measure on
at least:
- Alibaba microservices (clustering workload)
- One TGB dataset (tgbl-wiki or tgbl-review)
- Uniform-degree synthetic graph as a low-clustering control

Write up: per-tier fraction of walks, per-tier fraction of time, speedup
vs `FULL_WALK`, G distribution on each dataset (validates the G cap choices).
Numbers only — paper framing deferred.

**Task 13 — Doc sweep + Python smoke.** Update this dossier's §6 to reflect
the landed state. Pybind smoke test exercising `NODE_GROUPED` with a
non-default picker through the Python path. CLI example refreshed if
binding surface changed.

### Cross-cutting, folded into the primary tasks

- **NVTX ranges.** Already land-good in the current dispatcher; port to the
  scheduler in task 4, extend to the five kernel launches in task 5. Not
  its own task.
- **compute-sanitizer + ncu.** Run with every cooperative kernel body in
  tasks 8–11. Keeps blame windows tight. Not standalone.
- **Parity harness.** Runs as the gate on every Phase I–II task; anchors
  "distribution unchanged" across tasks 5–7.
