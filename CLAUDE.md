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
- Solo kernels functional for every picker × directed/undirected × forward/
  backward combination. Current names are still old-style
  (`pick_start_edges_kernel`, `pick_intermediate_edges_kernel`); rename to
  `node_grouped_solo_kernel` lands in task 2.
- Per-step pipeline inline inside `dispatch_node_grouped_kernel`: filter
  (`walk_alive_flags_kernel`) → compact (`cub_partition_flagged`) → gather
  (`gather_last_nodes_kernel`) → sort (`cub_sort_pairs`) → RLE
  (`cub_run_length_encode`) → exclusive-scan → scatter
  (`scatter_walk_group_sizes_kernel`) → solo+coop launch.
- Pipeline extents driven by `num_active`: sort / RLE / scan / gather /
  scatter / pick all run on `host_num_active` read once per step via a D2H
  copy + stream sync on `trw->stream()` (CUB's `num_items` is host-side, no
  device-pointer overload exists). Filter and the `walk_to_group_size`
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
- `TemporalNode2Vec` pinned to the solo tier: coop dispatcher early-exits on
  that picker; solo's group-size guard is compile-time elided for the
  Node2Vec specialization. Per-walk `prev_node` bias makes a shared
  cooperative panel impossible by construction.
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
- Kernel naming still old-style. Rename `pick_*_edges_kernel` →
  `node_grouped_solo_kernel` in task 2.
- Cooperative tier is two TODO stubs (`pick_start_edges_cooperative_kernel`,
  `pick_intermediate_edges_cooperative_kernel`), not the four scaffolds the
  spec requires (`node_grouped_warp_smem_kernel`, `_warp_global_kernel`,
  `_block_smem_kernel`, `_block_global_kernel`). Scaffolds land in task 3.
- No `temporal_random_walk_node_grouped_scheduler.cu`. Pipeline is inline in
  the dispatcher. Extraction (with `DeviceArena`-backed scratch, reset once
  per step) lands in task 4.
- Tier routing is still guard-and-exit: solo kernel gates on
  `walk_to_group_size[walk_idx] > TRW_NODE_GROUPED_T_WARP`, coop stubs are
  no-ops. Spec requires pre-partitioned task lists (solo_walks,
  warp_smem_nodes, warp_global_nodes, block_smem_tasks, block_global_tasks)
  — lands across tasks 5–7.
- `DeviceArena` exists (`src/data/device_arena.cuh`) but the dispatcher
  still allocates per-invocation `Buffer<T>` scratch. Arena switch lands
  with the scheduler extraction in task 4.
- No W-partition, no G-partition, no block-task expansion for mega-hubs.

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

**Task 2 — Rename kernels to spec.** Collapse `pick_start_edges_kernel` and
`pick_intermediate_edges_kernel` under the single `node_grouped_solo_kernel`
name; keep the internal `<IsDirected, Forward, EdgePickerType, Constrained>`
template specializations (start edges retain the `Constrained` tag, step
edges don't need it). Tear out the two cooperative stubs
(`pick_*_cooperative_kernel`) — superseded by the four scaffolds in task 3.
Update `dispatch_start_edges_kernel`, `dispatch_intermediate_edges_kernel`,
and their cooperative counterparts. Parity harness passes.

**Task 3 — Five-kernel scaffold.** Introduce
`node_grouped_warp_smem_kernel`, `_warp_global_kernel`, `_block_smem_kernel`,
`_block_global_kernel` — each a verbatim copy of the solo body so
distribution is mathematically unchanged. Dispatcher still launches only
the solo kernel (scaffold kernels are declared but unused). Template
specialization on `<IsDirected, Forward, EdgePickerType>` matches spec.
Parity harness passes.

**Task 4 — Scheduler extraction.** Move filter / gather / argsort / RLE /
exclusive-scan / scatter / `num_active` readback / NVTX markers out of
`dispatch_node_grouped_kernel` into `temporal_random_walk_node_grouped_scheduler.cu`.
Per-invocation `Buffer<T>` scratch → `DeviceArena`-allocated, reset once per
step. Behavior-preserving refactor. Parity harness passes.

### Phase II — Partition (behavior change, distribution unchanged)

**Task 5 — W partition.** Scheduler emits three disjoint task lists:
`solo_walks`, `warp_nodes`, `block_nodes`. Solo kernel consumes `solo_walks`
(group-size guard removed — kernel no longer receives walks it skips).
Warp-smem launches over `warp_nodes`, block-smem over `block_nodes`. All
four cooperative bodies still solo-copies, so distribution is identical to
task 4. Parity harness passes.

**Task 6 — G partition.** Scheduler splits `warp_nodes` →
`warp_smem_nodes` + `warp_global_nodes` using
`TRW_NODE_GROUPED_G_CAP_WARP_{INDEX,WEIGHTED}`; same for block with
`G_CAP_BLOCK_*`. Dispatcher launches all five kernels over their five
task lists. Bodies still solo-copies. Parity harness passes.

**Task 7 — Block-task expansion.** Nodes with
`W > TRW_NODE_GROUPED_BLOCK_WALK_CAP` (8192) split into ⌈W/cap⌉ block-tasks
of disjoint walk slices. `block_smem_nodes` → `block_smem_tasks`; same for
global. Block-smem and block-global kernels consume
`(node_id, walk_start, walk_count)` tasks, not raw nodes. Parity harness
passes.

### Phase III — Kernel bodies (first real speedup)

**Task 8 — Block-smem body.** Replace solo-copy with the cooperative design:
read the block-task record, cooperative preload of `s_group_offsets[G]` and
`s_first_ts[G]` (index pickers) plus `s_cum_weights[G]` (weighted),
`__syncthreads()` after header broadcast and after panel load, stride loop
of 256 threads over `walk_count` walks. Binary search and picker run
against smem; final edge load goes to global. compute-sanitizer +
memcheck/racecheck clean. ncu report for baseline→cooperative delta.

**Task 9 — Block-global body.** Same task-consumption shape and stride loop
as task 8, without the smem panel. Binary search goes against global arrays
— same access pattern as solo, inside a cooperative stride loop.
compute-sanitizer + ncu report.

**Task 10 — Warp-smem body.** Warp equivalent of task 8: 8 warps per block
(`TRW_NODE_GROUPED_COOP_WARPS_PER_BLOCK`), per-warp 5.5 KB panel slice
(`SMEM_PANEL_BYTES_PER_WARP`), lane-strided preload, `__syncwarp()` in
place of `__syncthreads()`. Intra-warp stride loop: lane `l` processes
walks at `l, l+32, l+64, …` up to `walk_count`. At `W=255` this is 8
iterations. compute-sanitizer + ncu report.

**Task 11 — Warp-global body.** Warp equivalent of task 9.
compute-sanitizer + ncu report.

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
