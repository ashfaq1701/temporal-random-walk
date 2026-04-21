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

**Done**
- Rename: `STEP_BASED` → `NODE_GROUPED` across enum, kernel file, dispatcher,
  tests, ablations.
- `src/common/warp_coop_config.cuh` — single source of truth for thresholds,
  launch shape, smem budget, derived G caps.
- Scheduler infrastructure (`temporal_random_walk_node_grouped_scheduler.cu`)
  with filter, gather, argsort, RLE, tier partition by W, fit partition by G,
  block-task expansion. Device-side counters, host-sync-free.
- Scheduler outputs the 5 task lists. Arena-backed scratch (`DeviceArena`),
  reset once per step.
- `WalkSetView.walk_padding_value` consumed in the filter kernel.
- Solo kernel functional. Warp-smem, warp-global, block-smem, block-global
  are copies of the solo body — scaffolding only.
- Dispatcher routes step 0 unconstrained to the existing start kernel;
  intermediate steps and constrained step 0 flow through the scheduler +
  solo kernel (no cooperation yet).
- All three entry points in `temporal_random_walk.cu` pass the arena through.

**Open**
- Warp-smem, warp-global, block-smem, block-global bodies are still solo
  copies. No cooperation implemented yet.
- Scheduler runs over full `num_walks` extent in sort + RLE rather than
  `num_active` — correctness holds because scatter uses `num_active` as slot
  count, dead-tail runs never reach task lists. Marked `TODO(perf)`.
- No block-task cap wiring yet on the kernel side (scheduler emits them,
  block-smem kernel body still single-task shape).

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

Ordered by dependency.

### Task 6 — Block-smem kernel body

Replace the solo-copy body of `node_grouped_block_smem_kernel` with the
cooperative design:

- Read the block-task record (node_id, walk_start, walk_count).
- Cooperative preload of the per-node panel into smem: `s_group_offsets[G]`
  and `s_first_ts[G]` for index pickers; add `s_cum_weights[G]` for weighted.
- One `__syncthreads()` after header broadcast, one after panel load.
- Stride loop over walks in the task's slice: thread `t` processes walks at
  positions `t, t+256, t+512, …` up to `walk_count`. Binary search runs
  against smem; picker runs against smem; final edge load goes to global.
- No intra-loop syncs. Writes to `walk_set` go to distinct walk indices.

Out of scope for this task: warp-tier, global fallback, start-kernel. Keep
the function focused.

Distribution tests pass. This is the first task where real speedup appears
on hub-clustered workloads.

### Task 7 — Block-global fallback kernel body

Replace the solo-copy body of `node_grouped_block_global_kernel` with the
same stride-loop structure as Task 6, but without the smem panel. Binary
search goes against the original global-memory path (same access pattern as
the solo kernel, just inside a cooperative stride loop).

Distribution tests pass.

### Task 8 — Warp-smem kernel body

Replace the solo-copy body of `node_grouped_warp_smem_kernel` with the warp
equivalent of Task 6:

- One warp per unique node. 8 warps per block (`COOP_WARPS_PER_BLOCK`).
- Per-warp smem slice (5.5 KB): same panel contents, scaled to per-warp G
  caps (340 index / 220 weighted).
- Lane-strided preload, `__syncwarp()` in place of `__syncthreads()`.
- Intra-warp stride loop: lane `l` processes walks at positions
  `l, l+32, l+64, …` up to `walk_count`. At W=255 this is ⌈255/32⌉ = 8
  iterations.

Distribution tests pass.

### Task 9 — Warp-global fallback kernel body

Warp-tier equivalent of Task 7. Same structure as warp-smem, no smem panel.

Distribution tests pass.

### Task 10 — Validation

Run distribution tests across all picker types, directed/undirected, forward/
backward. Measure on at least:
- Alibaba microservices (the clustering workload)
- One TGB dataset (tgbl-wiki or tgbl-review)
- A uniform-degree synthetic graph as a low-clustering control

Write up: per-tier fraction of walks, per-tier fraction of time, speedup vs.
`FULL_WALK`, distribution of G on each dataset (validates the G cap choices).
No paper-ready framing yet — just numbers.
