# NODE_GROUPED walk kernel — design dossier

Scoped to the Phase-2 walk-sampling refactor on `feature/warp-collaboration`.
This file is the durable context for the node-grouped pipeline: what it is,
why it exists, what it looks like in the code today, and what is still open.

## 1. Problem

The `FULL_WALK` kernel assigns one CUDA thread to one walk for the walk's
entire life. Each thread, at every step, re-reads the adjacency list of
*its own* current node, computes a picker-specific CDF, samples an index,
and advances. This is fine when walks are uncorrelated and graphs are
small, but on our workloads it shows three pathologies:

1. **No cache reuse across walks that happen to sit on the same node.**
   On skewed graphs, a hub node hosts thousands of concurrent walks, but
   each thread re-fetches the hub's adjacency list from global memory.
2. **Warp divergence widens with step count.** Threads in a warp terminate
   at wildly different times; late steps run with most lanes idle.
3. **Random, scattered adjacency reads defeat the L1/L2.** Threads in a
   warp are rarely on neighbouring nodes, so each read is effectively a
   cold line.

The `NODE_GROUPED` path is the response: **group walks by their current
node per step**, then process each group cooperatively so the adjacency
list / CDF is fetched once per group instead of once per walk.

## 2. Overall plan

Per step (including step 0 / start-edge selection) we run this pipeline:

1. **Filter**: drop walks that have terminated
   (`walk.nodes[walk_idx][last] == walk.walk_padding_value`).
2. **Gather**: collect the surviving walks' current/last nodes.
3. **Sort-by-key**: `(last_node, walk_idx)` pairs sorted by last_node.
4. **Run-length encode**: yields per-unique-node run starts + lengths.
5. **Scatter group sizes**: write each surviving walk's group size into a
   per-walk array indexed by *original* walk_idx.
6. **Tiered launch**:
   - **solo** kernel — thread per walk, for group_size ≤ T_warp
   - **warp-coop** kernel — warp per group, for T_warp < group_size ≤ T_block
   - **block-coop** kernel — block per group, for group_size > T_block
7. Write results into the walk set at slot `walk_idx`.

Step 0 has one extra branch: if `start_node_ids[i] == -1` for all i
("fully unconstrained"), there is nothing to group by — every walk picks
a random start edge from the global pool — so we short-circuit to a single
solo kernel and skip the sort/RLE infrastructure entirely.

The "preserve original walk_idx" invariant holds end-to-end: compaction,
sort, and RLE all carry walk_idx as the payload, so every downstream
kernel writes into `walk_set[walk_idx]` without dense renumbering.

## 3. Scheduling strategy & thresholds

Initial thresholds (calibration pending — see future tasks):

| Tier        | Group size range        | Cooperation unit | Rationale                          |
| ----------- | ----------------------- | ---------------- | ---------------------------------- |
| solo        | group_size ≤ 1 (or skew tail) | 1 thread  | No reuse benefit; coop is overhead |
| warp-coop   | 2 ≤ group_size ≤ 32     | 1 warp (32 thr)  | Load adj list into smem once/warp  |
| block-coop  | group_size > 32         | 1 block (≥128 thr) | Very hot nodes — amortize load more |

Dispatch is driven by `walk_to_group_size[walk_idx]` (set in step 5 of
the pipeline). Each tier's kernel guards on group size and returns early
if the walk does not belong to it, so all three tiers can launch over
the same walk index space without coordination.

Rules:
- Thresholds are tunable constants, not compile-time requirements.
- Bit-exactness across tiers is non-negotiable: all tiers key Philox on
  `(base_seed, walk_idx, step_kernel_philox_offset(step))`. The
  `START_KERNEL_BUDGET = 3` offset guarantees step kernels never collide
  with the start kernel's counter range.
- No host sync between stages — all counters (`num_active`, `num_runs`)
  live on device; kernels launch with upper-bound grids and early-exit
  past the device-side count.

## 4. Shared-memory plan

### What goes in smem (warp-coop / block-coop tiers)

Per group:
1. **Adjacency offsets + edge indices** for the group's current node
   (CSR slice of length `degree`).
2. **Edge timestamps** for that slice (needed for directional filtering
   and for timestamp-based pickers).
3. **Picker CDF** — either built in smem from the weights, or streamed
   from a precomputed global array into smem once per group.
4. **Small scratch for reduction / scan** when the picker needs it
   (e.g. weighted pickers).

The fundamental win is #1–#3 are loaded *once* and reused across every
walk in the group (potentially thousands).

### Smem transfer limit

Usable smem per block on our target arches is ~48 KB (configurable up to
~100 KB on Ampere, but we stay conservative). Budgeting:

- edge_idx: `degree * sizeof(int)` = 4·deg B
- edge_ts:  `degree * sizeof(int64_t)` = 8·deg B
- cdf:      `degree * sizeof(float)` = 4·deg B
- misc scratch: ~1 KB

Rough cap: `degree ≲ 2800` fits cleanly; `degree ≲ 4000` fits with
aggressive packing. Beyond that the adjacency slice *does not fit* and
we fall back.

### Branching on smem fit

At group dispatch time we compare `degree` against a compile-time /
constexpr `SMEM_DEG_LIMIT`:

- **degree ≤ SMEM_DEG_LIMIT** → smem-backed coop kernel (fast path).
- **degree > SMEM_DEG_LIMIT** → global-memory coop kernel, or demote to
  solo if the group is small enough that reuse doesn't pay the global
  round-trip.

The specific demotion policy (block-coop-global vs warp-coop-global vs
solo) is still TBD and will be driven by empirical measurement.

## 5. Kernel matrix

| Kernel                                           | Tier  | Smem-resident adj? | Status    |
| ------------------------------------------------ | ----- | ------------------ | --------- |
| `pick_start_edges_kernel`                        | solo  | n/a                | done      |
| `pick_start_edges_cooperative_kernel`            | warp  | yes                | TODO stub |
| (future) `pick_start_edges_block_kernel`         | block | yes                | not started |
| (future) `pick_start_edges_cooperative_global_k` | warp  | no                 | not started |
| `pick_intermediate_edges_kernel`                 | solo  | n/a                | done      |
| `pick_intermediate_edges_cooperative_kernel`     | warp  | yes                | TODO stub |
| (future) `pick_intermediate_edges_block_kernel`  | block | yes                | not started |
| (future) `pick_intermediate_edges_coop_global_k` | warp  | no                 | not started |
| `reverse_walks_kernel`                           | —     | n/a                | done      |

Each coop kernel is dispatched via a `dispatch_*` wrapper that lifts
runtime values (is_directed, walk direction, picker type, constrained
flag) into template tags — the kernels themselves are fully specialized.

## 6. Where we are

**Done**
- Rename: `STEP_BASED` → `NODE_GROUPED` across enum, kernel file,
  dispatcher, test, test_run, and ablation entrypoints.
- `src/common` helpers: `cub_sort_pairs` (cuda_sort.cuh),
  `cub_run_length_encode` + `cub_partition_flagged` (cuda_scan.cuh) —
  stream-aware, `Buffer<uint8_t>` scratch, two-call pattern.
- Node-grouped dispatcher
  (`temporal_random_walk_node_grouped_dispatch.cuh`) with:
  - Step-0 constrained vs unconstrained branch.
  - Full per-step sort/RLE/scatter infra, device-side counters.
  - Terminated-walk filter via flag+compaction with original walk_idx
    preserved.
- Solo kernels (start + intermediate) fully functional, driven by
  `Constrained` tag dispatch for start edges.
- `WalkSetView.walk_padding_value` consumed inside the filter kernel —
  consistent with the already-merged walk-padding fix (commit 83ab033).
- All three walk entrypoints in `temporal_random_walk.cu` pass
  `all_starts_unconstrained` correctly:
  - `_for_all_nodes_cuda`: false (seeds from graph node list)
  - `_for_last_batch_cuda`: false (seeds from real edge endpoints)
  - `_cuda`: true (fills start_node_ids with -1)

**Open (this pass intentionally deferred)**
- Warp-coop kernel bodies are TODO stubs with structural comments.
- No block-coop tier yet.
- `cub_sort_pairs` and `cub_run_length_encode` currently operate over
  the full `num_walks` extent rather than `*num_active` — correctness
  is preserved because the scatter step uses `*num_active` as its slot
  count, so dead-tail runs never leak into `walk_to_group_size`. Marked
  `TODO(perf)` in the dispatcher.

## 7. Problems solved

1. **Preserving original walk indices through filter + sort.** Solved
   by threading compacted walk_idx through as the value array of the
   sort-by-key, so every downstream kernel addresses slots directly.
2. **Host-sync-free pipeline.** Device-side `num_active` / `num_runs`
   counters, upper-bound launch grids, kernel-side early-exit past the
   device count.
3. **RNG bit-exactness across tiers.** Philox keyed on `(base_seed,
   walk_idx, step_philox_offset)`; independent of which tier services
   the walk. `START_KERNEL_BUDGET = 3` prevents start/step counter
   collision.
4. **Wasted bandwidth on terminated walks.** `walk_alive_flags_kernel`
   reads `walk_padding_value` from the view; terminated walks are
   compacted out before the sort.
5. **Walk-padding-value follow-up from commit 83ab033.** Filter kernel
   consumes the configured padding from WalkSetView rather than
   hardcoding -1, matching the already-merged fix on master.
6. **Unconstrained-start short-circuit.** Detecting "all -1" at the
   dispatch layer skips the entire sort/RLE/scatter infra at step 0.

## 8. Architecture guardrails

These are the standing rules for this refactor — imposed by the user,
respected going forward:

- **Don't auto-kick builds or tests.** The user signals when ready.
  Applies to `cmake --build`, `ctest`, `pytest`, and anything that
  spins CUDA work. (See `memory/feedback_hold_builds_during_edits.md`.)
- **No half-finished implementations.** Either land a fully working
  kernel or a clearly marked TODO stub with structural comments. No
  silently-broken code.
- **Preserve original walk_idx end-to-end.** Never renumber walks to a
  dense range mid-pipeline; carry the original index through every
  sort/compact.
- **Read `walk_padding_value` from `WalkSetView`.** Never hardcode -1
  as the padding sentinel in new code.
- **Buffer-backed CUB scratch.** All temp storage is a scope-local
  `Buffer<uint8_t>` — no manual cudaMalloc/cudaFree.
- **Stream-aware everywhere.** All kernels, CUB calls, and scratch
  allocations on `trw->stream()` (or the caller-provided stream). No
  unexplained sync on stream 0.
- **Tag-dispatch for runtime booleans / enums.** Use `dispatch_bool` /
  `dispatch_picker_type` to lift values into template parameters
  rather than branching inside the kernel hot path.
- **One responsibility per kernel.** Filter, gather, sort, RLE, scatter,
  pick — all separate kernels / CUB calls. No mega-kernel.
- **Rename, don't shim.** When we rename `STEP_BASED` → `NODE_GROUPED`,
  all call sites update; no compatibility alias.
- **CUDA-only features land behind `#ifdef HAS_CUDA`.** CPU build must
  still compile and run full-walk paths.

## 9. Future tasks

Ordered roughly by dependency and blast radius.

1. **Correctness parity harness.** Deterministic test that runs the
   same seed through `FULL_WALK` and `NODE_GROUPED` (solo-only today),
   compares walk sequences element-wise. Must pass before any
   coop-kernel work lands.
2. **Warp-coop start-edges kernel.** Implement
   `pick_start_edges_cooperative_kernel` body: warp-per-group, adj
   slice in smem, lane-parallel CDF build + binary search. Gate by a
   `warp_smem_fits` precheck.
3. **Warp-coop intermediate-edges kernel.** Mirror of #2 for the
   per-step kernel. Same smem layout; extra constraint from directional
   filtering (directed + direction tag).
4. **Smem-fit precheck + global-memory coop fallback.** When
   `degree > SMEM_DEG_LIMIT`, run a coop variant that keeps the adj
   slice in global but still amortizes picker setup across the group.
5. **Block-coop tier.** Separate kernel launched for groups above
   `T_block`. Block-wide reduction/scan for weighted pickers. Revisit
   tile-in-smem strategy if `degree` does not fit.
6. **Threshold calibration.** Sweep `T_warp` / `T_block` /
   `SMEM_DEG_LIMIT` on representative workloads (stack-overflow,
   reddit, a synthetic skewed graph), pick defaults from the Pareto
   curve, commit the chosen constants with a comment pointing at the
   sweep rep files.
7. **Device-side item count through `cub_sort_pairs` + RLE.** Remove
   the current full-extent sort. Needs a CUB-side extension or a
   stream-capture trick — investigate.
8. **NVTX ranges on each pipeline stage.** Filter / gather / sort /
   RLE / scatter / pick — make the stages legible in nsys.
9. **ncu profiling pass.** Check smem bank conflicts on the coop
   kernels, stall reasons on the solo kernel's divergent path,
   occupancy on both. Reports already in tree:
   `report-large-test*.ncu-rep`, `report-ncp*.ncu-rep` — compare
   before/after.
10. **compute-sanitizer (memcheck + racecheck) run.** After coop
    kernels land. Required before merging to master per the
    `project_cuda_available.md` standing expectation.
11. **Python binding smoke test.** Confirm `NODE_GROUPED` reaches
    through the pybind layer with the right default (probably leave
    `FULL_WALK` as default until the coop kernels are empirically
    wins).
12. **Doc sweep.** README + examples updated once #11 lands; this
    CLAUDE.md updated with calibrated thresholds and final kernel
    matrix.
