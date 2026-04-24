# temporal-random-walk — system reference

A CUDA-accelerated library for sampling temporal random walks on streaming
graphs. Public entry point is a Python package (`temporal_random_walk`) backed
by a pybind11 module; the test_run binaries are pure C++ and useful for the
tight measurement loop.

Two walk-sampling paths share one API (`TemporalRandomWalk.get_random_walks_*`):

- **FULL_WALK** — one CUDA thread per walk for the entire walk's life. Walk
  state lives in registers across all steps. Simple, fast on low-clustering
  graphs, but loses to global-memory pressure on skewed temporal graphs.
- **NODE_GROUPED** — per-step cooperative pipeline that groups concurrent
  walks by their current node, preloads each hub's metadata into shared
  memory once, and services all co-located walks from that panel. Default.

A third mode, **NODE_GROUPED_GLOBAL_ONLY**, exists as a paper ablation knob
(see [Ablation variant](#ablation-variant)).

---

## Source tree

```
temporal_random_walk/src/
├── common/                     # cross-cutting utilities
│   ├── warp_coop_config.cuh     ★ tier constants, G caps, smem budget
│   ├── cuda_scan.cuh            # CUB wrappers: exclusive sum, RLE, partition flagged
│   ├── cuda_sort.cuh            # CUB wrappers: radix sort pairs
│   ├── nvtx.cuh                 # RAII NVTX range markers
│   ├── error_handlers.cuh       # CUDA_CHECK / CUB_CHECK macros
│   ├── picker_dispatch.cuh      # tag-dispatch helpers for picker enum → template
│   ├── random_gen.cuh           # Philox state + keying helpers
│   └── warp_coop_config.cuh
│
├── core/
│   ├── temporal_random_walk.{cuh,cu}  # top-level walk entry points
│   ├── temporal_random_walk_cpu.cuh   # CPU fallback (no CUDA)
│   ├── temporal_random_walk_kernels_full_walk.cuh  # FULL_WALK kernel
│   ├── helpers.cuh                    # enum-from-string parsers
│   └── node_grouped/
│       ├── scheduler.{cuh,cu}         ★ per-step pipeline orchestrator
│       ├── dispatch.cuh               ★ template dispatcher for NODE_GROUPED
│       ├── kernels.cuh                # umbrella include
│       └── kernels/
│           ├── per_walk.cuh           # solo / per-walk / start-edges kernels
│           ├── coop_warp.cuh          # warp-tier kernels (smem + global)
│           ├── coop_block.cuh         # block-tier kernels (smem + global)
│           └── common.cuh             # shared device helpers (NodeDirPtrs etc.)
│
├── graph/
│   ├── temporal_graph.{cuh,cu}        # stream-time windowed graph
│   ├── edge_data.{cuh,cu}             # edge storage, timestamp groups
│   ├── node_edge_index.{cuh,cu}       # per-node outbound/inbound indexes
│   ├── node_features.{cuh,cu}         # optional node features
│   ├── edge_selectors.cuh             # get_node_edge_at_device (hot path)
│   ├── walk_step_helpers.cuh          # per-step pick + advance math
│   └── temporal_node2vec_helpers.cuh  # biased-walk math (Node2Vec path)
│
├── data/
│   ├── temporal_graph_data.cuh        # owning graph container
│   ├── temporal_graph_view.cuh        # non-owning view passed to kernels
│   ├── buffer.{cuh,cu}                # device/host buffer, GPU-aware resize
│   ├── device_arena.cuh               # bump-pointer GPU arena (per-step scratch)
│   ├── enums.cuh                      ★ RandomPickerType, WalkDirection, KernelLaunchType
│   ├── structs.cuh                    # misc POD structs
│   └── walk_set/
│       ├── walk_set_device.{cuh,cu}   # device-resident walk output
│       ├── walk_set_host.{cuh,cu}     # host copy / iterator
│       ├── walk_set_view.cuh          # non-owning view, threaded into kernels
│       └── walks_with_edge_features_host.cuh
│
├── proxies/                            # public API surface (C++)
│   ├── TemporalRandomWalk.{cuh,cu}     # main class
│   └── RandomPicker.{cuh,cu}
│
├── random/
│   └── pickers.cuh                     # per-picker sample() device code
│
└── utils/
    ├── random.cuh                      # Philox helpers
    ├── omp_utils.cuh
    └── utils.cuh

temporal_random_walk/py_interface/
├── _temporal_random_walk.cu            # pybind11 module — Python API surface
├── buffer_to_numpy.cuh                 # zero-copy device→NumPy
└── CMakeLists.txt

temporal_random_walk/test/               # GoogleTest unit + parity tests
└── test_node_grouped_*.cpp              # scheduler / W-partition / G-partition / parity

temporal_random_walk/test_run/            # standalone benchmark / driver binaries
├── test_run_temporal_random_walk.cpp   # smoke runner over a CSV dataset
├── walk_sampling_speed_test.cpp        # single-config walk-time timer
├── ablation_streaming.cpp              # temporal-streaming ablation runner
└── test_alibaba_streaming.cpp          # Alibaba-dataset streaming benchmark (CSV)
```

Files marked ★ are the ones most worth reading before modifying anything.

---

## Build & run

### Python wheel (pybind module)

From the repo root:

```bash
# Optional fast rebuild for a single arch matching your GPU.
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" python setup.py bdist_wheel
pip install --force-reinstall --no-deps dist/temporal_random_walk-*-cp3*-*.whl
```

The default arch list is `75;80;86;89;90`. Single-arch cuts build time ~4×.
`CMAKE_ARGS` is forwarded to cmake by `setup.py`.

### Test_run binaries (pure C++)

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ablation_streaming test_alibaba_streaming \
                            test_run_temporal_random_walk walk_sampling_speed_test
# Binaries land in build/bin/
```

### C++ tests (GoogleTest)

```bash
cmake --build build --target trw_tests && build/bin/trw_tests
```

Includes the parity harness
(`test_node_grouped_parity.cpp`) that asserts `NODE_GROUPED` produces the
same walk distribution as `FULL_WALK` across directed/undirected × forward/
backward × every picker.

### Build gotchas (both fixes live in `CMakeLists.txt`)

- **Ubuntu 24.04 TBB cmake config** injects `/usr/include` into
  `TBB::tbb` includes, which becomes `-isystem /usr/include` and breaks
  gcc-13's `<cmath>`. Workaround: `list(REMOVE_ITEM _tbb_inc "/usr/include")`
  right after `find_package(TBB REQUIRED)`.
- **CUDA_SEPARABLE_COMPILATION is OFF** on the pybind target. Enabling
  RDC with the multi-arch fatbin (75;80;86;89;90) causes CUB-dispatched
  kernels (`thrust::copy_if`, etc.) to launch the wrong policy kernel on
  non-highest-arch devices, failing with
  `cudaLaunchKernel: cudaErrorInvalidValue`. Nothing here uses RDC features;
  keep it off.

---

## Kernel launch types

Three values of `KernelLaunchType` in `data/enums.cuh`. All produce the same
walk distribution for a given `(walk_idx, base_seed)` — Philox keying is
path-invariant.

| Enum | Paper name | What it does |
|---|---|---|
| `FULL_WALK` | FULL_WALK (baseline) | One thread per walk, entire walk in one kernel, state in registers. |
| `NODE_GROUPED` | COOP_WITH_SMEM | Per-step scheduler + five cooperative kernels. Hubs ≤ G-cap use the smem-panel fast path. |
| `NODE_GROUPED_GLOBAL_ONLY` | COOP_WITHOUT_SMEM | Same pipeline, but the G-partition's caps are forced to −1 so every coop task lands in the `*_global` tier (no smem preload). |

`DEFAULT_KERNEL_LAUNCH_TYPE = NODE_GROUPED`. Selected at the public API call
site:

```python
walks, times, lens, feats = trw.get_random_walks_and_times_for_last_batch(
    max_walk_len=100, walk_bias="ExponentialIndex", num_walks_per_node=20,
    kernel_launch_type="NODE_GROUPED_GLOBAL_ONLY")  # any of the three strings
```

From C++:

```cpp
trw.get_random_walks_and_times_for_all_nodes(
    max_walk_len, &picker, num_walks_per_node, &start_picker,
    WalkDirection::Forward_In_Time,
    KernelLaunchType::NODE_GROUPED);  // or FULL_WALK / NODE_GROUPED_GLOBAL_ONLY
```

---

## FULL_WALK path

`core/temporal_random_walk_kernels_full_walk.cuh` holds one big kernel,
`generate_random_walks_kernel`. Grid shape: `num_walks / block_dim.x`,
one thread per walk. Each thread:

1. Picks the start edge (honours `start_node_ids[i]` if constrained).
2. Loops `max_walk_len` steps: at each step, calls
   `get_node_edge_at_device` (in `graph/edge_selectors.cuh`), which does a
   temporal-cutoff binary search on the current node's timestamp-group
   array, picks a group, samples an edge, advances.
3. Writes `walk_set.nodes[walk_idx][step]` directly.

Walk state (`current_node`, `current_ts`, `prev_node`, `current_edge_id`)
lives in registers across all steps. The only per-hop global write is
`walk_set.add_hop`. This is why FULL_WALK is so fast on low-clustering
graphs: it amortizes nothing across walks but also pays no per-step
cross-kernel overhead.

Three pathologies appear on skewed temporal graphs:

1. **No reuse across co-located walks.** Thousands of walks can share one
   hub node at a given step; each thread re-fetches the hub's per-node
   arrays from global memory.
2. **3-deep dependent-load chain in the binary search.** Each probe
   reads `node_ts_groups_offsets[probe] → sorted_idx`, then
   `node_ts_sorted_indices[sorted_idx] → edge_idx`, then
   `view.timestamps[edge_idx] → compared_ts`. Each load waits for the
   prior to return. Binary search does `log₂(G)` probes, all serialized.
3. **Warp divergence grows with step count.** Threads in a warp terminate
   at wildly different times; late steps run with most lanes idle.

NODE_GROUPED addresses all three.

---

## NODE_GROUPED scheduling — deep dive

### The idea

Across walks, at each step, group threads by *current node*. Every walk at
the same node needs the same per-node metadata: the timestamp-group array
for the binary search, the first-timestamp-per-group array (to collapse
the 3-deep load chain). Load that metadata into shared memory once per
hub per step, then let every co-located walk binary-search against the
smem copy. That's the whole mechanism.

The cost is the grouping itself — sort walks by current node per step,
RLE to find runs, partition by workload size — plus the cross-kernel
dispatch of the pick phase. On workloads where the grouping cost amortizes
(many walks per hub, many steps), NODE_GROUPED wins. On low-clustering or
short-walk workloads, the scheduler overhead dominates and FULL_WALK wins.

### Per-step pipeline

`NodeGroupedScheduler::run_step` in
`src/core/node_grouped/scheduler.cu`. Stages, in order:

1. **Filter.** `walk_alive_flags_kernel` marks every walk whose current
   slot equals `walk_padding_value` as dead. Output: `uint8_t[num_walks]`.
2. **Compact.** `cub_partition_flagged` writes the compacted list of alive
   walk indices (from a persistent iota source). Output: `active_walk_idx`.
3. **num_active D2H readback.** Drives sizes for downstream CUB calls,
   which need a host-side `num_items`. One `cudaMemcpyAsync` + stream
   sync per step.
4. **Gather.** `gather_last_nodes_kernel` reads each active walk's
   current node. Output: parallel `(last_node, walk_idx)` pairs.
5. **Sort.** `cub_sort_pairs` sorts `(key=last_node, value=walk_idx)`.
   Output: `sorted_last_nodes[]`, `sorted_active_idx[]`.
6. **RLE.** `cub_run_length_encode` over the sorted keys produces
   `unique_nodes[]`, `run_lengths[]`, and `num_runs` (a device scalar).
   Each run is one (node, W) group where W = walks-per-node at this step.
7. **Exclusive scan.** `cub_exclusive_sum` over `run_lengths` produces
   `run_starts[]` — where each node's walk slice starts in
   `sorted_active_idx`.
8. **W-partition** (see below).
9. **G-partition** (see below). Two passes — one for the warp tier, one
   for the block tier.
10. **Block-task expansion** (see below). Two passes — one for each block
    sub-tier (smem, global).
11. **Tier-count D2H readback.** Single 9-int copy at end of step. Drives
    grid sizes for the five pick-kernel launches.

Everything between steps 1 and 11 runs on `trw->stream()`; the two syncs
(num_active + tier counts) are the only stall points.

NVTX ranges are emitted on every stage (`NG step`, `NG filter alive`,
`NG compact`, `NG num_active readback`, `NG sort`, `NG RLE+scan`,
`NG W-partition`, `NG G-partition (warp)`, `NG G-partition (block)`,
`NG block-task expansion (smem|global)`, `NG tier-count readback`,
`NG pick`) for legible nsys traces.

### W-partition — classifying by walks-per-node

`partition_by_w_kernel` in `scheduler.cu`. One thread per RLE run. For
each unique node it sees `W = run_lengths[r]` and routes the run into one
of three tiers:

```
W <= T_WARP (=1)                 → solo      (one walk_idx per entry)
T_WARP < W <= T_BLOCK (=255)     → warp      (one task; services W walks cooperatively)
W > T_BLOCK                       → block     (one task; services W walks cooperatively)
```

Atomics into a shared `int[3]` counter, per tier. Output order within each
tier is non-deterministic — the cooperative kernels iterate their task
lists independently, so order doesn't matter.

Threshold rationale:

- **`T_WARP = 1`.** Below this, there's only one walk for this node at
  this step — nothing to cooperate over. Solo threads don't pay any
  cooperation bookkeeping.
- **`T_BLOCK = 255`.** Tuned to fit 8 intra-warp stride rounds
  (`⌈255/32⌉ = 8`): a warp servicing up to 255 walks does at most 8
  rounds of its stride loop. Above that, a full 256-thread block
  amortizes the smem-panel preload better than 8 warps spread across
  blocks would.

Both are tunable constants in `warp_coop_config.cuh`. Relative ordering
(`T_WARP < T_BLOCK`) is what matters for correctness; absolute values are
perf knobs.

### G-partition — smem fit test

`partition_by_g_kernel` in `scheduler.cu`, called twice per step (once
for the warp tier, once for the block tier). For each task in the input
tier, computes

```
G = count_ts_group_per_node[node_id + 1] - count_ts_group_per_node[node_id]
```

— the number of distinct timestamp groups at this node — and routes into
one of two sub-tiers:

```
G ≤ g_cap    → *_smem tier    (panel preload + stride loop)
G >  g_cap   → *_global tier  (no panel; binary-search against global arrays)
```

`g_cap` is chosen per-tier × per-picker-class:

| tier × picker class | constant | value | bytes used |
|---|---|---:|---|
| block × index | `G_CAP_BLOCK_INDEX` | **2800** | 2800 × 16 B = 44800 B |
| block × weighted | `G_CAP_BLOCK_WEIGHTED` | **1800** | 1800 × 24 B = 43200 B |
| warp × index | `G_CAP_WARP_INDEX` | **340** | 340 × 16 B = 5440 B |
| warp × weighted | `G_CAP_WARP_WEIGHTED` | **220** | 220 × 24 B = 5280 B |

Index pickers are Uniform / Linear / ExponentialIndex. Weighted is
ExponentialWeight (needs cumulative weights; an extra 8 B/group budgeted
for a future `s_cum_weights` preload — see Node2Vec note below for why it
isn't preloaded yet). The picker-class branch is a runtime
`is_index_based_picker(edge_picker_type)` in the scheduler; at kernel
level it's a compile-time template tag via `dispatch_picker_type`.

The `NODE_GROUPED_GLOBAL_ONLY` ablation sets both caps to `-1` in the
scheduler, so `G ≤ -1` is false for every node (G ≥ 1 for any node with at
least one edge in the traversal direction). Every coop task then routes to
the global sub-tier.

Why key on G, not edge count E? The binary search runs over G-length
arrays (timestamp groups). The weighted `lower_bound` runs over G-length
cumulative weights. The only E-scale work is the single final edge load
per walk, which is not a search and not a candidate for smem. On temporal
graphs at Tempest scale, G is typically O(thousands) on hubs (distinct
timestamps per node) while E can be O(millions) (every appearance of the
node in the stream). Keying on G keeps smem tractable.

### Block-task expansion — mega-hub handling

`expand_block_tasks_kernel` in `scheduler.cu`, called twice per step (once
for `block_smem`, once for `block_global`). Power-law temporal graphs at
Tempest scale produce mega-hubs with W in the hundreds of thousands. One
block processing all such walks would monopolize one SM while others
idle.

Cap: `TRW_NODE_GROUPED_BLOCK_WALK_CAP = 8192`. For any block-tier task
with `walk_count > cap`, expand into `⌈walk_count / cap⌉` disjoint
sub-tasks, each carrying the same `node_id` and a disjoint slice of walks
(`walk_start + k*cap`, `min(cap, remaining)`).

Each thread handles one input task, computes `num_sub_tasks`, reserves a
contiguous output range via a single `atomicAdd(counter, num_sub_tasks)`,
then writes every sub-task. No per-sub-task atomic.

The warp tier is never expanded: warp tasks have W ≤ `T_BLOCK = 255` ≪
`cap = 8192`.

This is **not** a grid-cooperative mechanism. Each split block is fully
independent: writes go to disjoint walk indices, no cross-block
coordination. Each block independently preloads the same panel; L2
absorbs the redundancy after the first block pays DRAM.

### Shared-memory plan

For every unique node serviced by a `*_smem` kernel, the preloaded panel
holds:

- `s_group_offsets[G]` (8 B/group) — per-group offsets into the sorted
  edge array.
- `s_first_ts[G]` (8 B/group) — timestamp of the first edge of each
  group, pre-gathered through `node_ts_sorted_indices` to **kill the
  3-deep dependent-load chain**. The binary-search comparator now does
  one smem load per probe instead of three serialized global loads.

That's **16 B/group** for index pickers. Weighted pickers are budgeted at
24 B/group (+ `s_cum_weights[G]`), but the current kernel body doesn't
preload `s_cum_weights` yet — `pickers::sample_weighted` still reads from
global. The G-cap numbers already reflect the 24 B budget so the preload
can be added without changing caps.

**Not preloaded**: the full edge list, per-edge timestamps, and the final
endpoint fetch. Final edge pick is one uncoalesced global read per walk
— unavoidable on this access pattern, not worth preloading.

Budget:

- Static envelope: **48 KB per block** on every arch sm_70+ without
  `cudaFuncSetAttribute` opt-in.
- Reserved: ~1 KB for the per-block header (broadcast task scalars,
  Philox state, alignment padding).
- Usable panel: **44 KB per block** (`TRW_NODE_GROUPED_SMEM_PANEL_BYTES =
  45056`).
- Per-warp slice (warp tier packs 8 warps/block): 44 KB / 8 = 5.5 KB/warp.

The static envelope is the portability floor — it's the max without
opt-in on every arch sm_70+ (even sm_86/89 where the opt-in ceiling is
only 100 KB). Opting in via `cudaFuncSetAttribute` could push block
budget to ~100 KB on sm_86/89, ~164 KB on A100, ~228 KB on H100 —
roughly doubling G caps — but past ~half the per-SM smem pool you lose
a block per SM (occupancy cliff). Current code is single-pathed to the
static envelope; the escape hatch is a `cudaFuncSetAttribute` call plus
updated G caps if measurement on a given target makes it worthwhile.

### Kernel matrix

After the scheduler's five task lists, the dispatcher
(`core/node_grouped/dispatch.cuh`) launches up to five kernels per step.
Each launch is skipped if its tier count is 0.

| Kernel | Tier | Panel? | Grid / block |
|---|---|---|---|
| `pick_start_edges_kernel` | step-0 unconstrained | — | `⌈num_walks/block⌉`, 256 thr |
| `node_grouped_solo_kernel` | solo | no | `⌈num_solo/block⌉`, 256 thr (1 thread per walk) |
| `node_grouped_warp_smem_kernel` | warp × smem | yes | `⌈num_warp_smem/8⌉`, 256 thr (8 warps, 1 task/warp) |
| `node_grouped_warp_global_kernel` | warp × global | no | same launch shape as warp_smem, no preload |
| `node_grouped_block_smem_kernel` | block × smem | yes | 1 block per task, 256 thr, dyn smem sized per picker |
| `node_grouped_block_global_kernel` | block × global | no | 1 block per task, 256 thr, static smem header only |
| `reverse_walks_kernel` | (post-step) | — | `⌈num_walks/block⌉`, only if walking backward |

Every cooperative kernel is template-specialized on
`<IsDirected, Forward, EdgePickerType>` via the existing `dispatch_bool`
/ `dispatch_picker_type` tag helpers. No runtime branching inside kernel
hot paths.

### Warp-tier synchronization note

The warp kernels (`coop_warp.cuh`) pack 8 warps per block, one task per
warp. Sync discipline is **`__syncwarp()` only, never
`__syncthreads()`**: warps in a block run independent tasks; a block-wide
barrier would deadlock against the idle warps in a partial last block
where `task_id = blockIdx.x * 8 + warp_id` overshoots the task count.
Partial-last-block warps return with all 32 lanes together at the
task-id guard before hitting any sync.

### Node2Vec bypass

`TemporalNode2Vec`'s sampling depends on each walker's own `prev_node`
— the CDF is per-walk, not per-node. Walks sharing a current node can't
share a panel. The dispatcher gates Node2Vec out at entry and routes it
through `per_walk_step_kernel` (one thread per walk per step, no
scheduler, no coop tiers). Dead walks no-op inside `advance_one_walk`
via `is_node_active`, so the filter/compact/sort stages are skipped
entirely.

The four cooperative kernels therefore **assume non-Node2Vec** and carry
no `prev_node` reads on the hot path.

### Walk index invariant

Walks preserve their original `walk_idx` end-to-end. iota →
`cub_partition_flagged` → sort-by-key values → `walk_set[walk_idx]`
direct writes. **Never renumber walks mid-pipeline.** Philox is keyed
on `(base_seed, walk_idx, step_philox_offset(step))`, which is what
makes tier routing RNG-invariant: the same walk produces the same
sequence regardless of whether it landed in solo / warp_smem /
warp_global / block_smem / block_global on a given step.

---

## Picker types

`RandomPickerType` in `data/enums.cuh`:

| Enum | Class | Sampling |
|---|---|---|
| `Uniform` | index | uniform over timestamp groups below cutoff |
| `Linear` | index | linear-decay over (cutoff - group_ts) |
| `ExponentialIndex` | index | exponential over group position from cutoff |
| `ExponentialWeight` | weighted | exponential over per-group cumulative weight |
| `TemporalNode2Vec` | (per-walk) | bypasses scheduler entirely — see above |

"Class" determines which G cap applies (`G_CAP_*_INDEX` vs
`G_CAP_*_WEIGHTED`). `is_index_based_picker()` in `random/pickers.cuh`
is the authority.

`TEST_FIRST` and `TEST_LAST` exist only for tests.

---

## Testing

`temporal_random_walk/test/` (GoogleTest), run with `build/bin/trw_tests`
after `cmake --build build --target trw_tests`:

- `test_temporal_random_walk.cpp` — public-API golden tests.
- `test_node_grouped_parity.cpp` — `FULL_WALK` vs `NODE_GROUPED` walk
  distribution parity (structural, not bit-exact — Philox counter offsets
  differ between the two paths). Asserts walk validity, slot-0 agreement,
  mean-length ratio band. **This is the gate** for any scheduler change.
- `test_node_grouped_w_partition.cpp`,
  `test_node_grouped_g_partition.cpp`,
  `test_node_grouped_walk_tier_routing.cpp`,
  `test_node_grouped_block_task_expansion.cpp` — unit tests for each
  scheduler stage.
- `test_edge_data*`, `test_node_edge_index*`, `test_temporal_graph*`,
  `test_random_picker*` — lower-level component tests.

Don't weaken tests. Fix code, not tests. Distribution tests pass bit-for-bit
against the baseline walk distribution.

---

## Benchmarks & drivers

### `test_run/ablation_streaming`

Temporal streaming over a CSV file sorted by timestamp. Splits the time
range into `num_batches` batches, streams them into a windowed TRW, and
measures (ingest, walk-sample) per batch. Warmup re-runs batch 0
untimed to absorb one-shot CUDA/CUB init costs.

```bash
build/bin/ablation_streaming data/sample_data.csv 1 \
    exponential_index node_grouped 0 1 5 3 80 -1
# args: file use_gpu picker kernel_launch_type is_directed walks_per_node
#       num_batches num_windows max_walk_len timescale_bound
```

`kernel_launch_type` accepts `full_walk`, `node_grouped`, or
`node_grouped_global_only`.

### `test_run/test_alibaba_streaming`

C++ port of `tempest-benchmarks/alibaba_benchmark/test_alibaba_dataset.py`,
for running the three-variant ablation without Python-side overhead in the
timed loop. Expects the dataset as `data_{0..total-1}.csv` (header +
`u,i,ts` columns). Convert from parquet once upstream; pyarrow→csv on the
60-file Alibaba dump takes ~18 s.

```bash
build/bin/test_alibaba_streaming /path/to/alibaba_csv_dir \
    1 exponential_index node_grouped 20 3 1800000 100 60 -1
# args: dataset_dir use_gpu picker kernel_launch_type walks_per_node
#       minutes_per_step window_ms max_walk_len total_minutes timescale_bound
```

Laptop VRAM cap: a 30-min window on the full Alibaba trace hits ~85 M
active edges around minute 21, which exceeds the 8 GB RTX 2000 Ada. Run
with `total_minutes=21` on this laptop; the A40 (48 GB) handles the full
60-minute trace.

### `tempest-benchmarks/alibaba_benchmark/` (Python side)

- `test_alibaba_dataset.py` — the original Python streaming bench. Has
  Python-side overhead in the timed path. Prefer the C++ port above for
  paper measurements.
- `laptop_bench.py` — reduced-scale variant for the 8 GB laptop.
- `profile_one.py` — single timed walk call under `nsys` for NVTX range
  inspection.

---

## Architecture guardrails

Standing rules. Respected going forward.

- **Buffer-backed CUB scratch.** All temp storage is `Buffer<uint8_t>`
  or `DeviceArena`-allocated — no manual `cudaMalloc` / `cudaFree` in
  kernel-surrounding code.
- **Stream-aware everywhere.** All kernels, CUB calls, allocations go
  through `trw->stream()` or the caller-provided stream.
- **Tag-dispatch for runtime booleans/enums.** Use `dispatch_bool` and
  `dispatch_picker_type`; do not branch inside kernel hot paths.
- **One responsibility per kernel.** Filter, gather, sort, RLE, scatter,
  partition, expand, pick — all separate. No mega-kernel.
- **Preserve `walk_idx` end-to-end.** Never renumber to a dense range
  mid-pipeline.
- **`walk_padding_value` from `WalkSetView`**, never hardcode -1.
- **`#ifdef HAS_CUDA` around CUDA-only code.** CPU build must compile
  and run FULL_WALK paths.
- **No bloated functions.** If a function passes ~100 lines, stop and
  decompose.
- **Don't weaken tests.** Fix code, not tests.
- **No half-finished implementations.** Either a working kernel or a
  clearly-marked scaffold copy. No silently-broken code.
- **Commit per task boundary.** Each commit leaves the code buildable
  and testable.

---

## Ablation variant

`NODE_GROUPED_GLOBAL_ONLY` in `enums.cuh` exists as the middle arm of the
paper's three-way ablation:

```
FULL_WALK                — baseline, one thread per walk
NODE_GROUPED_GLOBAL_ONLY — COOP_WITHOUT_SMEM (cooperation, no panel preload)
NODE_GROUPED             — COOP_WITH_SMEM (full pipeline)
```

Implementation is a single flag threaded through
`dispatch_node_grouped_kernel → NodeGroupedScheduler::run_step`. When set,
both G caps are forced to `-1` so the G-partition sends every coop task to
the `*_global` sub-tier. The `*_smem` kernels never launch; the `*_global`
kernels do exactly what they do in the default path. Walk distribution is
identical to `NODE_GROUPED` (same Philox keying, same sampling math).

Reading: the delta `FULL_WALK → NODE_GROUPED_GLOBAL_ONLY` shows what
cooperation alone buys (or costs — scheduler overhead is real). The
delta `NODE_GROUPED_GLOBAL_ONLY → NODE_GROUPED` shows what smem preload
buys on top.

On a workload where most hub G values exceed the block cap (Alibaba at
30-min window on this laptop), the two NG variants collapse to within 1 %:
the smem tier barely fires, so disabling it costs nothing. Workloads where
G distribution crosses the cap for a non-trivial fraction of hubs (narrower
windows, denser synthetic graphs, TGBL-wiki) are where the three-way split
tells a story.
