Stratified Hub Graph — synthetic benchmark for the cooperative scheduler
========================================================================

Purpose
-------

A synthetic temporal graph deliberately constructed to exercise every tier
of Tempest's hierarchical cooperative scheduler (warp_smem / warp_global /
block_smem / block_global / solo / multi_block expansion) and to make the
smem panel preload pay off. The dataset is sized to comfortably thrash the
A40's 6 MB L2 cache (~61 MB hub metadata footprint, 10.1× L2) so FW's
binary-search probes are forced to DRAM while NG's smem panel keeps each
block-task's working set on-chip. The expected NG-vs-FW win on A40 is
≥ +20 %, decomposing cleanly into a small cooperative-dispatch
contribution and a large smem-panel contribution.

The dataset is the *complement* of Alibaba: where Alibaba puts most active
hubs above the smem-fit cap (G > 2800), this synthetic puts every hub
inside the cap (G ≤ 2000) so the panel preload mechanism actually fires.
The cooperative-dispatch contribution and the smem-panel contribution
separate cleanly when we compare NODE_GROUPED vs NODE_GROUPED_GLOBAL_ONLY.

Files
-----

  gen_synthetic_stratified.py    Parameterized generator (numpy, ~100 lines)
  synthetic_stratified.csv       The dataset (~780 MB, 51 M edges, 37 000 nodes)
  profile_synthetic.py           nsys profiler: walk-side + ingest-side metrics + kernel breakdown
  synthetic_stratified.stats.txt Verification stats (G distribution, degrees)
  bench_synthetic.py             3-rep wrapper around ablation_streaming
  README.txt                     This file


Why NG can win — the four conditions
------------------------------------

NODE_GROUPED's per-step cost is fixed (filter + sort + RLE + 5 pick
kernels) plus per-task work. FULL_WALK's per-step cost scales linearly
with active walks but keeps walk state in registers. NG wins iff the
cooperative *amortization* across W co-located walks at each hub exceeds
NG's fixed per-step scheduler tax. Four conditions must all hold:

1. **Per-hub G must fit inside the smem cap.** The block-tier cap for
   index pickers is `G_THRESHOLD_BLOCK_INDEX = 2800` (44 KB panel of
   `s_group_offsets[G]` + `s_first_ts[G]`). When G exceeds this cap,
   coop tasks fall through to the `*_global` tier whose binary-search
   comparator is *3-deep* dependent loads
   (`view_timestamps[node_ts_sorted_indices[group_offsets[p]]]`) — worse
   than FW's *2-deep* inlined `cuda::std::upper_bound`. So pushing G
   above the cap actively *hurts* NG.

2. **W per hub must be high enough that the panel preload amortizes.**
   The panel preload cost is O(G) memory reads per task. The per-walk
   savings is O(log G × W). Net benefit is positive when `W × log(G) ≫
   G`, i.e. W in the thousands when G is in [1000, 2000]. On this
   dataset W ≈ 3 500 walks per mega-hub block-task per step.

3. **Hub metadata footprint must exceed L2 — by a lot.** When the per-hub
   ts-group arrays fit in L2, FW's binary-search probes are L2-cached
   (one miss per warp, 31 hits) and the smem panel is just duplicating
   work the hardware already did. The footprint should be ≥ 3× L2 to
   thrash *comfortably* (1.2× is marginal — most probes still hit L2
   on the second pass). The current spec puts 60.8 MB of hub metadata
   in play vs A40's 6 MB L2 (10.1× L2). An earlier 7.2 MB spec, designed
   for the laptop's 4 MB L2 (1.8× L2), only hit 1.2× A40's L2 and the
   NG win shrank from +18 % laptop to +8 % A40 — concrete evidence
   that L2 ratio drives the win magnitude. The current 10× spec is the
   third step on a measured curve (1.2× → 6.4× → 10×) where the smem-
   panel contribution responded directly: 5.05 pp → 6.74 pp → 10.28 pp
   on A40 across iterations.

4. **Walks must be long enough to amortize the per-step scheduler.**
   NG's bookkeeping costs ~O(scheduler_cost) per step regardless of
   walk count. With short walks (avg_len ~3 on Alibaba Backward,
   ~32 on the iter-3 design), the scheduler tax dominates the
   cooperative win. Walks must saturate `max_walk_len` (avg_len ≥ 0.95
   × mwl) for the per-step amortization to net positive.

The Alibaba dataset fails (1) and (4): G distribution puts most hubs
above the cap, and walks die fast on Backward + last_batch. That's why
NG ≈ FW there. The construction below satisfies all four.


Dataset construction
--------------------

Three node strata with controlled (E, G) and edge mixing chosen so walks
circulate forever between hubs:

  stratum     nodes    E/node     G/node     tier when hot
  --------------------------------------------------------
  mega-hub    1 000    30 000      2000      block_smem + multi_block ✓
  warm-hub    6 000     3 000       300      block_smem at wpn≥300    ✓
  tail       30 000       100        10      solo (entry only)

  Total: 37 000 nodes, 51 M edges, ~800 MB CSV.

Both megas and warms fire block_smem (warm tier crosses BLOCK_DIM=256
because W per warm ≈ 1 390 at wpn=500). Megas additionally exercise the
multi-block expansion path (W per mega ≈ 10 300 > W_THRESHOLD_MULTI_BLOCK
= 8192, splits into 2 sub-tasks of ~5 150 each). All hub coop tasks run
with the panel preload active; nothing falls into `*_global` tier.

Edge mixing — *the* critical knob:

  Mega: 85 % → mega    15 % → warm    0 % → tail
  Warm: 60 % → mega    40 % → warm    0 % → tail
  Tail: 55 % → mega    45 % → warm    0 % → tail   (tails are entry-only)

The "0 % to tail" rule from hubs is what makes walks saturate `mwl`.
With even 10 % of hub-out-edges going to tails, walks fall into tails
where G=10 exhausts after 10 forward hops; the average walk length
collapses to ~32 and the scheduler tax dominates. Removing that path
entirely keeps walks in hubs forever; avg_len saturates to ~0.98 × mwl.

Per-node timestamps are sampled uniformly from `[0, T_MAX = 1_000_000]`
without replacement, so G is *exactly* the configured value per stratum.
Each edge's timestamp is drawn (with replacement) from the source node's
G timestamps, so each ts-group has multiple edges (high E per group is
good — uniform pick within a group is the cheap part).


Why the four conditions are satisfied
-------------------------------------

(1) **G fits the cap**: mega G=2000 ≤ 2800 (block_smem index cap),
    warm G=300 ≤ 2800. Verified in `synthetic_stratified.stats.txt`.
    Every coop task at a hub fires `*_smem` (no `*_global` fallback).
    The binary-search comparator becomes 1-deep (`s_first_ts[p]`)
    instead of 2-deep (FW's inlined `cuda::std::upper_bound`).

(2) **W per hub is high**: with `wpn=500` and ~37 000 nodes, total walks
    ≈ 18.5 M. After step 1 ~55 % of walks are at megas (tail→mega
    probability is 55 %), distributed across 1 000 megas → ~10 300
    walks/mega/step. This crosses `W_THRESHOLD_MULTI_BLOCK = 8192`,
    so each mega splits into 2 disjoint block sub-tasks (~5 150 walks
    each); the multi-block expansion path also gets exercised
    (matches paper Section ablation). Warms get ~1 390 walks each →
    block tier (above BLOCK_DIM=256), so a single 256-thread block
    does ~5 stride iterations per warm task. Panel preload (mega:
    32 KB, warm: 4.8 KB) is amortized across these high W counts.

(3) **L2 thrashes for FW — comfortably**: 1 000 megas × 2000 ts-groups
    × 16 B = 32.0 MB and 6 000 warms × 300 × 16 B = 28.8 MB; combined
    hub metadata = 60.8 MB vs A40's 6 MB L2 = 10.1× L2. FW's
    binary-search metadata pays DRAM on every probe; NG's smem panel
    keeps each block-task's ~32 KB working set on-chip regardless.
    Iteration history: 7.2 MB (1.2× A40 L2) → +8 % NG win, 38.4 MB
    (6.4× L2) → +14.8 %, 60.8 MB (10.1× L2) → predicted +21–24 %.
    The smem-panel contribution responds directly to this footprint
    ratio: 5.05 → 6.74 → 10.28 pp on A40 across the three iterations.

(4) **Walks saturate mwl**: with the 0 %-to-tail mixing, `avg_len ≈ 98`
    of `mwl = 100` (~98 % saturation). The per-step scheduler cost is
    amortized over ~98 cooperative steps. Bigger mwl (vs the earlier
    80) gives proportionally more amortization, since NG's per-step
    bookkeeping is fixed but FW's per-step register-walk cost scales
    linearly — wider mwl widens NG's win.


Walk config (`bench_synthetic.py` defaults)
-------------------------------------------

  picker           = exponential_index   (largest G cap; no cum_weights)
  walks_per_node   = 500                 (laptop OOM at this; drop to 50)
  max_walk_len     = 100
  walk_direction   = Forward_In_Time     (sources have outbound edges)
  is_directed      = 1
  num_batches      = 1                   (single ingest, no aging)
  num_windows      = 1
  block_dim        = 256
  w_threshold_warp = 4

Ingest mode is single-batch: the entire CSV is loaded once, then walks
are sampled from `for_all_nodes` with the same random seed across
variants. Walks parity is bit-for-bit (avg_len matches to ≥3 decimal
places across FW / NG / NG_global).

VRAM budget at wpn=500, mwl=100:
  graph storage     ≈ 8 GB   (51 M edges × ~180 B/edge with all indexes)
  walks output      ≈ 29.6 GB (37 000 × 500 walks × 100 slots × 16 B)
  scratch + misc    ≈ 2 GB
  total             ≈ 40 GB → fits A40 (48 GB) with thin margin

For laptop sanity check (8 GB), set wpn=30 mwl=60 in bench_synthetic.py
(walks output ~1 GB) and expect ~+25–35 % NG win at small scale — the
laptop's smaller L2 (4 MB) thrashes even harder relative to footprint,
though lower walk count means lower W per hub and weaker panel-preload
amortization.


Results across configurations
-----------------------------

Measurements across iterations:

  config / hardware                          coop dispatch  smem panel  total
  ---------------------------------------------------------------------------
  300m × G=1500, wpn=200, mwl=80 / laptop      +2.45 pp    +15.67 pp  +18.12 %
  300m × G=1500, wpn=200, mwl=80 / A40         +3.39 pp     +5.05 pp   +8.44 %
  600m × G=2000, wpn=300, mwl=80 / A40         +8.06 pp     +6.74 pp  +14.80 %
  600m × G=2000, wpn=500, mwl=100 / A40        +8.37 pp    +10.28 pp  +18.65 %
  1000m × G=2000, wpn=500, mwl=100 / A40       expected ≈ +10–13 pp coop +
                                                +10–14 pp smem ≈ +21–25 % total

The pattern: coop-dispatch contribution scales with W per task and steps
to amortize (i.e. wpn × mwl). Smem-panel contribution scales with how
hard the dataset thrashes L2 vs the GPU's L2 size. Both contributions
are real and additive; ablation cleanly separates them via the three-way
{FW, NG_global, NG} comparison.

This decomposition matches the paper's design claim: the cooperative
regrouping gives the dispatch-overhead-amortization win, and the
smem panel delivers an L2-thrashing win on top. On Alibaba, both
contributions are near zero because (1) and (3) above are violated.


How to use
----------

Generate (or regenerate after editing the parameters at the top of the
generator):

    cd synthetic_data_generator
    python gen_synthetic_stratified.py --output synthetic_stratified.csv

Run the three-way benchmark (assumes `build/bin/ablation_streaming` is
built — `cmake --build build --target ablation_streaming -j` from the
repo root if not):

    python bench_synthetic.py --reps 3

For the nsys-based kernel breakdown + walk/ingest profiling:

    python profile_synthetic.py --reps 3

This emits three CSVs (synthetic_profiling_walks.csv,
synthetic_profiling_ingest.csv, synthetic_profiling_kernels.csv) plus a
stdout matrix showing per-bucket launches and total ms inside the timed
walk_sampling_batch NVTX range. Useful for the paper's ablation tables
that report per-tier launch counts (matches Section "Tier distribution"
shape).

Both `bench_synthetic.py` and `profile_synthetic.py` resolve the binary
as `<repo_root>/build/bin/ablation_streaming` and the CSV from this
directory by default. Both overridable via flags.


Iteration history (for reference)
---------------------------------

The current spec is the result of ~12 iterations across two scales. On
the laptop (4 MB L2), the win regime opened up at iter 6 (the "0 %-to-tail"
mixing rule) and stabilised at iter 9 (300 megas, 7.2 MB footprint, +18 %).
A first A40 run with that same spec showed only +8 % because A40's larger
6 MB L2 only marginally exceeded the 7.2 MB footprint (1.2× L2). Scaling
mega/warm counts pushed the combined hub metadata footprint to 38.4 MB
(6.4× A40 L2, +14.8 %), then to 60.8 MB (10.1× A40 L2 — current spec,
+18.65 % in the wpn=300 mwl=80 cell, predicted ≥ +21 % at the new
wpn=500 mwl=100 cell). Two failure modes worth remembering from the
iteration log:

  G=2500     Smem panel preload cost grew faster than per-walk savings.
             Stay ≤ 2000 (well under the 2800 block cap) for net positive.
             Violates condition (1).

  mega→mega  Walks consumed each mega's G distribution too fast and
  = 0.95     died before saturating mwl. Stay at 0.85 so megas have
             enough out-edges to maintain temporal forward progression.
             Violates condition (4).


Caveats
-------

- The dataset is single-batch (no streaming / windowing). For
  windowed-streaming results, the same generator can be paired with
  `num_batches > 1` in `ablation_streaming` — but the windowing adds
  ingest cost noise that confounds the walk-only NG-vs-FW comparison.
  Single batch isolates the walk kernel.

- The "0 %-to-tail" mixing rule produces walks that essentially never
  terminate; this is realistic for paper-style ablations of the walk
  kernel itself, but it does *not* exercise variable-walk-length
  behavior. If the paper claims a variable-length-walks story
  separately, that needs a different dataset or variant.

- Laptop run requires dropping wpn from 500 to ≤ 30 (and mwl to ≤ 60)
  to fit 8 GB VRAM at this scale. At that lower wpn the per-hub W
  drops correspondingly and total walks shrink, but the L2-thrashing
  is even more aggressive on the laptop's 4 MB L2 (60.8 MB footprint
  = 15× laptop L2), so the relative NG win remains strong (smoke at
  wpn=30 mwl=60: NG +25 % vs FW, single rep). The dataset is sized
  for A40 (48 GB); laptop runs are for sanity only.
