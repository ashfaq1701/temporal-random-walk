Stratified Hub Graph — synthetic benchmark for the cooperative scheduler
========================================================================

Purpose
-------

A synthetic temporal graph deliberately constructed to exercise every tier
of Tempest's hierarchical cooperative scheduler (warp_smem / warp_global /
block_smem / block_global / solo / multi_block expansion) and to make the
smem panel preload pay off. The dataset is sized to comfortably thrash the
A40's 6 MB L2 cache (~38 MB hub metadata footprint, 6.4× L2) so FW's
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
  synthetic_stratified.csv       The dataset (~330 MB, 27 M edges, 24 600 nodes)
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
   on the second pass). The current spec puts 38.4 MB of hub metadata
   in play vs A40's 6 MB L2 (6.4× L2). An earlier 7.2 MB spec, designed
   for the laptop's 4 MB L2 (1.8× L2), only hit 1.2× A40's L2 and the
   NG win shrank from +18 % laptop to +8 % A40 — concrete evidence
   that L2 ratio drives the win magnitude.

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
  mega-hub      600    25 000      2000      block_smem (cap 2800)  ✓
  warm-hub    4 000     2 500       300      block_smem at wpn=300  ✓
  tail       20 000       100        10      solo (entry only)

  Total: 24 600 nodes, 27 M edges, ~420 MB CSV.

Both megas and warms fire block_smem (warm tier crosses BLOCK_DIM=256
because W per warm ≈ 830 at wpn=300). All hub coop tasks run with the
panel preload active; nothing falls into `*_global` tier.

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

(2) **W per hub is high**: with `wpn=300` and ~24 600 nodes, total walks
    ≈ 7.4 M. After step 1 ~55 % of walks are at megas (tail→mega
    probability is 55 %), distributed across 600 megas → ~6 800
    walks/mega/step (single block per mega, < W_THRESHOLD_MULTI_BLOCK
    = 8192). Warms get ~830 walks each → block tier (above BLOCK_DIM
    = 256), so a single 256-thread block does ~3 stride iterations
    per warm task. Panel preload (mega: 32 KB, warm: 4.8 KB) is
    amortized across these high W counts.

(3) **L2 thrashes for FW — comfortably**: 600 megas × 2000 ts-groups
    × 16 B = 19.2 MB and 4 000 warms × 300 × 16 B = 19.2 MB; combined
    hub metadata = 38.4 MB vs A40's 6 MB L2 = 6.4× L2. FW's
    binary-search metadata pays DRAM on every probe; NG's smem panel
    keeps each block-task's ~32 KB working set on-chip regardless.
    The earlier laptop spec (300 megas × 1500 G = 7.2 MB) hit only
    1.2× A40's L2 and produced a +8 % win there — confirming the L2
    ratio drives win magnitude.

(4) **Walks saturate mwl**: with the 0 %-to-tail mixing, `avg_len ≈ 78`
    of `mwl = 80` (~98 % saturation). The per-step scheduler cost is
    amortized over ~78 cooperative steps.


Walk config (`bench_synthetic.py` defaults)
-------------------------------------------

  picker           = exponential_index   (largest G cap; no cum_weights)
  walks_per_node   = 300                 (laptop OOM at this; drop to 80)
  max_walk_len     = 80
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

VRAM budget at wpn=300:
  graph storage     ≈ 5 GB  (27 M edges × ~180 B/edge with all indexes)
  walks output      ≈ 9.4 GB (24 600 × 300 walks × 80 slots × 16 B)
  total             ≈ 14.5 GB → fits A40 (48 GB) with margin

For laptop sanity check (8 GB), set wpn=80 in bench_synthetic.py and
expect ~+10–15 % NG win — the laptop's smaller L2 (4 MB) thrashes even
harder relative to footprint, but lower walk count means lower W per
hub and weaker panel-preload amortization.


Expected results on A40
-----------------------

Predicted from the L2-ratio extrapolation:

  laptop (4 MB L2, 7.2 MB hub footprint = 1.8×):  +18 % NG vs FW
  earlier A40 spec (6 MB L2, 7.2 MB = 1.2×):       +8 % NG vs FW
  current  A40 spec (6 MB L2, 38.4 MB = 6.4×):    +20 % or higher

Mechanism decomposition (target):

  cooperative dispatch alone:           ~+3 pp   (NG_global vs FW)
  smem panel adds on top:              ~+18 pp   (NG vs NG_global)
  total NG vs FW:                      ~+21 pp

This decomposition matches the paper's design claim: the cooperative
regrouping gives a small dispatch-overhead-amortization win, and the
smem panel delivers the bulk of the speedup by keeping each hub's
ts-group metadata on-chip. On Alibaba, both contributions are near
zero because (1) and (3) above are violated.


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

`bench_synthetic.py` resolves the binary as `<repo_root>/build/bin/
ablation_streaming` and the CSV from this directory by default. Both
overridable via flags.


Iteration history (for reference)
---------------------------------

The current spec is the result of ~10 iterations across two scales. On
the laptop (4 MB L2), the win regime opened up at iter 6 (the "0 %-to-tail"
mixing rule) and stabilised at iter 9 (300 megas, 7.2 MB footprint, +18 %).
A first A40 run with that same spec showed only +8 % because A40's larger
6 MB L2 only marginally exceeded the 7.2 MB footprint (1.2× L2). The
current spec scales mega and warm counts to push the combined hub
metadata footprint to 38.4 MB (6.4× A40 L2). Two failure modes worth
remembering from the iteration log:

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

- Laptop run requires dropping wpn from 300 to ≤ 80 to fit 8 GB VRAM.
  At that lower wpn the per-hub W drops correspondingly (~1800 walks
  per mega) and the win shrinks; expect ~+10 % on laptop vs ≥ +20 %
  on A40. The dataset is sized for A40; laptop runs are for sanity.
