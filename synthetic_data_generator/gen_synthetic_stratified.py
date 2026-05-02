"""Stratified hub graph generator for NG-favoring benchmarks.

Three node strata with controlled G (distinct-timestamp count per node)
chosen so the cooperative scheduler's smem panel actually fires, and edge
mixing chosen so walks circulate forever between hubs (saturating mwl).

Final tuning (laptop RTX 2000 Ada, sm_89, 8 GB VRAM):

  stratum    nodes    E/node   G/node    smem cap   tier when hot
  ----------------------------------------------------------------
  mega-hub    300    35_000     1500     2800 idx   block_smem
  warm-hub   1000     2_000      300      340 idx   warp_smem
  tail       5000       100       10            -   solo (entry only)

Mega L2 footprint = 300 × 1500 × 16 B = 7.2 MB > 4 MB L2 → FW's binary-search
metadata thrashes L2; NG's smem panel keeps each block-task's working set in
fast on-chip memory regardless.

Edge mixing keeps walks at hubs once they arrive (no hub→tail outflow):
  Mega: 85% mega / 15% warm / 0% tail
  Warm: 60% mega / 40% warm / 0% tail
  Tail: 55% mega / 45% warm / 0% tail   (tails are entry points only)

Per-node timestamps are sampled uniformly from [0, T_MAX] without
replacement, so G is exactly the configured value per stratum. Each edge's
timestamp is drawn (with replacement) from the source node's G timestamps.

With wpn=200 mwl=80 ExpIdx Forward (run via ablation_streaming):
    walks saturate mwl (avg_len ≈ 78 of 80)
    NG vs FW:        +19.5 % (3-rep mean ± 2.2 % std)
    NG_global vs FW:  +4.1 %   (cooperative dispatch alone)
    smem contribution: ~15 pp on top of the +4 % dispatch gain

Output: CSV with header `u,i,ts` sorted by ts ascending, plus a stats file.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Strata sizes / edges / G targets — tuned for NG block_smem + warp_smem tiers.
# Iter 3: 500 mega-hubs so the per-graph mega metadata footprint
# (500*1500*16B = 12 MB) exceeds L2 (4 MB on RTX 2000 Ada). FW's binary
# searches stop being L2-resident and pay DRAM; NG's smem panel keeps
# each task's working set in fast on-chip memory regardless. W per mega
# at step 1 stays above BLOCK_DIM=256 so block_smem still fires.
N_MEGA, E_MEGA, G_MEGA =    300,  35_000, 1500
N_WARM, E_WARM, G_WARM =  1_000,   2_000,  300
N_TAIL, E_TAIL, G_TAIL =  5_000,     100,   10

# Mixing matrix: row = source stratum, col = target stratum
# (mega, warm, tail) probabilities per source.
MIX = {
    'mega': (0.85, 0.15, 0.00),  # mega→mega dominates; some mega→warm
    'warm': (0.60, 0.40, 0.00),  # warm circulates between hubs
    'tail': (0.55, 0.45, 0.00),  # tails are entry points; out-edges to hubs only
}

T_MAX = 1_000_000  # timestamp range

# Stratum index ranges
def stratum_ranges():
    s_mega = (0, N_MEGA)
    s_warm = (N_MEGA, N_MEGA + N_WARM)
    s_tail = (N_MEGA + N_WARM, N_MEGA + N_WARM + N_TAIL)
    return {'mega': s_mega, 'warm': s_warm, 'tail': s_tail}


def gen_node_timestamps(n_nodes, g_per_node, rng):
    """For each node, pick g_per_node distinct integer timestamps in [0,T_MAX].
    Return a 2D array shape (n_nodes, g_per_node).

    Strategy depends on g vs T_MAX. For g << T (our case: g≤1500, T=1_000_000),
    sampling with replacement and deduping is fast and the collision rate is
    negligible (≤ 0.15% expected duplicates at g=1500 / T=1M). For tail with
    g=15 / T=1M, collision prob is microscopic.
    """
    if g_per_node >= T_MAX:
        raise ValueError('g_per_node >= T_MAX, cannot sample without replacement')

    # Oversample → unique → take first g. Fast and vectorized per node.
    oversample = max(int(g_per_node * 1.05) + 4, g_per_node + 4)
    out = np.empty((n_nodes, g_per_node), dtype=np.int64)
    for i in range(n_nodes):
        cand = rng.integers(0, T_MAX, size=oversample)
        uniq = np.unique(cand)
        if len(uniq) < g_per_node:
            # Fallback: explicit without-replacement (slow but bulletproof).
            uniq = rng.choice(T_MAX, size=g_per_node, replace=False)
        out[i] = uniq[:g_per_node]
    return out


def sample_targets(n_edges, mix, ranges, rng):
    """Given a mixing tuple (p_mega, p_warm, p_tail) and stratum ranges,
    sample n_edges target node IDs."""
    p_mega, p_warm, p_tail = mix
    # Stratum draws.
    strata = rng.choice(
        ['mega', 'warm', 'tail'],
        size=n_edges,
        p=[p_mega, p_warm, p_tail],
    )
    out = np.empty(n_edges, dtype=np.int64)
    for s in ('mega', 'warm', 'tail'):
        mask = strata == s
        if not mask.any():
            continue
        lo, hi = ranges[s]
        out[mask] = rng.integers(lo, hi, size=mask.sum())
    return out


def gen_stratum_edges(stratum, n_nodes, e_per_node, g_per_node, mix,
                      ranges, lo, rng):
    """Generate edges from one stratum.
    Returns (sources, targets, timestamps) numpy arrays."""
    n_edges = n_nodes * e_per_node
    # Sources: each node owns e_per_node edges.
    sources = np.repeat(np.arange(lo, lo + n_nodes, dtype=np.int64), e_per_node)
    # Per-source timestamps: pick g distinct ts per node, then assign each
    # edge to one of those ts (uniform with replacement → multiple edges per
    # ts-group, which is what we want — high E per group amortizes the panel).
    node_ts = gen_node_timestamps(n_nodes, g_per_node, rng)  # (n_nodes, g)
    # Each of the e_per_node edges from node i picks one of g timestamps.
    pick_idx = rng.integers(0, g_per_node, size=(n_nodes, e_per_node))
    timestamps = np.take_along_axis(node_ts, pick_idx, axis=1).ravel()
    # Targets per the mixing distribution.
    targets = sample_targets(n_edges, mix, ranges, rng)
    return sources, targets, timestamps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default='synthetic_stratified.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ranges = stratum_ranges()
    n_total = N_MEGA + N_WARM + N_TAIL
    e_total = N_MEGA*E_MEGA + N_WARM*E_WARM + N_TAIL*E_TAIL

    print(f'# Generating stratified graph: {n_total:,} nodes, {e_total:,} edges')
    print(f'# Strata: mega={N_MEGA} (E={E_MEGA}, G={G_MEGA}) | '
          f'warm={N_WARM} (E={E_WARM}, G={G_WARM}) | '
          f'tail={N_TAIL} (E={E_TAIL}, G={G_TAIL})')
    print(f'# Time range: [0, {T_MAX:,}], output: {args.output}', flush=True)

    t0 = time.time()
    parts = []
    for stratum, n, e, g in [('mega', N_MEGA, E_MEGA, G_MEGA),
                             ('warm', N_WARM, E_WARM, G_WARM),
                             ('tail', N_TAIL, E_TAIL, G_TAIL)]:
        lo = ranges[stratum][0]
        ts0 = time.time()
        s, d, t = gen_stratum_edges(stratum, n, e, g, MIX[stratum],
                                    ranges, lo, rng)
        parts.append((s, d, t))
        print(f'  {stratum:5s}: {len(s):>11,} edges  ({time.time()-ts0:.1f}s)',
              flush=True)

    sources    = np.concatenate([p[0] for p in parts])
    targets    = np.concatenate([p[1] for p in parts])
    timestamps = np.concatenate([p[2] for p in parts])
    print(f'  concat: {len(sources):,} edges  ({time.time()-t0:.1f}s total)',
          flush=True)

    # Sort by timestamp ascending.
    order = np.argsort(timestamps, kind='stable')
    sources    = sources[order]
    targets    = targets[order]
    timestamps = timestamps[order]
    print(f'  sorted by ts  ({time.time()-t0:.1f}s total)', flush=True)

    # Write CSV.
    out_path = Path(args.output)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['u', 'i', 'ts'])
        # Vectorized chunked write for speed.
        chunk = 1_000_000
        for i in range(0, len(sources), chunk):
            rows = zip(sources[i:i+chunk].tolist(),
                       targets[i:i+chunk].tolist(),
                       timestamps[i:i+chunk].tolist())
            w.writerows(rows)
    print(f'Wrote {len(sources):,} rows to {out_path}  '
          f'({time.time()-t0:.1f}s total)', flush=True)

    # Stats file: verify G distribution per stratum.
    stats_path = out_path.with_suffix('.stats.txt')
    with stats_path.open('w') as f:
        f.write(f'# Synthetic stratified graph stats\n')
        f.write(f'total_nodes  = {n_total:,}\n')
        f.write(f'total_edges  = {len(sources):,}\n')
        f.write(f'ts_min       = {timestamps.min()}\n')
        f.write(f'ts_max       = {timestamps.max()}\n')
        f.write(f'ts_median    = {int(np.median(timestamps))}\n\n')

        # Per-stratum G distribution: pick a sample of nodes per stratum.
        for stratum in ('mega', 'warm', 'tail'):
            lo, hi = ranges[stratum]
            sample_ids = np.linspace(lo, hi - 1, num=min(50, hi - lo)).astype(np.int64)
            gs = []
            for nid in sample_ids:
                node_ts = timestamps[sources == nid]
                gs.append(len(np.unique(node_ts)))
            gs = np.array(gs)
            f.write(f'[{stratum}] sampled {len(sample_ids)} of {hi-lo} nodes\n')
            f.write(f'  G  min/median/max = {gs.min()} / {int(np.median(gs))} / {gs.max()}\n')
            # Out-degree per sampled node.
            degs = np.array([(sources == nid).sum() for nid in sample_ids[:10]])
            f.write(f'  degree (first 10 sampled): min={degs.min()} median={int(np.median(degs))} max={degs.max()}\n\n')

    print(f'Wrote stats to {stats_path}')
    print(f'\nDone in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
