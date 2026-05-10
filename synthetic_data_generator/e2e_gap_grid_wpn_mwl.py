"""End-to-end FW-vs-NG gap on a 2D wpn × mwl grid for Hub-Synthetic.

Hypothesis under test: the cooperative scheduler's gain is driven by
walks-per-hub-per-step (W density), which scales linearly with --wpn,
NOT by walk length (mwl) past a small saturation point. Cross-dataset
gain differences in the paper are therefore attributable to graph
structure (hub skew, G distribution, walk concentration), not walk
length.

The 2D grid makes this visible in one shot:

  - wpn axis (the proposed driver): {10, 50, 100, 200, 300, 500}
  - mwl axis (the alleged driver): {20, 50, 100}, all in the saturated
    regime past mwl=10.

If wpn is the real driver, the curves at different mwls should overlap
when plotted as gap_pct vs wpn, and gap_pct should increase
monotonically with wpn until W per hub saturates the block-tier panel.

For each (wpn, mwl) cell:
  1. Run ablation_streaming with FULL_WALK and NODE_GROUPED, --reps each
     (default 5).
  2. Reject reps whose steps/sec deviates from the running median by
     more than --outlier-threshold (default 0.10), iteratively, until
     --min-keep reps remain (default 3).
  3. Mean+std on the kept reps.
  4. Compute gap = (NG_sps_mean - FW_sps_mean) / FW_sps_mean.

Final output: a long-form DataFrame (one row per cell) plus a pivoted
gap_pct table (rows=wpn, cols=mwl) for at-a-glance inspection.

Path assumption: script lives in
    <root>/temporal-random-walk/synthetic_data_generator/
so binary is at ../build/bin/ablation_streaming relative to here.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# --- Strata identical to gen_synthetic_stratified.py (no sinks). ---
N_MEGA, E_MEGA, G_MEGA =    600,  25_000, 1500
N_WARM, E_WARM, G_WARM =  4_000,   2_500,  300
N_TAIL, E_TAIL, G_TAIL = 20_000,     100,   10
T_MAX = 1_000_000

MIX = {
    'mega': (0.85, 0.15, 0.00),
    'warm': (0.60, 0.40, 0.00),
    'tail': (0.55, 0.45, 0.00),
}

USE_GPU      = '1'
IS_DIRECTED  = '1'
TIMESCALE    = '-1'
NUM_BATCHES  = '1'
NUM_WINDOWS  = '1'
BLOCK_DIM    = '256'

VARIANTS = ['full_walk', 'node_grouped']

THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def stratum_ranges():
    s_mega = (0, N_MEGA)
    s_warm = (N_MEGA, N_MEGA + N_WARM)
    s_tail = (N_MEGA + N_WARM, N_MEGA + N_WARM + N_TAIL)
    return {'mega': s_mega, 'warm': s_warm, 'tail': s_tail}


def gen_node_timestamps(n_nodes, g_per_node, rng):
    if g_per_node >= T_MAX:
        raise ValueError('g_per_node >= T_MAX')
    oversample = max(int(g_per_node * 1.05) + 4, g_per_node + 4)
    out = np.empty((n_nodes, g_per_node), dtype=np.int64)
    for i in range(n_nodes):
        cand = rng.integers(0, T_MAX, size=oversample)
        uniq = np.unique(cand)
        if len(uniq) < g_per_node:
            uniq = rng.choice(T_MAX, size=g_per_node, replace=False)
        out[i] = uniq[:g_per_node]
    return out


def sample_targets_3(n_edges, mix, ranges, rng):
    p_mega, p_warm, p_tail = mix
    strata = rng.choice(['mega', 'warm', 'tail'], size=n_edges,
                        p=[p_mega, p_warm, p_tail])
    out = np.empty(n_edges, dtype=np.int64)
    for s in ('mega', 'warm', 'tail'):
        mask = strata == s
        if not mask.any():
            continue
        lo, hi = ranges[s]
        out[mask] = rng.integers(lo, hi, size=mask.sum())
    return out


def gen_stratum(n_nodes, e_per_node, g_per_node, mix, ranges, lo, rng):
    n_edges = n_nodes * e_per_node
    sources = np.repeat(np.arange(lo, lo + n_nodes, dtype=np.int64),
                        e_per_node)
    node_ts = gen_node_timestamps(n_nodes, g_per_node, rng)
    pick_idx = rng.integers(0, g_per_node, size=(n_nodes, e_per_node))
    timestamps = np.take_along_axis(node_ts, pick_idx, axis=1).ravel()
    targets = sample_targets_3(n_edges, mix, ranges, rng)
    return sources, targets, timestamps


def generate_graph_arrays(seed: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranges = stratum_ranges()
    parts = []
    for stratum, n, e, g in [('mega', N_MEGA, E_MEGA, G_MEGA),
                             ('warm', N_WARM, E_WARM, G_WARM),
                             ('tail', N_TAIL, E_TAIL, G_TAIL)]:
        lo = ranges[stratum][0]
        s, d, t = gen_stratum(n, e, g, MIX[stratum], ranges, lo, rng)
        parts.append((s, d, t))
    sources    = np.concatenate([p[0] for p in parts])
    targets    = np.concatenate([p[1] for p in parts])
    timestamps = np.concatenate([p[2] for p in parts])
    order = np.argsort(timestamps, kind='stable')
    return sources[order], targets[order], timestamps[order]


def write_csv(sources, targets, timestamps, path: Path):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['u', 'i', 'ts'])
        chunk = 1_000_000
        for i in range(0, len(sources), chunk):
            rows = zip(sources[i:i+chunk].tolist(),
                       targets[i:i+chunk].tolist(),
                       timestamps[i:i+chunk].tolist())
            w.writerows(rows)


def parse_summary(stdout: str):
    m_t = THROUGHPUT_RE.search(stdout)
    m_s = STEPS_RE.search(stdout)
    m_a = AVGLEN_RE.search(stdout)
    if not (m_t and m_s and m_a):
        return None
    return float(m_t.group(1)), float(m_s.group(1)), float(m_a.group(1))


def run_binary(binary: Path, csv_path: Path, variant: str,
               wpn: int, mwl: int, picker: str, w: int):
    cmd = [str(binary), str(csv_path),
           USE_GPU, picker, variant, IS_DIRECTED,
           str(wpn), NUM_BATCHES, NUM_WINDOWS, str(mwl),
           TIMESCALE, BLOCK_DIM, str(w)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, proc.stderr[-400:]
    parsed = parse_summary(proc.stdout)
    if parsed is None:
        return None, 'parse failed (stdout tail):\n' + proc.stdout[-400:]
    return parsed, None


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def reject_outliers_by_key(rows, key, threshold_frac, min_keep):
    """Iterative median-relative outlier rejection on rows[i][key]."""
    kept = list(rows)
    dropped = []
    while len(kept) > min_keep:
        vals = [r[key] for r in kept]
        med = statistics.median(vals)
        if med == 0:
            break
        devs = [abs(v - med) / med for v in vals]
        worst_idx = max(range(len(kept)), key=lambda i: devs[i])
        if devs[worst_idx] > threshold_frac:
            dropped.append(kept.pop(worst_idx))
        else:
            break
    return kept, dropped


def main():
    here = Path(__file__).resolve().parent
    default_binary = here.parent / 'build' / 'bin' / 'ablation_streaming'

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wpns', nargs='+', type=int,
                    default=[10, 50, 100, 200, 300, 500],
                    help='walks_per_node sweep (the proposed driver)')
    ap.add_argument('--mwls', nargs='+', type=int,
                    default=[20, 50, 100],
                    help='max_walk_len sweep — should NOT change gap '
                         'past saturation if hypothesis holds')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--outlier-threshold', type=float, default=0.10)
    ap.add_argument('--min-keep', type=int, default=3)
    ap.add_argument('--picker', default='exponential_index')
    ap.add_argument('--w-warp', type=int, default=4,
                    help='w_threshold_warp for NG (FW pinned to 1)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--binary', default=str(default_binary))
    ap.add_argument('--tmp-dir', default=None)
    ap.add_argument('--keep-csv', action='store_true')
    ap.add_argument('--out-csv', default=None,
                    help='Where to write the long-form results CSV. '
                         'Default: stdout only.')
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')

    print(f'# binary       : {binary}')
    print(f'# wpns         : {args.wpns}')
    print(f'# mwls         : {args.mwls}')
    print(f'# reps (raw)   : {args.reps}')
    print(f'# outlier band : ±{args.outlier_threshold*100:.0f}% of running '
          f'median (steps/sec); min-keep={args.min_keep}')
    print(f'# picker       : {args.picker}')
    print(f'# W (NG)       : {args.w_warp}    (FW pinned to 1)')
    print(f'# total cells  : {len(args.wpns) * len(args.mwls)} '
          f'× {len(VARIANTS)} variants × {args.reps} reps = '
          f'{len(args.wpns) * len(args.mwls) * len(VARIANTS) * args.reps} runs')
    print()

    if args.tmp_dir:
        tmp_root = Path(args.tmp_dir)
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix='e2e_grid_')
        tmp_root = Path(tmp_ctx.name)

    rows = []
    try:
        # Generate the graph ONCE — it's the same for every cell.
        print('=== generating sink-free Hub-Synthetic graph (once) ===')
        t0 = time.perf_counter()
        src, dst, ts = generate_graph_arrays(args.seed)
        n_edges = len(src)
        print(f'  gen   : {n_edges:>10,} edges  '
              f'{time.perf_counter()-t0:5.1f}s')

        csv_path = tmp_root / 'hub_synth.csv'
        t0 = time.perf_counter()
        write_csv(src, dst, ts, csv_path)
        print(f'  csv   : {csv_path.stat().st_size/1e6:6.1f} MB  '
              f'{time.perf_counter()-t0:5.1f}s\n')

        for wpn in args.wpns:
            for mwl in args.mwls:
                print(f'=== wpn={wpn}  mwl={mwl} ===')
                per_variant = {}
                for variant in VARIANTS:
                    w = args.w_warp if variant != 'full_walk' else 1
                    rep_records = []
                    for r in range(args.reps):
                        t0 = time.perf_counter()
                        parsed, err = run_binary(
                            binary, csv_path, variant,
                            wpn, mwl, args.picker, w)
                        if parsed is None:
                            print(f'  {variant:<13} rep {r+1}: FAIL — {err}',
                                  file=sys.stderr)
                            continue
                        wps, sps, avg_len = parsed
                        rep_records.append({'sps': sps, 'wps': wps,
                                            'avg_len': avg_len})
                        print(f'  {variant:<13} rep {r+1}/{args.reps}: '
                              f'steps/s={sps/1e6:7.2f}M  '
                              f'walks/s={wps/1e6:5.2f}M  '
                              f'avg_len={avg_len:6.2f}  '
                              f'({time.perf_counter()-t0:4.1f}s)',
                              flush=True)
                    if not rep_records:
                        per_variant[variant] = None
                        continue
                    kept, dropped = reject_outliers_by_key(
                        rep_records, key='sps',
                        threshold_frac=args.outlier_threshold,
                        min_keep=args.min_keep)
                    if dropped:
                        ds = ', '.join(f'{d["sps"]/1e6:.2f}M' for d in dropped)
                        print(f'  {variant:<13} dropped {len(dropped)} '
                              f'outlier(s): {ds}  '
                              f'(kept {len(kept)}/{len(rep_records)})')
                    sps_m, sps_s = mean_std([r['sps']     for r in kept])
                    wps_m, _     = mean_std([r['wps']     for r in kept])
                    len_m, _     = mean_std([r['avg_len'] for r in kept])
                    per_variant[variant] = {
                        'sps_mean': sps_m, 'sps_std': sps_s,
                        'wps_mean': wps_m, 'avg_len': len_m,
                        'n_reps_raw': len(rep_records),
                        'n_reps_kept': len(kept),
                    }

                fw = per_variant.get('full_walk')
                ng = per_variant.get('node_grouped')
                if fw is None or ng is None:
                    print(f'  -- one variant failed; skipping cell\n')
                    continue
                gap_pct = (ng['sps_mean'] - fw['sps_mean']) / fw['sps_mean'] * 100
                print(f'  -- gap NG vs FW: {gap_pct:+6.2f}%   '
                      f'avg_len(FW)={fw["avg_len"]:.2f}\n')

                rows.append({
                    'wpn':         wpn,
                    'mwl':         mwl,
                    'avg_len_fw':  fw['avg_len'],
                    'fw_msps':     fw['sps_mean'] / 1e6,
                    'fw_msps_std': fw['sps_std']  / 1e6,
                    'ng_msps':     ng['sps_mean'] / 1e6,
                    'ng_msps_std': ng['sps_std']  / 1e6,
                    'fw_kept':     f'{fw["n_reps_kept"]}/{fw["n_reps_raw"]}',
                    'ng_kept':     f'{ng["n_reps_kept"]}/{ng["n_reps_raw"]}',
                    'gap_pct':     gap_pct,
                })
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if not rows:
        print('No successful cells.', file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r['wpn'], r['mwl']))

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print('=' * 96)
        print(' Long-form results (sorted by wpn, mwl asc)')
        print('=' * 96)
        with pd.option_context('display.float_format', '{:.3f}'.format,
                               'display.width', 200):
            print(df.to_string(index=False))
        print()

        # Pivot: rows = wpn, cols = mwl, values = gap_pct.
        pivot = df.pivot(index='wpn', columns='mwl', values='gap_pct')
        pivot = pivot.sort_index().sort_index(axis=1)
        print('=' * 96)
        print(' gap_pct heatmap  (rows: wpn,  cols: mwl)')
        print('=' * 96)
        with pd.option_context('display.float_format', '{:+.2f}'.format,
                               'display.width', 200):
            print(pivot.to_string())
        print()

        # Pivot: realized FW avg_len for sanity (should equal mwl in plateau).
        avg_len_pivot = df.pivot(index='wpn', columns='mwl',
                                 values='avg_len_fw')
        avg_len_pivot = avg_len_pivot.sort_index().sort_index(axis=1)
        print('=' * 96)
        print(' avg_len_fw heatmap (sanity: should ≈ mwl)')
        print('=' * 96)
        with pd.option_context('display.float_format', '{:.2f}'.format,
                               'display.width', 200):
            print(avg_len_pivot.to_string())
        print()

        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f'Wrote {args.out_csv}')
    except ImportError:
        print(' (pandas not available; printing long-form only)')
        for r in rows:
            print(f'  wpn={r["wpn"]:>4}  mwl={r["mwl"]:>4}  '
                  f'gap={r["gap_pct"]:+6.2f}%  '
                  f'avg_len={r["avg_len_fw"]:.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
