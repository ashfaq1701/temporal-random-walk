"""End-to-end FW-vs-NG gap-vs-mwl experiment, sink-free.

Cleaner companion to e2e_gap_vs_length.py: rather than introducing
absorbing sinks (which decimate alive-walk counts and starve NG's
panel amortization), this script generates the ORIGINAL stratified
synthetic graph once and just sweeps `max_walk_len` from 10 to 100.

Walks on the sink-free graph all run to `mwl` (natural death is rare
on hub-circulating walks). So the realized avg_walk_len ≈ mwl, panel
amortization stays high throughout every walk, and the experiment
measures purely "does NG's per-step advantage grow with the number of
steps" without confounding from walk-decimation behavior.

For each mwl in the sweep:
  1. Invoke the ablation_streaming C++ binary with FW and NG, REPS each.
  2. Parse Throughput / Steps/sec / Final avg walk length from stdout.
  3. Compute gap = (NG_steps_per_sec - FW_steps_per_sec) / FW_steps_per_sec.

At the end, print a single DataFrame: mwl | avg_len | FW Msps | NG Msps
| gap %.

Path assumption: script lives in
    <root>/temporal-random-walk/synthetic_data_generator/
so binary is at ../build/bin/ablation_streaming relative to here.

Usage:
    source /home/ms2420/CLionProjects/tempest-benchmarks/.venv/bin/activate
    python e2e_gap_vs_mwl.py
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

# --- Strata identical to gen_synthetic_stratified.py (no sinks) ---
N_MEGA, E_MEGA, G_MEGA =    600,  25_000, 1500
N_WARM, E_WARM, G_WARM =  4_000,   2_500,  300
N_TAIL, E_TAIL, G_TAIL = 20_000,     100,   10
T_MAX = 1_000_000

# Original sink-free MIX (matches gen_synthetic_stratified.py).
MIX = {
    'mega': (0.85, 0.15, 0.00),
    'warm': (0.60, 0.40, 0.00),
    'tail': (0.55, 0.45, 0.00),
}

# Binary CLI fixed args.
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
    """Original sink-free stratified graph; matches gen_synthetic_stratified.py.
    Returns (sources, targets, timestamps), sorted by ts ascending."""
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


def main():
    here = Path(__file__).resolve().parent
    default_binary = here.parent / 'build' / 'bin' / 'ablation_streaming'

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mwls', nargs='+', type=int,
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--wpn', type=int, default=500,
                    help='Walks per node (500 matches bench_synthetic.py for '
                         'A40; drop to 50-100 for laptop)')
    ap.add_argument('--picker', default='exponential_index')
    ap.add_argument('--w-warp', type=int, default=4,
                    help='w_threshold_warp for NG (FW pinned to 1)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--binary', default=str(default_binary))
    ap.add_argument('--tmp-dir', default=None)
    ap.add_argument('--keep-csv', action='store_true')
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')

    print(f'# binary    : {binary}')
    print(f'# wpn       : {args.wpn}')
    print(f'# reps      : {args.reps}')
    print(f'# picker    : {args.picker}')
    print(f'# W (NG)    : {args.w_warp}    (FW pinned to 1)')
    print(f'# mwls      : {args.mwls}')
    print()

    if args.tmp_dir:
        tmp_root = Path(args.tmp_dir)
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix='e2e_synth_mwl_')
        tmp_root = Path(tmp_ctx.name)

    rows = []
    try:
        # Generate the (sink-free) graph ONCE — it's the same for every mwl.
        print('=== generating sink-free stratified graph (once) ===')
        t0 = time.perf_counter()
        src, dst, ts = generate_graph_arrays(args.seed)
        n_edges = len(src)
        print(f'  gen   : {n_edges:>10,} edges  '
              f'{time.perf_counter()-t0:5.1f}s')

        csv_path = tmp_root / 'synth_stratified.csv'
        t0 = time.perf_counter()
        write_csv(src, dst, ts, csv_path)
        print(f'  csv   : {csv_path.stat().st_size/1e6:6.1f} MB  '
              f'{time.perf_counter()-t0:5.1f}s\n')

        for mwl in args.mwls:
            if mwl < 2:
                print(f'  SKIP mwl={mwl} (< 2)')
                continue

            print(f'=== mwl={mwl} ===')
            per_variant = {}
            for variant in VARIANTS:
                w = args.w_warp if variant != 'full_walk' else 1
                wps_list, sps_list, len_list = [], [], []
                for r in range(args.reps):
                    t0 = time.perf_counter()
                    parsed, err = run_binary(
                        binary, csv_path, variant,
                        args.wpn, mwl, args.picker, w)
                    if parsed is None:
                        print(f'  {variant:<13} rep {r+1}: FAIL — {err}',
                              file=sys.stderr)
                        continue
                    wps, sps, avg_len = parsed
                    wps_list.append(wps)
                    sps_list.append(sps)
                    len_list.append(avg_len)
                    print(f'  {variant:<13} rep {r+1}/{args.reps}: '
                          f'steps/s={sps/1e6:6.2f}M  '
                          f'walks/s={wps/1e6:5.2f}M  '
                          f'avg_len={avg_len:6.2f}  '
                          f'({time.perf_counter()-t0:4.1f}s)',
                          flush=True)
                if not sps_list:
                    per_variant[variant] = None
                    continue
                wps_m, _ = mean_std(wps_list)
                sps_m, sps_s = mean_std(sps_list)
                len_m, _ = mean_std(len_list)
                per_variant[variant] = {
                    'sps_mean': sps_m, 'sps_std': sps_s,
                    'wps_mean': wps_m, 'avg_len': len_m,
                    'n_reps': len(sps_list),
                }

            fw = per_variant.get('full_walk')
            ng = per_variant.get('node_grouped')
            if fw is None or ng is None:
                print(f'  -- one variant failed; skipping gap row\n')
                continue
            gap_pct = (ng['sps_mean'] - fw['sps_mean']) / fw['sps_mean'] * 100
            print(f'  -- gap NG vs FW: {gap_pct:+6.2f}%   '
                  f'avg_len(FW)={fw["avg_len"]:.2f}\n')

            rows.append({
                'mwl':         mwl,
                'avg_len_fw':  fw['avg_len'],
                'avg_len_ng':  ng['avg_len'],
                'fw_msps':     fw['sps_mean'] / 1e6,
                'fw_msps_std': fw['sps_std']  / 1e6,
                'ng_msps':     ng['sps_mean'] / 1e6,
                'ng_msps_std': ng['sps_std']  / 1e6,
                'gap_pct':     gap_pct,
            })
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if not rows:
        print('No successful cells.', file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r['mwl'])

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print('=' * 84)
        print(' Gap percentage table (sorted by mwl asc)')
        print('=' * 84)
        with pd.option_context('display.float_format', '{:.3f}'.format,
                               'display.width', 160):
            print(df.to_string(index=False))
        print()
        gaps = df['gap_pct'].to_numpy()
    except ImportError:
        print('=' * 84)
        print(' Gap percentage (sorted by mwl asc)')
        print('=' * 84)
        print(f'{"mwl":>5} {"avg_len":>8} {"fw_msps":>9} {"ng_msps":>9} '
              f'{"gap_pct":>8}')
        for r in rows:
            print(f'{r["mwl"]:>5} {r["avg_len_fw"]:>8.2f} '
                  f'{r["fw_msps"]:>9.3f} {r["ng_msps"]:>9.3f} '
                  f'{r["gap_pct"]:>+7.2f}%')
        print()
        gaps = np.array([r['gap_pct'] for r in rows])

    diffs = np.diff(gaps)
    n_drops = int((diffs < 0).sum())
    if n_drops == 0:
        print(f' Monotonicity: STRICTLY INCREASING '
              f'({gaps[0]:+.2f}% → {gaps[-1]:+.2f}%)')
    else:
        print(f' Monotonicity: NOT MONOTONIC — {n_drops} drop(s); '
              f'range {min(gaps):+.2f}% → {max(gaps):+.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
