"""End-to-end FW-vs-NG gap-vs-walk-length experiment, single script.

For each target avg walk length L in [10, 20, ..., 100]:
  1. Generate the stratified synthetic graph in-memory (numpy arrays) with
     p_sink = 1/(L-1) absorbing-sink fraction.
  2. Write to a temp CSV (the binary reads from a file path).
  3. Invoke the ablation_streaming C++ binary with FW and NG, REPS each.
  4. Parse Throughput / Steps/sec / Final avg walk length from stdout.
  5. Compute gap = (NG_steps_per_sec - FW_steps_per_sec) / FW_steps_per_sec.

At the end, print a single DataFrame: target | avg_len | FW Msps | NG Msps
| gap %, sorted by realized avg_len ascending.

Path assumption: the script lives in
    <root>/temporal-random-walk/synthetic_data_generator/

so the binary is at ../build/bin/ablation_streaming relative to here.
The companion tempest-benchmarks/ tree is a sibling under <root>/ — same
layout on laptop and server.

Usage:
    source /home/ms2420/CLionProjects/tempest-benchmarks/.venv/bin/activate
    python e2e_gap_vs_length.py
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

# --- Strata identical to gen_synthetic_stratified.py ---
N_MEGA, E_MEGA, G_MEGA =    600,  25_000, 1500
N_WARM, E_WARM, G_WARM =  4_000,   2_500,  300
N_TAIL, E_TAIL, G_TAIL = 20_000,     100,   10
N_SINK                 =  5_000   # absorbing nodes; zero outbound edges
T_MAX = 1_000_000

# Binary CLI fixed args.
USE_GPU      = '1'
IS_DIRECTED  = '1'
TIMESCALE    = '-1'
NUM_BATCHES  = '1'
NUM_WINDOWS  = '1'
BLOCK_DIM    = '256'

# Variants under test.
VARIANTS = ['full_walk', 'node_grouped']

# Throughput/steps/avg-len regex (binary stdout).
THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def stratum_ranges():
    s_mega = (0, N_MEGA)
    s_warm = (N_MEGA, N_MEGA + N_WARM)
    s_tail = (N_MEGA + N_WARM, N_MEGA + N_WARM + N_TAIL)
    s_sink = (N_MEGA + N_WARM + N_TAIL,
              N_MEGA + N_WARM + N_TAIL + N_SINK)
    return {'mega': s_mega, 'warm': s_warm, 'tail': s_tail, 'sink': s_sink}


def build_mix(p_sink):
    one_minus_p = 1.0 - p_sink
    return {
        'mega': (one_minus_p * 0.85, one_minus_p * 0.15, 0.0, p_sink),
        'warm': (one_minus_p * 0.60, one_minus_p * 0.40, 0.0, p_sink),
        'tail': (0.55, 0.45, 0.0, 0.0),
    }


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


def sample_targets_4(n_edges, mix, ranges, rng):
    p_mega, p_warm, p_tail, p_sink = mix
    strata = rng.choice(
        ['mega', 'warm', 'tail', 'sink'], size=n_edges,
        p=[p_mega, p_warm, p_tail, p_sink])
    out = np.empty(n_edges, dtype=np.int64)
    for s in ('mega', 'warm', 'tail', 'sink'):
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
    targets = sample_targets_4(n_edges, mix, ranges, rng)
    return sources, targets, timestamps


def generate_graph_arrays(p_sink: float, seed: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sources, targets, timestamps) numpy arrays sorted by ts asc."""
    rng = np.random.default_rng(seed)
    ranges = stratum_ranges()
    mix = build_mix(p_sink)

    parts = []
    for stratum, n, e, g in [('mega', N_MEGA, E_MEGA, G_MEGA),
                             ('warm', N_WARM, E_WARM, G_WARM),
                             ('tail', N_TAIL, E_TAIL, G_TAIL)]:
        lo = ranges[stratum][0]
        s, d, t = gen_stratum(n, e, g, mix[stratum], ranges, lo, rng)
        parts.append((s, d, t))

    sources    = np.concatenate([p[0] for p in parts])
    targets    = np.concatenate([p[1] for p in parts])
    timestamps = np.concatenate([p[2] for p in parts])

    order = np.argsort(timestamps, kind='stable')
    return sources[order], targets[order], timestamps[order]


def write_csv(sources, targets, timestamps, path: Path):
    """Write u,i,ts CSV the binary expects. Chunked for speed."""
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
    # ../build/bin/ablation_streaming relative to synthetic_data_generator/.
    default_binary = here.parent / 'build' / 'bin' / 'ablation_streaming'

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='+', type=int,
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument('--reps', type=int, default=3,
                    help='Reps per (target, variant)')
    ap.add_argument('--wpn', type=int, default=20)
    ap.add_argument('--mwl-mult', type=int, default=4,
                    help='mwl = mwl_mult * target  (4 fits 8 GB at target=100)')
    ap.add_argument('--picker', default='exponential_index')
    ap.add_argument('--w-warp', type=int, default=4,
                    help='w_threshold_warp for NG (FW pinned to 1)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--binary', default=str(default_binary))
    ap.add_argument('--tmp-dir', default=None,
                    help='Where to put temp CSVs (default: TemporaryDirectory)')
    ap.add_argument('--keep-csv', action='store_true',
                    help='Do not delete temp CSVs after each target')
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}\n'
                 f'(expected at <repo>/build/bin/ablation_streaming)')

    print(f'# binary    : {binary}')
    print(f'# wpn       : {args.wpn}    (mwl_mult: {args.mwl_mult})')
    print(f'# reps      : {args.reps}')
    print(f'# picker    : {args.picker}')
    print(f'# W (NG)    : {args.w_warp}    (FW pinned to 1)')
    print(f'# targets   : {args.targets}')
    print()

    # Long-lived tmp dir if --tmp-dir given, else use a TemporaryDirectory.
    if args.tmp_dir:
        tmp_root = Path(args.tmp_dir)
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix='e2e_synth_')
        tmp_root = Path(tmp_ctx.name)

    rows = []
    try:
        for target in args.targets:
            if target < 2:
                print(f'  SKIP target={target} (< 2)')
                continue
            p_sink = 1.0 / (target - 1)
            mwl = max(args.mwl_mult * target, 30)

            print(f'=== target={target}  p_sink={p_sink:.4f}  mwl={mwl} ===')

            t0 = time.perf_counter()
            src, dst, ts = generate_graph_arrays(p_sink, args.seed)
            sink_lo = N_MEGA + N_WARM + N_TAIL
            sink_frac = float((dst >= sink_lo).mean())
            n_edges = len(src)
            print(f'  gen   : {n_edges:>10,} edges  '
                  f'sink_frac={sink_frac*100:5.2f}%  '
                  f'{time.perf_counter()-t0:5.1f}s')

            csv_path = tmp_root / f'synth_len{target:03d}.csv'
            t0 = time.perf_counter()
            write_csv(src, dst, ts, csv_path)
            print(f'  csv   : {csv_path.stat().st_size/1e6:6.1f} MB  '
                  f'{time.perf_counter()-t0:5.1f}s')

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
                len_m, _    = mean_std(len_list)
                per_variant[variant] = {
                    'sps_mean':  sps_m,
                    'sps_std':   sps_s,
                    'wps_mean':  wps_m,
                    'avg_len':   len_m,
                    'n_reps':    len(sps_list),
                }

            if not args.keep_csv:
                csv_path.unlink(missing_ok=True)

            fw = per_variant.get('full_walk')
            ng = per_variant.get('node_grouped')
            if fw is None or ng is None:
                print(f'  -- one variant failed; skipping gap row\n')
                continue
            gap_pct = (ng['sps_mean'] - fw['sps_mean']) / fw['sps_mean'] * 100
            print(f'  -- gap NG vs FW: {gap_pct:+6.2f}%   '
                  f'avg_len(FW)={fw["avg_len"]:.2f}\n')

            rows.append({
                'target':      target,
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

    rows.sort(key=lambda r: r['avg_len_fw'])

    # Final dataframe-style table.
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print('=' * 84)
        print(' Gap percentage table (sorted by realized avg_len_fw asc)')
        print('=' * 84)
        with pd.option_context('display.float_format', '{:.3f}'.format,
                               'display.width', 160):
            print(df.to_string(index=False))
        print()
        gaps = df['gap_pct'].to_numpy()
    except ImportError:
        print('=' * 84)
        print(' Gap percentage (sorted by realized avg_len_fw asc)')
        print('=' * 84)
        print(f'{"target":>7} {"avg_len":>8} {"fw_msps":>9} {"ng_msps":>9} '
              f'{"gap_pct":>8}')
        for r in rows:
            print(f'{r["target"]:>7} {r["avg_len_fw"]:>8.2f} '
                  f'{r["fw_msps"]:>9.3f} {r["ng_msps"]:>9.3f} '
                  f'{r["gap_pct"]:>+7.2f}%')
        print()
        gaps = np.array([r['gap_pct'] for r in rows])

    # Monotonicity verdict.
    diffs = np.diff(gaps)
    n_drops = int((diffs < 0).sum())
    if n_drops == 0:
        verdict = (f' Monotonicity: STRICTLY INCREASING '
                   f'({gaps[0]:+.2f}% → {gaps[-1]:+.2f}%)')
    else:
        verdict = (f' Monotonicity: NOT MONOTONIC — {n_drops} drop(s); '
                   f'range {min(gaps):+.2f}% → {max(gaps):+.2f}%')
    print(verdict)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
