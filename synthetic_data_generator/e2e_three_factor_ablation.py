"""End-to-end three-factor ablation for the cooperative scheduler.

Tests whether NG's gain is the product of three orthogonal architectural
properties of the workload:

  Factor A — Walks-per-hub-per-step density (W). Determines tier dispatch
             (solo / warp / block); block tier amortizes ~10× more per
             walk than warp tier.
  Factor B — Per-hub timestamp-group count G. When G is in the smem-panel
             band, the panel preload saves DRAM probes per walk; when G
             exceeds the panel cap (~2800 indexed, ~1800 weighted) walks
             fall to the global tier with no smem preload.
  Factor C — Walk persistence (mwl). Fixed per-step scheduler costs need
             ~10 steps to amortize against per-step kernel work.

Five variants, each turning off exactly one factor (or all three):

  Full       — all on (baseline)
  LowW       — Factor A off (--wpn dropped → warp tier)
  HighG      — Factor B off (G > smem panel cap → global tier)
  ShortWalk  — Factor C off (--mwl=10 → no compounding)
  AllOff     — three stacked

The script generates two CSVs on the fly (G_default and G_high) into a
TemporaryDirectory, then runs FW and NG (only — not NG_global) on each
variant, --reps times each, with ±10% median-relative outlier rejection.
A final ranked table reports gap_pct per variant.

Usage (A40-scoped defaults: wpn=500 baseline / wpn=100 LowW; drop to
100/20 for laptop sanity):

    source ../../tempest-benchmarks/.venv/bin/activate
    python e2e_three_factor_ablation.py                    # A40 (default)
    python e2e_three_factor_ablation.py --wpn-full 100 --wpn-low 20   # laptop
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

# --- Strata sizes (identical to gen_synthetic_stratified.py). ---
N_MEGA, E_MEGA = 600, 25_000
N_WARM, E_WARM = 4_000, 2_500
N_TAIL, E_TAIL, G_TAIL = 20_000, 100, 10
T_MAX = 1_000_000

# Hub-skewed mixing (matches Hub-Synthetic baseline).
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

# Variants under test (only FW and NG; not NG_global).
EXEC_VARIANTS = ['full_walk', 'node_grouped']

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


def generate_graph(g_mega: int, g_warm: int, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (sources, targets, timestamps), sorted by ts ascending."""
    rng = np.random.default_rng(seed)
    ranges = stratum_ranges()
    parts = []
    for stratum, n, e, g in [('mega', N_MEGA, E_MEGA, g_mega),
                             ('warm', N_WARM, E_WARM, g_warm),
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


def run_binary(binary: Path, csv_path: Path, exec_variant: str,
               wpn: int, mwl: int, picker: str, w: int):
    cmd = [str(binary), str(csv_path),
           USE_GPU, picker, exec_variant, IS_DIRECTED,
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


def run_variant_cell(binary, csv_path, variant_name, wpn, mwl,
                     reps, picker, w_warp, threshold, min_keep):
    """Run FW and NG for one variant cell. Returns dict per exec_variant."""
    out = {}
    for exec_v in EXEC_VARIANTS:
        w = w_warp if exec_v != 'full_walk' else 1
        rep_records = []
        for r in range(reps):
            t0 = time.perf_counter()
            parsed, err = run_binary(binary, csv_path, exec_v,
                                     wpn, mwl, picker, w)
            if parsed is None:
                print(f'    {exec_v:<13} rep {r+1}: FAIL — {err}',
                      file=sys.stderr)
                continue
            wps, sps, avg_len = parsed
            rep_records.append({'sps': sps, 'wps': wps,
                                'avg_len': avg_len})
            print(f'    {exec_v:<13} rep {r+1}/{reps}: '
                  f'steps/s={sps/1e6:7.2f}M  '
                  f'walks/s={wps/1e6:5.2f}M  '
                  f'avg_len={avg_len:6.2f}  '
                  f'({time.perf_counter()-t0:4.1f}s)',
                  flush=True)
        if not rep_records:
            out[exec_v] = None
            continue
        kept, dropped = reject_outliers_by_key(
            rep_records, key='sps', threshold_frac=threshold,
            min_keep=min_keep)
        if dropped:
            ds = ', '.join(f'{d["sps"]/1e6:.2f}M' for d in dropped)
            print(f'    {exec_v:<13} dropped {len(dropped)} outlier(s): '
                  f'{ds}  (kept {len(kept)}/{len(rep_records)})')
        sps_m, sps_s = mean_std([r['sps']     for r in kept])
        wps_m, _     = mean_std([r['wps']     for r in kept])
        len_m, _     = mean_std([r['avg_len'] for r in kept])
        out[exec_v] = {
            'sps_mean': sps_m, 'sps_std': sps_s,
            'wps_mean': wps_m, 'avg_len': len_m,
            'n_reps_raw': len(rep_records),
            'n_reps_kept': len(kept),
        }
    return out


def main():
    here = Path(__file__).resolve().parent
    default_binary = here.parent / 'build' / 'bin' / 'ablation_streaming'

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default knob values: A40-scaled, matching bench_synthetic.py's
    # baseline cell. wpn-full=500 puts walks deeply in block tier on
    # the 24,600-node Hub-Synthetic graph (W ≈ 2670 per active hub at
    # baseline mixing, well past the BLOCK_DIM=256 saturation). wpn-low
    # is a 5× reduction (W ≈ 535) — still block tier on Hub-Synthetic
    # but reduces panel-amortization 5× relative to Full. For a sharper
    # Factor-A disabling that crosses into warp tier, override to
    # wpn-low ≤ 40 (W < 256). For laptop sanity, override:
    # --wpn-full 100 --wpn-low 20 (the laptop wpn=20 W ≈ 107 is
    # genuinely warp tier).
    ap.add_argument('--wpn-full',  type=int, default=500,
                    help='wpn for Full / HighG / ShortWalk (block tier; '
                         'A40 default = 500).')
    ap.add_argument('--wpn-low',   type=int, default=100,
                    help='wpn for LowW / AllOff. Default (100) is a 5× '
                         'reduction from wpn-full, still block tier on '
                         'Hub-Synthetic; drop to ≤ 40 for warp-tier '
                         'crossing.')
    ap.add_argument('--mwl-full',  type=int, default=50,
                    help='mwl for Full / LowW / HighG (compounding).')
    ap.add_argument('--mwl-short', type=int, default=10,
                    help='mwl for ShortWalk / AllOff (no compounding).')
    ap.add_argument('--g-mega-default', type=int, default=1500)
    ap.add_argument('--g-warm-default', type=int, default=300)
    ap.add_argument('--g-mega-high',    type=int, default=3000,
                    help='G for HighG / AllOff (above panel cap = 2800; '
                         'forces global tier).')
    ap.add_argument('--g-warm-high',    type=int, default=3000)

    ap.add_argument('--reps', type=int, default=5,
                    help='Reps per (variant, exec_variant)')
    ap.add_argument('--outlier-threshold', type=float, default=0.10)
    ap.add_argument('--min-keep', type=int, default=3)
    ap.add_argument('--picker', default='exponential_index')
    ap.add_argument('--w-warp', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--binary', default=str(default_binary))
    ap.add_argument('--tmp-dir', default=None)
    ap.add_argument('--keep-csv', action='store_true')
    ap.add_argument('--out-csv', default=None,
                    help='Where to write the long-form results CSV.')
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')

    # Variant table.  csv_key in {'default', 'high'} selects which CSV.
    VARIANTS = [
        # name        csv_key   wpn              mwl              factors-tag
        ('Full',      'default', args.wpn_full,  args.mwl_full,   'A=on  B=on  C=on'),
        ('LowW',      'default', args.wpn_low,   args.mwl_full,   'A=OFF B=on  C=on'),
        ('HighG',     'high',    args.wpn_full,  args.mwl_full,   'A=on  B=OFF C=on'),
        ('ShortWalk', 'default', args.wpn_full,  args.mwl_short,  'A=on  B=on  C=OFF'),
        ('AllOff',    'high',    args.wpn_low,   args.mwl_short,  'A=OFF B=OFF C=OFF'),
    ]

    print(f'# binary       : {binary}')
    print(f'# picker       : {args.picker}')
    print(f'# W (NG)       : {args.w_warp}    (FW pinned to 1)')
    print(f'# reps (raw)   : {args.reps}')
    print(f'# outlier band : ±{args.outlier_threshold*100:.0f}% of running '
          f'median; min-keep={args.min_keep}')
    print(f'# Variants:')
    for name, csv_key, wpn, mwl, tag in VARIANTS:
        print(f'    {name:<10} csv={csv_key:<7} wpn={wpn:<4} mwl={mwl:<4} '
              f'[{tag}]')
    print()

    if args.tmp_dir:
        tmp_root = Path(args.tmp_dir); tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix='e2e_3factor_')
        tmp_root = Path(tmp_ctx.name)

    rows = []
    try:
        # Generate the two CSVs on the fly.
        csv_paths = {}
        for csv_key, g_mega, g_warm in [
                ('default', args.g_mega_default, args.g_warm_default),
                ('high',    args.g_mega_high,    args.g_warm_high)]:
            print(f'=== generating CSV "{csv_key}" '
                  f'(G_mega={g_mega}, G_warm={g_warm}) ===')
            t0 = time.perf_counter()
            src, dst, ts = generate_graph(g_mega, g_warm, args.seed)
            print(f'  gen   : {len(src):>10,} edges  '
                  f'{time.perf_counter()-t0:5.1f}s')
            csv_path = tmp_root / f'synth_{csv_key}.csv'
            t0 = time.perf_counter()
            write_csv(src, dst, ts, csv_path)
            print(f'  csv   : {csv_path.stat().st_size/1e6:6.1f} MB  '
                  f'{time.perf_counter()-t0:5.1f}s')
            csv_paths[csv_key] = csv_path
        print()

        # Run each variant.
        for name, csv_key, wpn, mwl, tag in VARIANTS:
            csv_path = csv_paths[csv_key]
            print(f'=== {name}  [{tag}]  '
                  f'csv={csv_key}  wpn={wpn}  mwl={mwl} ===')
            cell = run_variant_cell(
                binary, csv_path, name, wpn, mwl,
                args.reps, args.picker, args.w_warp,
                args.outlier_threshold, args.min_keep)
            fw = cell.get('full_walk')
            ng = cell.get('node_grouped')
            if fw is None or ng is None:
                print(f'  -- one exec_variant failed; skipping cell\n')
                continue
            gap_pct = (ng['sps_mean'] - fw['sps_mean']) / fw['sps_mean'] * 100
            print(f'  -- gap NG vs FW: {gap_pct:+6.2f}%   '
                  f'avg_len(FW)={fw["avg_len"]:.2f}\n')
            rows.append({
                'variant':      name,
                'csv_key':      csv_key,
                'wpn':          wpn,
                'mwl':          mwl,
                'factors':      tag,
                'avg_len_fw':   fw['avg_len'],
                'avg_len_ng':   ng['avg_len'],
                'fw_msps':      fw['sps_mean'] / 1e6,
                'fw_msps_std':  fw['sps_std']  / 1e6,
                'ng_msps':      ng['sps_mean'] / 1e6,
                'ng_msps_std':  ng['sps_std']  / 1e6,
                'fw_kept':      f'{fw["n_reps_kept"]}/{fw["n_reps_raw"]}',
                'ng_kept':      f'{ng["n_reps_kept"]}/{ng["n_reps_raw"]}',
                'gap_pct':      gap_pct,
            })
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if not rows:
        print('No successful variants.', file=sys.stderr)
        return 1

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        # Variant order: Full, LowW, HighG, ShortWalk, AllOff (preserve).
        order = ['Full', 'LowW', 'HighG', 'ShortWalk', 'AllOff']
        df['_o'] = df['variant'].apply(lambda v: order.index(v))
        df = df.sort_values('_o').drop(columns=['_o']).reset_index(drop=True)
        print('=' * 110)
        print(' Three-factor ablation — long-form results')
        print('=' * 110)
        with pd.option_context('display.float_format', '{:.3f}'.format,
                               'display.width', 200):
            print(df.to_string(index=False))
        print()

        # Compact bar-chart-style summary.
        print('=' * 110)
        print(' Bar-chart summary  (gap_pct relative to Full baseline)')
        print('=' * 110)
        full_gap = float(df.loc[df['variant'] == 'Full', 'gap_pct'].iloc[0])
        for _, r in df.iterrows():
            label = f'{r["variant"]:<10} [{r["factors"]}]'
            gap = r['gap_pct']
            # Render a +/- 50pp bar, 1 char per pp.
            n_chars = int(round(abs(gap)))
            bar = ('+' if gap >= 0 else '-') * min(n_chars, 60)
            delta_from_full = gap - full_gap if r['variant'] != 'Full' else 0.0
            print(f' {label:<35}  {gap:+6.2f}%   '
                  f'(Δ vs Full: {delta_from_full:+6.2f}pp)  {bar}')
        print()

        # Hypothesis check.
        print('=' * 110)
        print(' Hypothesis check')
        print('=' * 110)
        full = full_gap
        low_w = float(df.loc[df['variant'] == 'LowW', 'gap_pct'].iloc[0])
        high_g = float(df.loc[df['variant'] == 'HighG', 'gap_pct'].iloc[0])
        short_walk = float(df.loc[df['variant'] == 'ShortWalk', 'gap_pct'].iloc[0])
        all_off = float(df.loc[df['variant'] == 'AllOff', 'gap_pct'].iloc[0])
        print(f'  Full  > LowW       (Factor A off): {full:+.2f}% > '
              f'{low_w:+.2f}%? {"YES" if full > low_w else "NO"}')
        print(f'  Full  > HighG      (Factor B off): {full:+.2f}% > '
              f'{high_g:+.2f}%? {"YES" if full > high_g else "NO"}')
        print(f'  Full  > ShortWalk  (Factor C off): {full:+.2f}% > '
              f'{short_walk:+.2f}%? {"YES" if full > short_walk else "NO"}')
        print(f'  Full  > AllOff     (all off)     : {full:+.2f}% > '
              f'{all_off:+.2f}%? {"YES" if full > all_off else "NO"}')
        print(f'  AllOff is smallest? '
              f'{"YES" if all_off == min(full, low_w, high_g, short_walk, all_off) else "NO"}')

        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f'\nWrote {args.out_csv}')
    except ImportError:
        print(' (pandas not available; printing long-form only)')
        for r in rows:
            print(f'  {r["variant"]:<10} gap={r["gap_pct"]:+6.2f}%  '
                  f'avg_len={r["avg_len_fw"]:.2f}  '
                  f'fw={r["fw_msps"]:.2f}M  ng={r["ng_msps"]:.2f}M')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
