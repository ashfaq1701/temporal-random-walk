"""For each pre-generated synth_len{NNN}.csv, run full_walk / node_grouped /
node_grouped_global_only with REPS repetitions, take the mean steps/sec
per variant, and compute coop-vs-FW gap %. Then check whether the gap
grows monotonically with realized avg walk length.

Assumes sweep_walk_lengths.py has already produced the CSVs (default
location /tmp/synthetic_lengths/synth_len{010..100}.csv).

Usage:
    source /home/ms2420/CLionProjects/tempest-benchmarks/.venv/bin/activate
    python sweep_gap_vs_length.py
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ['full_walk', 'node_grouped', 'node_grouped_global_only']

THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def parse_summary(stdout):
    m_t = THROUGHPUT_RE.search(stdout)
    m_s = STEPS_RE.search(stdout)
    m_a = AVGLEN_RE.search(stdout)
    if not (m_t and m_s and m_a):
        return None
    return float(m_t.group(1)), float(m_s.group(1)), float(m_a.group(1))


def run_once(binary, csv_path, variant, mwl, wpn, picker, w):
    cmd = [str(binary), str(csv_path),
           '1', picker, variant, '1',
           str(wpn), '1', '1', str(mwl), '-1', '256', str(w)]
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='+', type=int,
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument('--input-dir', default='/tmp/synthetic_lengths')
    ap.add_argument('--binary',
                    default=str(here.parent / 'build' / 'bin'
                                / 'ablation_streaming'))
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--wpn', type=int, default=20)
    ap.add_argument('--mwl-mult', type=int, default=5)
    ap.add_argument('--picker', default='exponential_index')
    ap.add_argument('--w-warp', type=int, default=4,
                    help='w_threshold_warp for NG variants (FW pinned to 1)')
    ap.add_argument('--out-csv', default=None,
                    help='where to write the per-cell results CSV '
                         '(default: <input-dir>/gap_vs_length.csv)')
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')
    in_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv) if args.out_csv else in_dir / 'gap_vs_length.csv'

    print(f'# binary    : {binary}')
    print(f'# input_dir : {in_dir}')
    print(f'# wpn       : {args.wpn}    (mwl_mult: {args.mwl_mult})')
    print(f'# reps      : {args.reps}')
    print(f'# picker    : {args.picker}')
    print(f'# W (NG)    : {args.w_warp}    (FW pinned to 1)')
    print()

    rows = []  # one row per (target, variant)
    for target in args.targets:
        csv_path = in_dir / f'synth_len{target:03d}.csv'
        if not csv_path.is_file():
            print(f'  SKIP target={target}: {csv_path} not found',
                  file=sys.stderr)
            continue
        mwl = max(args.mwl_mult * target, 30)

        print(f'=== target={target}  mwl={mwl}  csv={csv_path.name} ===')
        per_variant = {}
        for variant in VARIANTS:
            w = args.w_warp if variant != 'full_walk' else 1
            sps_list, wps_list, len_list = [], [], []
            for r in range(args.reps):
                t0 = time.time()
                parsed, err = run_once(binary, csv_path, variant,
                                       mwl, args.wpn, args.picker, w)
                if parsed is None:
                    print(f'  {variant:<26} rep {r+1}: FAIL — {err}',
                          file=sys.stderr)
                    continue
                wps, sps, avg_len = parsed
                sps_list.append(sps)
                wps_list.append(wps)
                len_list.append(avg_len)
                print(f'  {variant:<26} rep {r+1}/{args.reps}: '
                      f'steps/s={sps/1e6:6.2f}M  walks/s={wps/1e6:5.2f}M  '
                      f'avg_len={avg_len:6.2f}  ({time.time()-t0:4.1f}s)',
                      flush=True)
            per_variant[variant] = {
                'sps_mean': mean_std(sps_list)[0],
                'sps_std':  mean_std(sps_list)[1],
                'wps_mean': mean_std(wps_list)[0],
                'len_mean': mean_std(len_list)[0],
                'n_reps':   len(sps_list),
            }

        if 'full_walk' not in per_variant or per_variant['full_walk']['n_reps'] == 0:
            print(f'  full_walk failed — skipping target={target}',
                  file=sys.stderr)
            continue

        fw_sps = per_variant['full_walk']['sps_mean']
        ng_sps  = per_variant.get('node_grouped',             {}).get('sps_mean', 0.0)
        ngg_sps = per_variant.get('node_grouped_global_only', {}).get('sps_mean', 0.0)
        gap_ng  = (ng_sps  - fw_sps) / fw_sps * 100 if fw_sps else 0.0
        gap_ngg = (ngg_sps - fw_sps) / fw_sps * 100 if fw_sps else 0.0
        avg_len = per_variant['full_walk']['len_mean']

        print(f'  -- gap NG  vs FW : {gap_ng:+6.2f}%   '
              f'NGG vs FW: {gap_ngg:+6.2f}%   '
              f'avg_len(FW)={avg_len:.2f}\n')

        rows.append({
            'target':       target,
            'avg_len_fw':   avg_len,
            'fw_sps_mean':  fw_sps,
            'fw_sps_std':   per_variant['full_walk']['sps_std'],
            'ng_sps_mean':  ng_sps,
            'ng_sps_std':   per_variant.get('node_grouped', {}).get('sps_std', 0.0),
            'ngg_sps_mean': ngg_sps,
            'ngg_sps_std':  per_variant.get('node_grouped_global_only', {}).get('sps_std', 0.0),
            'gap_ng_pct':   gap_ng,
            'gap_ngg_pct':  gap_ngg,
        })

    if not rows:
        print('No successful cells.', file=sys.stderr)
        return 1

    # Sort by measured avg_len_fw (the right x-axis for the plot).
    rows.sort(key=lambda r: r['avg_len_fw'])

    print('=' * 96)
    print(' Gap vs realized avg walk length (sorted ascending)')
    print('=' * 96)
    print(f' {"target":>6}  {"avg_len":>8}  {"FW Msps":>8}  '
          f'{"NG Msps":>8}  {"NGG Msps":>9}  '
          f'{"gap NG":>8}  {"gap NGG":>9}')
    print('-' * 96)
    for r in rows:
        print(f' {r["target"]:>6}  {r["avg_len_fw"]:>8.2f}  '
              f'{r["fw_sps_mean"]/1e6:>8.2f}  '
              f'{r["ng_sps_mean"]/1e6:>8.2f}  '
              f'{r["ngg_sps_mean"]/1e6:>9.2f}  '
              f'{r["gap_ng_pct"]:>+7.2f}%  '
              f'{r["gap_ngg_pct"]:>+8.2f}%')

    # Monotonicity check.
    print()
    print('=' * 60)
    print(' Monotonicity check (gap %, ascending in avg_len)')
    print('=' * 60)
    for label, key in [('NG vs FW', 'gap_ng_pct'),
                       ('NGG vs FW', 'gap_ngg_pct')]:
        seq = [r[key] for r in rows]
        violations = []
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1]:
                violations.append((i, seq[i-1], seq[i]))
        if not violations:
            verdict = f'STRICTLY MONOTONIC ({seq[0]:+.2f}% → {seq[-1]:+.2f}%)'
        else:
            verdict = (f'NOT MONOTONIC — {len(violations)} violation(s)'
                       f', range {min(seq):+.2f}% → {max(seq):+.2f}%')
        print(f'  {label:<11}: {verdict}')
        for i, prev, cur in violations:
            avg_prev = rows[i-1]['avg_len_fw']
            avg_cur  = rows[i]['avg_len_fw']
            print(f'    drop at avg_len {avg_prev:.2f} → {avg_cur:.2f}  '
                  f'({prev:+.2f}% → {cur:+.2f}%)')

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f'Wrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
