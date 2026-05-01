"""A40 walk-sampling bench: FULL_WALK vs NODE_GROUPED vs NODE_GROUPED_GLOBAL_ONLY.

Driver around `build/bin/test_alibaba_streaming`. For each variant × rep:
    1. Invoke the C++ binary with --per-batch-csv pointed at a tmp file.
    2. Parse the binary's stdout for the run-level summary numbers.
    3. Read the per-batch CSV back and stamp (variant, rep) onto each row.
After all runs, merge every per-batch row into a single CSV at --output and
print a summary table (mean ± std walks/sec, steps/sec) per variant.

Walk config — closest match to the laptop sweep's "winning" cell, subject
to what the C++ binary exposes:
    walks_per_node     = 20
    max_walk_len       = 100
    walk_bias          = exponential_weight
    is_directed        = true (binary default; not a CLI knob)
    walk_direction     = Backward_In_Time (hardcoded in the binary)
    window_ms          = 900_000  (15-minute sliding window)
    minutes_per_step   = 3        (30 streaming iters across 90 minutes)
    total_minutes      = 90
    REPS               = 3 per variant
This matches the laptop sweep's P0 cell exactly (the binary was patched
to use Backward_In_Time in both the warmup and the main loop).

Place this script at the repo root next to `build/`. Resolves the
binary as <script_dir>/build/bin/test_alibaba_streaming.

Usage on the A40 server:
    python a40_winning_config_bench.py /path/to/alibaba_dir \\
        --output a40_results.csv

Uses only stdlib + pandas-free CSV reading, so no extra venv setup needed
beyond what the existing C++ build already requires.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


VARIANTS = ['full_walk', 'node_grouped', 'node_grouped_global_only']

# Regexes match run_ablation.py's parser exactly so future merging is easy.
THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def parse_run_summary(stdout):
    """Returns (walks_per_sec, steps_per_sec, avg_walk_length).

    Raises if any of the three lines is missing — the binary always prints
    them at the end of a successful run, so missing means the run failed
    or the binary's output format changed.
    """
    m_t = THROUGHPUT_RE.search(stdout)
    m_s = STEPS_RE.search(stdout)
    m_a = AVGLEN_RE.search(stdout)
    if not (m_t and m_s and m_a):
        raise RuntimeError(
            'missing one of Throughput/Steps/AvgLen in stdout tail:\n'
            + stdout[-500:])
    return float(m_t.group(1)), float(m_s.group(1)), float(m_a.group(1))


def build_argv(binary, dataset_dir, *, use_gpu, picker, klt,
               walks_per_node, minutes_per_step, window_ms, max_walk_len,
               total_minutes, timescale_bound, w_threshold_warp,
               per_batch_csv, walk_direction):
    """Positional args required by test_alibaba_streaming (see its main())."""
    return [
        str(binary),
        str(dataset_dir),
        '1' if use_gpu else '0',
        str(picker),
        str(klt),
        str(walks_per_node),
        str(minutes_per_step),
        str(window_ms),
        str(max_walk_len),
        str(total_minutes),
        str(timescale_bound),
        str(w_threshold_warp),
        str(per_batch_csv),
        str(walk_direction),
    ]


def run_one(binary, dataset_dir, args, klt, rep, scratch_dir):
    per_batch_csv = scratch_dir / f'per_batch_{klt}_rep{rep}.csv'
    argv = build_argv(
        binary, dataset_dir,
        use_gpu=args.use_gpu, picker=args.picker, klt=klt,
        walks_per_node=args.walks_per_node,
        minutes_per_step=args.minutes_per_step,
        window_ms=args.window_ms,
        max_walk_len=args.max_walk_len,
        total_minutes=args.total_minutes,
        timescale_bound=args.timescale_bound,
        w_threshold_warp=args.w_threshold_warp,
        per_batch_csv=per_batch_csv,
        walk_direction=args.walk_direction,
    )

    print(f'  > {" ".join(argv)}', flush=True)
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr[-500:], file=sys.stderr)
        raise RuntimeError(f'binary exited {proc.returncode} on klt={klt} rep={rep}')

    walks_ps, steps_ps, avg_len = parse_run_summary(proc.stdout)
    if not per_batch_csv.is_file():
        raise RuntimeError(f'binary did not write per-batch CSV: {per_batch_csv}')

    with per_batch_csv.open('r', newline='') as f:
        reader = csv.DictReader(f)
        per_batch_rows = list(reader)

    return {
        'walks_per_sec': walks_ps,
        'steps_per_sec': steps_ps,
        'avg_walk_length': avg_len,
        'per_batch_rows': per_batch_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_path',
                        help='Directory containing data_{0..89}.{parquet,csv}')
    parser.add_argument('--output', default='a40_winning_config_results.csv',
                        help='Merged CSV output path (default: %(default)s)')
    parser.add_argument('--binary',
                        default=str(Path(__file__).resolve().parent
                                    / 'build' / 'bin'
                                    / 'test_alibaba_streaming'),
                        help='Path to test_alibaba_streaming binary '
                             '(default: <script_dir>/build/bin/test_alibaba_streaming)')
    parser.add_argument('--reps', type=int, default=3,
                        help='Repetitions per variant (default: 3)')

    # Walk-config knobs — defaults pinned to the "winning" cell.
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--picker', default='exponential_weight')
    parser.add_argument('--walks_per_node', type=int, default=20)
    parser.add_argument('--minutes_per_step', type=int, default=3)
    parser.add_argument('--window_ms', type=int, default=900_000)
    parser.add_argument('--max_walk_len', type=int, default=100)
    parser.add_argument('--total_minutes', type=int, default=90)
    parser.add_argument('--timescale_bound', default='-1')
    parser.add_argument('--w_threshold_warp', type=int, default=4)
    parser.add_argument('--walk_direction', default='backward',
                        choices=['forward', 'backward'])
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        print(f'binary not found: {binary}', file=sys.stderr)
        print('Build it first:  cmake --build build --target test_alibaba_streaming',
              file=sys.stderr)
        return 1

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        print(f'data_path is not a directory: {data_path}', file=sys.stderr)
        return 1
    # Pre-flight: ensure data_{0..total-1} exists in either parquet or csv form.
    missing = []
    for i in range(args.total_minutes):
        if not ((data_path / f'data_{i}.parquet').is_file()
                or (data_path / f'data_{i}.csv').is_file()):
            missing.append(i)
    if missing:
        print(f'missing shards in {data_path}: data_{missing[:5]}{"..." if len(missing) > 5 else ""}',
              file=sys.stderr)
        return 1

    print('=' * 72)
    print(' A40 winning-config bench (C++ test_alibaba_streaming driver)')
    print(f'   binary:        {binary}')
    print(f'   data_path:     {data_path}')
    print(f'   total_min:     {args.total_minutes}  (step={args.minutes_per_step}m,'
          f' window={args.window_ms}ms)')
    print(f'   walk cfg:      picker={args.picker}'
          f' wpn={args.walks_per_node} mwl={args.max_walk_len}'
          f' w_thr_warp={args.w_threshold_warp}')
    print(f'   variants:      {", ".join(VARIANTS)}  (reps={args.reps} each)')
    print(f'   merged csv:    {args.output}')
    print('=' * 72, flush=True)

    # Run-level summary (one row per (variant, rep)) for the final stdout table.
    run_summary = []
    # Per-batch rows (one row per (variant, rep, step)) for the merged CSV.
    all_rows = []

    with tempfile.TemporaryDirectory(prefix='a40_per_batch_') as scratch:
        scratch_dir = Path(scratch)

        for variant in VARIANTS:
            for rep in range(args.reps):
                print(f'\n=== variant={variant}  rep={rep+1}/{args.reps} ===',
                      flush=True)
                result = run_one(binary, data_path, args, variant, rep, scratch_dir)

                run_summary.append({
                    'variant':         variant,
                    'rep':             rep,
                    'walks_per_sec':   result['walks_per_sec'],
                    'steps_per_sec':   result['steps_per_sec'],
                    'avg_walk_length': result['avg_walk_length'],
                })
                print(f'    walks/sec={result["walks_per_sec"]/1e6:.3f}M  '
                      f'steps/sec={result["steps_per_sec"]/1e6:.3f}M  '
                      f'avg_len={result["avg_walk_length"]:.2f}')

                for row in result['per_batch_rows']:
                    row['variant'] = variant
                    row['rep'] = rep
                    # step_idx 1 corresponds to the first measured step
                    # (the binary already discards a separate warmup batch
                    # internally), but keep is_warmup=True for it as a
                    # convention so downstream filters work uniformly.
                    row['is_warmup'] = (int(row['step_idx']) == 1)
                    all_rows.append(row)

    # Persist merged CSV.
    if not all_rows:
        print('No rows captured.', file=sys.stderr)
        return 1
    fieldnames = list(all_rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nWrote {len(all_rows)} per-batch rows to {args.output}')

    # Final summary table — mean ± std per variant over reps.
    print()
    print('=' * 72)
    print(' Summary (run-level metrics from binary stdout)')
    print('=' * 72)
    print(f' {"variant":<28} {"n":>3} {"walks/s mean(M)":>16} {"std(M)":>8} '
          f'{"steps/s mean(M)":>16} {"avg_len":>8}')
    print('-' * 84)
    fw_walks_mean = None
    for variant in VARIANTS:
        cells = [r for r in run_summary if r['variant'] == variant]
        if not cells:
            continue
        walks = [r['walks_per_sec'] for r in cells]
        steps = [r['steps_per_sec'] for r in cells]
        avgl = [r['avg_walk_length'] for r in cells]
        wmu = statistics.mean(walks)
        wsd = statistics.stdev(walks) if len(walks) > 1 else 0.0
        smu = statistics.mean(steps)
        amu = statistics.mean(avgl)
        if variant == 'full_walk':
            fw_walks_mean = wmu
        print(f' {variant:<28} {len(cells):>3d} '
              f'{wmu/1e6:>16.3f} {wsd/1e6:>8.3f} '
              f'{smu/1e6:>16.3f} {amu:>8.2f}')

    if fw_walks_mean is not None:
        print()
        print(' Speedup vs full_walk (positive = variant faster):')
        for variant in VARIANTS:
            if variant == 'full_walk':
                continue
            cells = [r for r in run_summary if r['variant'] == variant]
            if not cells:
                continue
            mu = statistics.mean(r['walks_per_sec'] for r in cells)
            gain = (mu - fw_walks_mean) / fw_walks_mean * 100.0
            print(f'   {variant:<28} {gain:+.2f} %')

    return 0


if __name__ == '__main__':
    sys.exit(main())
