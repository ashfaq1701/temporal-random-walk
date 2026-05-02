"""Run test_alibaba_streaming for FW / NG / NG_global, 3 reps each, write CSV.

Single arg: path to alibaba data directory (containing data_0..89.parquet).
All other params hardcoded to the winning Backward config. C++ binary stdout
is suppressed; the script prints one summary line per run, a per-variant
roll-up after each variant's reps, and a final 3-way comparison.

    python alibaba_bench.py /path/to/alibaba_dir
"""
import csv
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

VARIANTS = ['full_walk', 'node_grouped', 'node_grouped_global_only']
REPS     = 3
OUTPUT   = 'alibaba_bench_results.csv'

# Positional CLI for test_alibaba_streaming (in order):
#   data_dir use_gpu picker klt wpn mins/step window_ms mwl total_min
#   timescale_bound w_thr_warp per_batch_csv walk_direction
COMMON = ['1', 'exponential_weight', None,  # placeholder for klt
          '20', '3', '900000', '100', '90', '-1', '4', None, 'backward']

THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)
WALK_TIME_RE  = re.compile(r'^Total walk time:\s+([\d.eE+-]+)',         re.MULTILINE)


def parse_summary(stdout):
    return (float(THROUGHPUT_RE.search(stdout).group(1)),
            float(STEPS_RE.search(stdout).group(1)),
            float(AVGLEN_RE.search(stdout).group(1)),
            float(WALK_TIME_RE.search(stdout).group(1)))


def mean_std(xs):
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


if len(sys.argv) != 2:
    sys.exit(f'Usage: {sys.argv[0]} <data_dir>')

data_dir = sys.argv[1]
binary = Path(__file__).resolve().parent / 'build' / 'bin' / 'test_alibaba_streaming'
if not binary.is_file():
    sys.exit(f'binary not found: {binary}\n'
             f'Build it: cmake --build build --target test_alibaba_streaming')

all_rows = []
results = {v: [] for v in VARIANTS}

with tempfile.TemporaryDirectory(prefix='alibaba_bench_') as scratch:
    for variant in VARIANTS:
        print(f'\n=== {variant} ===', flush=True)
        for rep in range(REPS):
            csv_path = f'{scratch}/{variant}_rep{rep}.csv'
            args = COMMON.copy()
            args[2] = variant
            args[-2] = csv_path
            argv = [str(binary), data_dir, *args]

            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                print(proc.stderr[-500:], file=sys.stderr)
                sys.exit(f'{variant} rep={rep+1} failed (exit {proc.returncode})')

            walks_ps, steps_ps, avg_len, walk_time = parse_summary(proc.stdout)
            results[variant].append((walks_ps, steps_ps, avg_len, walk_time))

            print(f'  rep {rep+1}/{REPS}  '
                  f'walks/s={walks_ps/1e6:6.3f}M  '
                  f'steps/s={steps_ps/1e6:6.3f}M  '
                  f'avg_len={avg_len:.4f}  '
                  f'walk_time={walk_time:7.3f}s',
                  flush=True)

            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    row['variant'] = variant
                    row['rep'] = rep
                    all_rows.append(row)

        rs = results[variant]
        wmu, wsd = mean_std([r[0] for r in rs])
        smu, ssd = mean_std([r[1] for r in rs])
        tmu, tsd = mean_std([r[3] for r in rs])
        print(f'  -- {variant} summary  '
              f'walks/s={wmu/1e6:6.3f}±{wsd/1e6:.3f}M  '
              f'steps/s={smu/1e6:6.3f}±{ssd/1e6:.3f}M  '
              f'walk_time={tmu:7.3f}±{tsd:.3f}s', flush=True)

with open(OUTPUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)
print(f'\nWrote {len(all_rows)} rows to {OUTPUT}')

# Final 3-way comparison table.
print()
print('=' * 80)
print(' Final comparison (positive % = variant faster than full_walk)')
print('=' * 80)
print(f' {"variant":<28} {"walks/s (M)":>15} {"steps/s (M)":>15} {"avg_len":>8} {"vs FW":>8}')
print('-' * 80)
fw_steps_mean = statistics.mean(r[1] for r in results['full_walk'])
for variant in VARIANTS:
    rs = results[variant]
    wmu, wsd = mean_std([r[0] for r in rs])
    smu, ssd = mean_std([r[1] for r in rs])
    amu, _   = mean_std([r[2] for r in rs])
    speedup = ((smu - fw_steps_mean) / fw_steps_mean * 100
               if variant != 'full_walk' else 0.0)
    speedup_str = '—' if variant == 'full_walk' else f'{speedup:+6.2f}%'
    print(f' {variant:<28} '
          f'{wmu/1e6:>9.3f}±{wsd/1e6:.3f} '
          f'{smu/1e6:>9.3f}±{ssd/1e6:.3f} '
          f'{amu:>8.4f} {speedup_str:>8}')
