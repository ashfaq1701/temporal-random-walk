"""Run ablation_streaming on the synthetic stratified graph under nsys,
extract walk-side + ingest-side metrics + per-bucket kernel breakdown,
write three CSVs.

Each (variant, rep) is profiled once with `nsys profile --stats=true`.
The resulting .sqlite has both `walk_sampling_batch` and `ingestion_batch`
NVTX ranges; the kernels inside those ranges drive all three extractors:
  - walk-side    : kern ms/call, launches/call, per-kernel μs, gpu active frac
  - ingest-side  : total / sort / weight / h2d ms (variant-agnostic; pooled)
  - kernel split : launches + total_ms per bucket (solo, warp_smem,
                   block_smem, multi_block, etc.)

Walk config is locked to the cell that demonstrates NG > FW on this dataset
(see synthetic_data_generator/README.txt). Same defaults as bench_synthetic.py.

    python profile_synthetic.py [--csv synthetic_stratified.csv] [--reps 3]

C++ stdout is captured (not echoed). Per-rep one-liners + per-variant
summaries print. Final three tables to stdout. CSVs:
  synthetic_profiling_walks.csv      one row per variant
  synthetic_profiling_ingest.csv     one pooled row (variant-agnostic)
  synthetic_profiling_kernels.csv    one row per (variant, bucket)
"""
import argparse
import csv
import re
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

VARIANTS = ['full_walk', 'node_grouped', 'node_grouped_global_only']
REPS     = 3

# Walk config — matches bench_synthetic.py exactly.
WPN          = '500'
MWL          = '100'
NUM_BATCHES  = '1'
NUM_WINDOWS  = '1'
PICKER       = 'exponential_index'
IS_DIRECTED  = '1'
TIMESCALE    = '-1'
BLOCK_DIM    = '256'
W_THR_WARP   = '4'

# Bucket name → demangled-name substring. Order matters; more specific first.
# Mirrors alibaba_bench_profiling.py — same kernel names, just exercised
# under ablation_streaming instead of test_alibaba_streaming.
BUCKETS = [
    ('solo',         'node_grouped_solo_kernel'),
    ('warp_smem',    'node_grouped_warp_smem_kernel'),
    ('warp_global',  'node_grouped_warp_global_kernel'),
    ('block_smem',   'node_grouped_block_smem_kernel'),
    ('block_global', 'node_grouped_block_global_kernel'),
    ('multi_block',  'expand_block_tasks_kernel'),
    ('start_edges',  'pick_start_edges_kernel'),
    ('prepop_start', 'prepopulate_start_slot_kernel'),
    ('filter_alive', 'walk_alive_flags_kernel'),
    ('gather',       'gather_last_nodes_kernel'),
    ('partition_w',  'partition_by_w_kernel'),
    ('partition_g',  'partition_by_g_kernel'),
    ('full_walk',    'generate_random_walks_kernel'),
    ('reverse',      'reverse_walks_kernel'),
]
BUCKET_NAMES = [b for b, _ in BUCKETS] + ['other']

NVTX_TABLE_CANDIDATES = ('NVTX_EVENTS', 'NVTX_PUSHPOP_EVENTS',
                         'NSYS_EVENTS_NVTX_PUSHPOP')

THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def find_nvtx_table(db):
    have = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for c in NVTX_TABLE_CANDIDATES:
        if c in have:
            return c
    raise RuntimeError(f'no NVTX events table; have: {sorted(have)}')


def parse_throughput(stdout):
    return (float(THROUGHPUT_RE.search(stdout).group(1)),
            float(STEPS_RE.search(stdout).group(1)),
            float(AVGLEN_RE.search(stdout).group(1)))


def mean_std(xs):
    mu = statistics.mean(xs) if xs else 0.0
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def extract_walk(db, nvtx_tbl):
    ranges = db.execute(
        f"SELECT start, end FROM {nvtx_tbl} "
        f"WHERE text='walk_sampling_batch' ORDER BY start").fetchall()
    if not ranges:
        return None
    total_kern_ms = total_n = total_nvtx_ms = 0
    for s, e in ranges:
        n_kern, t_kern_ns = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(end - start), 0) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ?", (s, e)).fetchone()
        total_kern_ms += t_kern_ns / 1e6
        total_n       += n_kern
        total_nvtx_ms += (e - s) / 1e6
    n_calls = len(ranges)
    return {
        'n_calls':                n_calls,
        'kern_time_per_call_ms':  total_kern_ms / n_calls,
        'kern_launches_per_call': total_n / n_calls,
        'per_kernel_us':          (total_kern_ms / max(1, total_n)) * 1000.0,
        'gpu_active_frac':        total_kern_ms / total_nvtx_ms if total_nvtx_ms else 0.0,
    }


def extract_ingest(db, nvtx_tbl):
    ranges = db.execute(
        f"SELECT start, end FROM {nvtx_tbl} "
        f"WHERE text='ingestion_batch' ORDER BY start").fetchall()
    if not ranges:
        return None
    per_call = []
    for s, e in ranges:
        nvtx_ms = (e - s) / 1e6
        kerns = db.execute("""
            SELECT k.start, k.end,
                   COALESCE(d.value, m.value, sn.value, '') AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds d  ON d.id  = k.demangledName
            LEFT JOIN StringIds m  ON m.id  = k.mangledName
            LEFT JOIN StringIds sn ON sn.id = k.shortName
            WHERE k.start >= ? AND k.end <= ?
        """, (s, e)).fetchall()
        sort_ms = sum((k[1] - k[0]) for k in kerns
                      if 'RadixSort' in k[2] or 'merge_kernel' in k[2]) / 1e6
        weight_ms = sum((k[1] - k[0]) for k in kerns
                        if 'compute_per_node_weights' in k[2]) / 1e6
        h2d_ms = (db.execute(
            "SELECT COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start >= ? AND end <= ? AND copyKind = 1", (s, e)
        ).fetchone()[0]) / 1e6
        per_call.append({'total': nvtx_ms, 'sort': sort_ms,
                         'weight': weight_ms, 'h2d': h2d_ms,
                         'launches': len(kerns)})
    n = len(per_call)
    return {f'ingest_{k}_ms' if k != 'launches' else 'ingest_launches':
            sum(c[k] for c in per_call) / n
            for k in ('total', 'sort', 'weight', 'h2d', 'launches')}


def extract_kernel_breakdown(db, nvtx_tbl):
    ranges = db.execute(
        f"SELECT start, end FROM {nvtx_tbl} "
        f"WHERE text='walk_sampling_batch' ORDER BY start").fetchall()
    if not ranges:
        return None
    where = ' OR '.join(f'(k.start>={s} AND k.end<={e})' for s, e in ranges)
    rows = db.execute(f"""
        SELECT COALESCE(d.value, m.value, sn.value, '?') AS name,
               COUNT(*) AS n,
               COALESCE(SUM(k.end - k.start), 0) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds d  ON d.id  = k.demangledName
        LEFT JOIN StringIds m  ON m.id  = k.mangledName
        LEFT JOIN StringIds sn ON sn.id = k.shortName
        WHERE {where}
        GROUP BY name
    """).fetchall()
    counts = {b: {'n': 0, 'ms': 0.0} for b in BUCKET_NAMES}
    for name, n, total_ns in rows:
        bucket = 'other'
        for b_name, b_pat in BUCKETS:
            if b_pat in name:
                bucket = b_name
                break
        counts[bucket]['n']  += n
        counts[bucket]['ms'] += total_ns / 1e6
    return counts


def invoke_nsys(nsys_bin, rep_path, binary, csv_path, variant):
    # Positional CLI for ablation_streaming (in order):
    #   file use_gpu picker klt is_directed wpn nb nw mwl tsb block_dim wtw
    cmd = [nsys_bin, 'profile', '--trace=cuda,nvtx', '--stats=true',
           '--force-overwrite=true',
           '--output', str(rep_path.with_suffix('')),
           str(binary), str(csv_path),
           '1', PICKER, variant, IS_DIRECTED,
           WPN, NUM_BATCHES, NUM_WINDOWS, MWL, TIMESCALE,
           BLOCK_DIM, W_THR_WARP]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'nsys profile exit {proc.returncode}\n'
                           f'stderr tail:\n{proc.stderr[-500:]}')
    return parse_throughput(proc.stdout)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',     default=str(here / 'synthetic_stratified.csv'))
    ap.add_argument('--reps',    type=int, default=REPS)
    ap.add_argument('--binary',  default=str(here.parent / 'build' / 'bin'
                                             / 'ablation_streaming'))
    ap.add_argument('--nsys',    default='nsys')
    ap.add_argument('--rep-dir', default=str(here / 'synthetic_profiling_nsys'))
    ap.add_argument('--out-prefix', default=str(here / 'synthetic_profiling'))
    args = ap.parse_args()

    binary   = Path(args.binary)
    csv_path = Path(args.csv)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')
    if not csv_path.is_file():
        sys.exit(f'csv not found: {csv_path}')
    rep_dir = Path(args.rep_dir); rep_dir.mkdir(parents=True, exist_ok=True)

    print(f'# binary     : {binary}')
    print(f'# nsys       : {args.nsys}')
    print(f'# csv        : {csv_path}')
    print(f'# config     : wpn={WPN} mwl={MWL} nb={NUM_BATCHES} nw={NUM_WINDOWS}'
          f' picker={PICKER}')
    print(f'# reps       : {args.reps}')
    print(f'# rep dir    : {rep_dir}')
    print(f'# out prefix : {args.out_prefix}')

    walk_runs   = {v: [] for v in VARIANTS}
    bucket_runs = {v: [] for v in VARIANTS}
    throughput  = {v: [] for v in VARIANTS}
    ingest_pool = []

    for variant in VARIANTS:
        print(f'\n=== {variant} ===', flush=True)
        for rep in range(args.reps):
            rep_path = rep_dir / f'{variant}_run{rep+1}.nsys-rep'
            print(f'  rep {rep+1}/{args.reps}: profiling ...', end=' ', flush=True)
            try:
                wps, sps, avg_len = invoke_nsys(
                    args.nsys, rep_path, binary, csv_path, variant)
            except RuntimeError as e:
                print(f'FAIL: {e}', file=sys.stderr); continue
            throughput[variant].append((wps, sps, avg_len))

            sql_path = rep_path.with_suffix('.sqlite')
            if not sql_path.exists():
                print(f'sqlite missing: {sql_path}', file=sys.stderr); continue

            db = sqlite3.connect(sql_path)
            try:
                nvtx_tbl = find_nvtx_table(db)
                w  = extract_walk(db, nvtx_tbl)
                ig = extract_ingest(db, nvtx_tbl)
                kb = extract_kernel_breakdown(db, nvtx_tbl)
            finally:
                db.close()

            if w:  walk_runs[variant].append(w)
            if ig: ingest_pool.append({'variant': variant, 'rep': rep, **ig})
            if kb: bucket_runs[variant].append(kb)

            if w:
                print(f'kern_t={w["kern_time_per_call_ms"]:6.2f}ms  '
                      f'launches={w["kern_launches_per_call"]:7.1f}  '
                      f'kern_us={w["per_kernel_us"]:7.2f}  '
                      f'active={w["gpu_active_frac"]:.3f}  '
                      f'walks/s={wps/1e6:.2f}M', flush=True)
            else:
                print('no walk-side NVTX in sqlite', flush=True)

        ws = walk_runs[variant]
        if ws:
            mt, st = mean_std([r['kern_time_per_call_ms']  for r in ws])
            ml, sl = mean_std([r['kern_launches_per_call'] for r in ws])
            mu, su = mean_std([r['per_kernel_us']          for r in ws])
            ma, sa = mean_std([r['gpu_active_frac']        for r in ws])
            print(f'  -- {variant} summary  '
                  f'kern_t={mt:6.2f}±{st:.2f}ms  '
                  f'launches={ml:7.1f}±{sl:.1f}  '
                  f'kern_us={mu:7.2f}±{su:.2f}  '
                  f'active={ma:.3f}±{sa:.3f}', flush=True)

    out_walks   = f'{args.out_prefix}_walks.csv'
    out_ingest  = f'{args.out_prefix}_ingest.csv'
    out_kernels = f'{args.out_prefix}_kernels.csv'

    walk_rows = []
    for variant in VARIANTS:
        ws = walk_runs[variant]
        if not ws: continue
        thr = throughput[variant]
        kt_m, kt_s = mean_std([r['kern_time_per_call_ms']  for r in ws])
        kn_m, kn_s = mean_std([r['kern_launches_per_call'] for r in ws])
        pk_m, pk_s = mean_std([r['per_kernel_us']          for r in ws])
        gf_m, gf_s = mean_std([r['gpu_active_frac']        for r in ws])
        walk_rows.append({
            'variant':                     variant,
            'n_reps':                      len(ws),
            'walks_per_sec_mean':          statistics.mean(t[0] for t in thr),
            'steps_per_sec_mean':          statistics.mean(t[1] for t in thr),
            'avg_walk_length_mean':        statistics.mean(t[2] for t in thr),
            'kern_time_per_call_ms_mean':  kt_m, 'kern_time_per_call_ms_std':  kt_s,
            'kern_launches_per_call_mean': kn_m, 'kern_launches_per_call_std': kn_s,
            'per_kernel_us_mean':          pk_m, 'per_kernel_us_std':          pk_s,
            'gpu_active_frac_mean':        gf_m, 'gpu_active_frac_std':        gf_s,
        })
    if walk_rows:
        with open(out_walks, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(walk_rows[0].keys()))
            w.writeheader(); w.writerows(walk_rows)
        print(f'\nWrote {len(walk_rows)} walk rows to {out_walks}')

    ingest_row = None
    if ingest_pool:
        keys = ('ingest_total_ms', 'ingest_sort_ms', 'ingest_weight_ms',
                'ingest_h2d_ms', 'ingest_launches')
        ingest_row = {'dataset':    'synthetic_stratified',
                      'n_samples':  len(ingest_pool),
                      'n_variants': len({r['variant'] for r in ingest_pool})}
        for k in keys:
            m, s = mean_std([r[k] for r in ingest_pool])
            ingest_row[f'{k}_mean'] = m
            ingest_row[f'{k}_std']  = s
        tot = ingest_row['ingest_total_ms_mean']
        if tot:
            for k in ('sort', 'weight', 'h2d'):
                ingest_row[f'ingest_{k}_frac'] = \
                    ingest_row[f'ingest_{k}_ms_mean'] / tot
        with open(out_ingest, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(ingest_row.keys()))
            w.writeheader(); w.writerow(ingest_row)
        print(f'Wrote ingest row to {out_ingest}')

    kb_rows = []
    for variant in VARIANTS:
        bs = bucket_runs[variant]
        if not bs: continue
        for bucket in BUCKET_NAMES:
            mn, sn = mean_std([r[bucket]['n']  for r in bs])
            mt, st = mean_std([r[bucket]['ms'] for r in bs])
            kb_rows.append({
                'variant':       variant,
                'bucket':        bucket,
                'n_reps':        len(bs),
                'launches_mean': mn, 'launches_std': sn,
                'total_ms_mean': mt, 'total_ms_std': st,
            })
    if kb_rows:
        with open(out_kernels, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(kb_rows[0].keys()))
            w.writeheader(); w.writerows(kb_rows)
        print(f'Wrote {len(kb_rows)} kernel-bucket rows to {out_kernels}')

    print()
    print('=' * 84)
    print(' Walk-side metrics (mean ± std across reps)')
    print('=' * 84)
    print(f' {"variant":<28} {"kern ms/call":>13} {"launches/call":>14} '
          f'{"kern μs":>11} {"gpu_active":>11}')
    print('-' * 84)
    for r in walk_rows:
        print(f' {r["variant"]:<28} '
              f'{r["kern_time_per_call_ms_mean"]:>7.2f}±{r["kern_time_per_call_ms_std"]:.2f} '
              f'{r["kern_launches_per_call_mean"]:>8.1f}±{r["kern_launches_per_call_std"]:.1f} '
              f'{r["per_kernel_us_mean"]:>7.2f}±{r["per_kernel_us_std"]:.2f} '
              f'{r["gpu_active_frac_mean"]:>6.3f}±{r["gpu_active_frac_std"]:.3f}')

    if ingest_row is not None:
        print()
        print('=' * 84)
        print(' Ingest-side metrics (variant-agnostic; pooled across all reps)')
        print('=' * 84)
        print(f' n_samples={ingest_row["n_samples"]}  '
              f'total_ms={ingest_row["ingest_total_ms_mean"]:.2f}±'
              f'{ingest_row["ingest_total_ms_std"]:.2f}  '
              f'sort={ingest_row["ingest_sort_ms_mean"]:.2f}ms ('
              f'{ingest_row["ingest_sort_frac"]*100:.1f}%)  '
              f'weight={ingest_row["ingest_weight_ms_mean"]:.2f}ms ('
              f'{ingest_row["ingest_weight_frac"]*100:.1f}%)  '
              f'h2d={ingest_row["ingest_h2d_ms_mean"]:.2f}ms ('
              f'{ingest_row["ingest_h2d_frac"]*100:.1f}%)  '
              f'launches={ingest_row["ingest_launches_mean"]:.1f}')

    if kb_rows:
        print()
        print('=' * 84)
        print(' Kernel breakdown — launches × total ms inside walk_sampling_batch')
        print('=' * 84)
        head = f'{"bucket":<14}'
        sub  = f'{"":<14}'
        active_variants = [v for v in VARIANTS if any(r['variant'] == v for r in kb_rows)]
        for v in active_variants:
            head += f' | {v:^21}'
            sub  += f' | {"launches":>10} {"total ms":>10}'
        print(head); print(sub); print('-' * len(head))
        for bucket in BUCKET_NAMES:
            line = f'{bucket:<14}'
            for v in active_variants:
                cell = next((r for r in kb_rows
                             if r['variant'] == v and r['bucket'] == bucket), None)
                if cell is None or cell['launches_mean'] == 0:
                    line += f' | {"0":>10} {"—":>10}'
                else:
                    line += f' | {cell["launches_mean"]:>10.1f} {cell["total_ms_mean"]:>10.2f}'
            print(line)

    return 0


if __name__ == '__main__':
    sys.exit(main())
