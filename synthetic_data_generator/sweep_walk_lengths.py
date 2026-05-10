"""Generate synthetic stratified graphs at progressively longer target avg
walk lengths and verify the realized avg walk length matches.

Mechanism: an absorbing-sink stratum. For target_avg_len = L, route a
fraction p_sink = 1/(L-1) of every hub's outbound edges to sink nodes
that have ZERO outbound edges. In directed mode, walks die on the next
step after entering a sink. Walk length is then approximately:

    walk_len = 1 + Geometric(p_sink)
    E[walk_len] = 1 + 1/p_sink = L

Verification: for each target L, generate the graph, run ablation_streaming
with full_walk + exponential_index, and compare the binary's reported
"Final avg walk length" against the design L.

Usage (laptop-scoped defaults):
    source /home/ms2420/CLionProjects/tempest-benchmarks/.venv/bin/activate
    python sweep_walk_lengths.py

The binary, output-dir, wpn, and target-list are all argparse-overridable.
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Same strata as gen_synthetic_stratified.py (A40 tuning; runs on laptop too).
N_MEGA, E_MEGA, G_MEGA =    600,  25_000, 1500
N_WARM, E_WARM, G_WARM =  4_000,   2_500,  300
N_TAIL, E_TAIL, G_TAIL = 20_000,     100,   10
N_SINK                 =  5_000   # absorbing nodes; zero outbound edges
T_MAX = 1_000_000


def stratum_ranges():
    s_mega = (0, N_MEGA)
    s_warm = (N_MEGA, N_MEGA + N_WARM)
    s_tail = (N_MEGA + N_WARM, N_MEGA + N_WARM + N_TAIL)
    s_sink = (N_MEGA + N_WARM + N_TAIL,
              N_MEGA + N_WARM + N_TAIL + N_SINK)
    return {'mega': s_mega, 'warm': s_warm, 'tail': s_tail, 'sink': s_sink}


def gen_node_timestamps(n_nodes, g_per_node, rng):
    """Per-node distinct timestamps in [0, T_MAX]. Same logic as the
    original generator — oversample, unique, take first g."""
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
    """4-way stratum routing including sinks. mix is (p_mega, p_warm,
    p_tail, p_sink) summing to 1."""
    p_mega, p_warm, p_tail, p_sink = mix
    strata = rng.choice(
        ['mega', 'warm', 'tail', 'sink'],
        size=n_edges,
        p=[p_mega, p_warm, p_tail, p_sink],
    )
    out = np.empty(n_edges, dtype=np.int64)
    for s in ('mega', 'warm', 'tail', 'sink'):
        mask = strata == s
        if not mask.any():
            continue
        lo, hi = ranges[s]
        out[mask] = rng.integers(lo, hi, size=mask.sum())
    return out


def gen_stratum_edges(n_nodes, e_per_node, g_per_node, mix,
                      ranges, lo, rng):
    """Generate (sources, targets, timestamps) for one source stratum.
    Sinks generate no edges (zero outbound)."""
    n_edges = n_nodes * e_per_node
    sources = np.repeat(np.arange(lo, lo + n_nodes, dtype=np.int64),
                        e_per_node)
    node_ts = gen_node_timestamps(n_nodes, g_per_node, rng)
    pick_idx = rng.integers(0, g_per_node, size=(n_nodes, e_per_node))
    timestamps = np.take_along_axis(node_ts, pick_idx, axis=1).ravel()
    targets = sample_targets_4(n_edges, mix, ranges, rng)
    return sources, targets, timestamps


def build_mix(p_sink):
    """4-way mix: hubs route p_sink to sinks, rest split per the original
    hub-circulation pattern. Tails are entry-only (no sinks)."""
    one_minus_p = 1.0 - p_sink
    return {
        'mega': (one_minus_p * 0.85, one_minus_p * 0.15, 0.0, p_sink),
        'warm': (one_minus_p * 0.60, one_minus_p * 0.40, 0.0, p_sink),
        # tails seed walks; never route to sinks (would cut off all walks)
        'tail': (0.55, 0.45, 0.0, 0.0),
    }


def generate_graph(p_sink, output_path, seed):
    """Generate the synthetic stratified+sink graph and write it to
    output_path as `u,i,ts` CSV sorted by ts ascending. Returns
    (n_edges, n_sink_edges, max_node_id)."""
    rng = np.random.default_rng(seed)
    ranges = stratum_ranges()
    mix = build_mix(p_sink)

    parts = []
    for stratum, n, e, g in [('mega', N_MEGA, E_MEGA, G_MEGA),
                             ('warm', N_WARM, E_WARM, G_WARM),
                             ('tail', N_TAIL, E_TAIL, G_TAIL)]:
        lo = ranges[stratum][0]
        s, d, t = gen_stratum_edges(n, e, g, mix[stratum],
                                    ranges, lo, rng)
        parts.append((s, d, t))

    sources    = np.concatenate([p[0] for p in parts])
    targets    = np.concatenate([p[1] for p in parts])
    timestamps = np.concatenate([p[2] for p in parts])

    order = np.argsort(timestamps, kind='stable')
    sources    = sources[order]
    targets    = targets[order]
    timestamps = timestamps[order]

    sink_lo = ranges['sink'][0]
    n_sink_edges = int((targets >= sink_lo).sum())

    out_path = Path(output_path)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['u', 'i', 'ts'])
        chunk = 1_000_000
        for i in range(0, len(sources), chunk):
            rows = zip(sources[i:i+chunk].tolist(),
                       targets[i:i+chunk].tolist(),
                       timestamps[i:i+chunk].tolist())
            w.writerows(rows)

    max_node_id = int(max(sources.max(), targets.max()))
    return len(sources), n_sink_edges, max_node_id


def run_binary(binary, csv_path, mwl, wpn, picker='exponential_index',
               variant='full_walk', is_directed='1', w='1'):
    """Run ablation_streaming once. Returns measured avg walk length."""
    cmd = [str(binary), str(csv_path),
           '1', picker, variant, is_directed,
           str(wpn), '1', '1', str(mwl), '-1', '256', str(w)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, proc.stderr[-500:]
    m = re.search(r'Final avg walk length:\s*([\d.eE+-]+)', proc.stdout)
    if not m:
        return None, 'avg-len regex did not match stdout'
    return float(m.group(1)), None


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='+', type=int,
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    help='Target avg walk lengths to sweep')
    ap.add_argument('--output-dir', default='/tmp/synthetic_lengths',
                    help='Where to put generated CSVs and verification.csv')
    ap.add_argument('--binary',
                    default=str(here.parent / 'build' / 'bin'
                                / 'ablation_streaming'),
                    help='ablation_streaming binary')
    ap.add_argument('--wpn', type=int, default=20,
                    help='Walks per node (laptop-friendly default)')
    ap.add_argument('--mwl-mult', type=int, default=5,
                    help='mwl = mwl_mult × target (so geometric tails not '
                         'truncated)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'binary not found: {binary}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'# binary    : {binary}')
    print(f'# wpn       : {args.wpn}')
    print(f'# mwl_mult  : {args.mwl_mult}')
    print(f'# targets   : {args.targets}')
    print(f'# output_dir: {out_dir}')
    print()

    results = []
    for target in args.targets:
        if target < 2:
            print(f'skipping target={target} (must be >= 2)', file=sys.stderr)
            continue
        p_sink = 1.0 / (target - 1)
        expected = 1.0 + 1.0 / p_sink

        csv_path = out_dir / f'synth_len{target:03d}.csv'
        mwl = max(args.mwl_mult * target, 30)

        print(f'=== target={target}  p_sink={p_sink:.4f}  '
              f'expected_E[len]={expected:.2f}  mwl={mwl} ===')

        t0 = time.time()
        n_edges, n_sink_edges, max_id = generate_graph(
            p_sink, csv_path, args.seed)
        sink_frac = n_sink_edges / n_edges
        print(f'  gen   : {n_edges:>10,} edges  '
              f'sink_edges={n_sink_edges:>9,} ({sink_frac*100:5.2f}%)  '
              f'max_id={max_id:>5}  '
              f'{time.time()-t0:5.1f}s')

        t0 = time.time()
        measured, err = run_binary(binary, csv_path, mwl, args.wpn)
        if measured is None:
            print(f'  bench : FAIL — {err}', file=sys.stderr)
            continue
        err_abs = measured - expected
        err_pct = err_abs / expected * 100
        print(f'  bench : measured_avg_len={measured:6.2f}  '
              f'err={err_abs:+6.2f} ({err_pct:+5.1f}%)  '
              f'{time.time()-t0:5.1f}s')

        results.append({
            'target':       target,
            'p_sink':       p_sink,
            'expected':     expected,
            'measured':     measured,
            'err_abs':      err_abs,
            'err_pct':      err_pct,
            'n_edges':      n_edges,
            'n_sink_edges': n_sink_edges,
            'sink_frac':    sink_frac,
            'mwl':          mwl,
        })
        print()

    if not results:
        print('No successful runs.', file=sys.stderr)
        return 1

    print('=' * 78)
    print(' Verification table')
    print('=' * 78)
    print(f' {"target":>7}  {"p_sink":>8}  {"expected":>9}  '
          f'{"measured":>9}  {"err":>9}  {"err %":>7}  '
          f'{"sink_frac":>9}')
    print('-' * 78)
    for r in results:
        print(f' {r["target"]:>7}  {r["p_sink"]:>8.4f}  '
              f'{r["expected"]:>9.2f}  {r["measured"]:>9.2f}  '
              f'{r["err_abs"]:>+9.2f}  {r["err_pct"]:>+6.1f}%  '
              f'{r["sink_frac"]*100:>8.2f}%')

    out_csv = out_dir / 'verification.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print()
    print(f'Wrote {out_csv}')

    # Quick sanity check: warn if any |err_pct| > 15.
    bad = [r for r in results if abs(r['err_pct']) > 15.0]
    if bad:
        print()
        print(f'WARNING: {len(bad)} target(s) deviate >15% from design '
              f'expected. Possible causes: (1) mwl too small (geometric '
              f'tail truncated); (2) walks not starting from hubs (rare '
              f'tail-start short walks dominate average); (3) p_sink '
              f'formula off. Inspect verification.csv.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
