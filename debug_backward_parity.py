"""Per-step death-rate diagnostic for Backward parity bug.

Streams a small Alibaba slice, then for each kernel variant calls
get_random_walks_and_times_for_last_batch(...), captures walk_lens, and
computes:
  - count of walks that reached length L for L in [0, max_walk_len]
  - per-step death rate = (# walks with walk_len == k) / (# walks with walk_len >= k)

Compares FW vs NG vs NG_global. For Backward direction, FW and NG should
agree on within-noise mean walk length; if they diverge, locate the step
where the death-rate curves separate.

Args mirror the C++ binary's winning config: ExpWeight, wpn=20, mwl=100,
window=900s, mins/step=3, total=12 (just enough for 4 streaming iters,
then one walk call per variant on the same final graph state).
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from temporal_random_walk import TemporalRandomWalk


VARIANTS = ['FULL_WALK', 'NODE_GROUPED', 'NODE_GROUPED_GLOBAL_ONLY']


def stream_one(trw, base_dir, start_minute, minutes_per_step):
    dfs = [pd.read_parquet(os.path.join(base_dir, f'data_{start_minute+j}.parquet'))
           for j in range(minutes_per_step)]
    merged = pd.concat(dfs, ignore_index=True)
    shuffled = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    trw.add_multiple_edges(
        shuffled['u'].astype(np.int32).values,
        shuffled['i'].astype(np.int32).values,
        shuffled['ts'].astype(np.int64).values,
    )


def walk_for_last_batch(trw, klt, walks_per_node, mwl, walk_bias, walk_dir):
    return trw.get_random_walks_and_times_for_last_batch(
        max_walk_len=mwl,
        walk_bias=walk_bias,
        num_walks_per_node=walks_per_node,
        initial_edge_bias='Uniform',
        walk_direction=walk_dir,
        kernel_launch_type=klt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('--total_minutes', type=int, default=12)
    ap.add_argument('--minutes_per_step', type=int, default=3)
    ap.add_argument('--window_ms', type=int, default=900_000)
    ap.add_argument('--walks_per_node', type=int, default=20)
    ap.add_argument('--max_walk_len', type=int, default=100)
    ap.add_argument('--walk_bias', type=str, default='ExponentialWeight')
    ap.add_argument('--walk_direction', type=str, default='Backward_In_Time')
    args = ap.parse_args()

    trw = TemporalRandomWalk(
        is_directed=True, use_gpu=True,
        max_time_capacity=args.window_ms,
    )

    n_iters = args.total_minutes // args.minutes_per_step
    # Match the C++ binary: ingest one minute-batch, walk per-variant, repeat.
    # The "last batch" is whatever was just ingested, so each step gives a
    # different (but per-iter-shared-across-variants) walk problem.
    results = {v: [] for v in VARIANTS}  # per-iter walk_lens arrays
    for it in range(n_iters):
        stream_one(trw, args.data_path, it * args.minutes_per_step,
                   args.minutes_per_step)
        active = trw.get_edge_count()
        print(f'iter {it+1}/{n_iters} | min={ (it+1)*args.minutes_per_step} '
              f'| active={active/1e6:.1f}M', flush=True)

        for klt in VARIANTS:
            # No warmup per variant — we want the same graph state, and the
            # C++ binary doesn't warm per-variant either inside the loop.
            t0 = time.time()
            nodes, times, walk_lens, _ = walk_for_last_batch(
                trw, klt, args.walks_per_node, args.max_walk_len,
                args.walk_bias, args.walk_direction)
            dt = time.time() - t0

            wl = np.asarray(walk_lens, dtype=np.int64)
            results[klt].append(wl)
            print(f'  {klt:<28} time={dt:.3f}s  walks={len(wl)}  '
                  f'mean_len={wl.mean():.4f}  '
                  f'min={wl.min()} max={wl.max()}', flush=True)

    # Concatenate iters per variant for global histogram.
    for v in VARIANTS:
        results[v] = np.concatenate(results[v])

    # Length-histogram side-by-side.
    print('\n' + '=' * 80)
    print(f'Walk-length histogram  (direction={args.walk_direction}, bias={args.walk_bias})')
    print('=' * 80)
    print(f'{"len":>4} | ' + ' | '.join(f'{v:>26}' for v in VARIANTS))
    print('-' * 90)
    max_len_seen = max(wl.max() for wl in results.values())
    for L in range(0, int(max_len_seen) + 1):
        cells = []
        for v in VARIANTS:
            c = int((results[v] == L).sum())
            cells.append(f'{c:>26d}')
        print(f'{L:>4} | ' + ' | '.join(cells))

    # Per-step death rate.
    print('\n' + '=' * 80)
    print('Per-step survival rate  (alive at step k / alive at step k-1)')
    print('=' * 80)
    print(f'{"step":>5} | ' + ' | '.join(f'{v:>26}' for v in VARIANTS))
    print('-' * 90)
    for k in range(1, int(max_len_seen) + 1):
        cells = []
        for v in VARIANTS:
            wl = results[v]
            alive_kminus1 = int((wl >= k).sum())  # reached step k-1
            alive_k       = int((wl >= k + 1).sum())  # reached step k
            rate = alive_k / alive_kminus1 if alive_kminus1 > 0 else 0.0
            cells.append(f'{rate:>20.4f}  ({alive_k}/{alive_kminus1})'
                         if alive_kminus1 < 1e8
                         else f'{rate:>26.4f}')
        print(f'{k:>5} | ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
