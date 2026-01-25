#!/usr/bin/env python3
"""
Tempest ablation runner (Python harness)

Runs 4 variants × N repetitions, storing:
- stdout.txt
- nsys.qdrep
- ncu.ncu-rep

Then parses stdout summaries and prints mean ± std per variant.

Usage:
  python run_ablation.py \
    --dataset-base-path /mnt/lustre/users/inf/ms2420/tgbl-datasets \
    [--cuda-tools-path /usr/local/cuda/bin] \
    [--code-path ~/] \
    [--results-path ./ablation-results] \
    [--runs 5] \
    [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Parsing helpers
# -----------------------------
SUMMARY_PATTERNS = {
    "total_ingestion_time_sec": re.compile(r"^Total ingestion time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "mean_ingestion_time_sec": re.compile(r"^Mean ingestion time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "total_walk_time_sec": re.compile(r"^Total walk time:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "mean_walk_time_per_batch_sec": re.compile(r"^Mean walk time/batch:\s*([0-9eE+\-\.]+)\s*sec\s*$"),
    "total_walks": re.compile(r"^Total walks:\s*([0-9]+)\s*$"),
    "final_avg_walk_length": re.compile(r"^Final avg walk length:\s*([0-9eE+\-\.]+)\s*$"),
    "throughput_walks_per_sec": re.compile(r"^Throughput:\s*([0-9eE+\-\.]+)\s*walks/sec\s*$"),
}


def parse_stdout_summary(stdout_path: Path) -> Dict[str, float]:
    """
    Parse the final === Summary === block from stdout.
    Returns a dict of metrics.
    Raises ValueError if parsing fails.
    """
    text = stdout_path.read_text(errors="replace").splitlines()

    # Find "=== Summary ===" then parse following lines
    try:
        idx = next(i for i, line in enumerate(text) if line.strip() == "=== Summary ===")
    except StopIteration:
        raise ValueError(f"Could not find '=== Summary ===' in {stdout_path}")

    metrics: Dict[str, float] = {}
    for line in text[idx + 1 :]:
        line = line.strip()
        if not line:
            continue
        for key, pat in SUMMARY_PATTERNS.items():
            m = pat.match(line)
            if m:
                val_str = m.group(1)
                # ints for total_walks, floats otherwise
                if key == "total_walks":
                    metrics[key] = float(int(val_str))
                else:
                    metrics[key] = float(val_str)
                break

    required = [
        "total_ingestion_time_sec",
        "mean_ingestion_time_sec",
        "total_walk_time_sec",
        "mean_walk_time_per_batch_sec",
        "total_walks",
        "final_avg_walk_length",
        "throughput_walks_per_sec",
    ]
    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValueError(f"Missing fields {missing} while parsing {stdout_path}")

    return metrics


# -----------------------------
# Experiment definition
# -----------------------------
@dataclass(frozen=True)
class Variant:
    key: str               # directory name
    description: str       # human-friendly label
    binary_rel: str        # relative to CODE_PATH
    picker: str            # exponential_index / exponential_weight
    kernel_mode: str       # full_walk / step_based

    def command(self, code_path: Path, dataset_csv: Path) -> List[str]:
        """
        Build the ablation_streaming command.
        Args list matches your binary's CLI:
          <csv> 1 <picker> <kernel_mode> 0 5 10 3
        """
        bin_path = (code_path / self.binary_rel).resolve()
        return [
            str(bin_path),
            str(dataset_csv),
            "1",
            self.picker,
            self.kernel_mode,
            "0",
            "5",
            "10",
            "3",
        ]


def which_or_die(prog: str) -> str:
    p = shutil.which(prog)
    if not p:
        raise FileNotFoundError(f"Could not find '{prog}' in PATH")
    return p


def tool_path(cuda_tools_path: Optional[str], name: str) -> str:
    """
    Resolve tool path.
    - If cuda_tools_path is None: resolve via PATH
    - Else: resolve from the given directory
    """
    if cuda_tools_path is None:
        return which_or_die(name)

    candidate = (Path(cuda_tools_path) / name).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find '{name}' at {candidate}")
    return str(candidate)


def run_cmd(
        cmd: List[str],
        cwd: Optional[Path],
        stdout_path: Optional[Path],
        stderr_path: Optional[Path],
        dry_run: bool,
) -> int:
    """
    Run a command, redirecting stdout/stderr if paths provided.
    Returns exit code.
    """
    cmd_str = " ".join(cmd)
    print(f"  $ {cmd_str}")

    if dry_run:
        return 0

    stdout_f = open(stdout_path, "w") if stdout_path else subprocess.DEVNULL
    stderr_f = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
            env=os.environ.copy(),
        )
        return proc.returncode
    finally:
        if stdout_path:
            stdout_f.close()
        if stderr_path:
            stderr_f.close()


def format_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Returns (mean, std). std=0 if only one value.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    return (mean(values), stdev(values))


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-tools-path", default=None, help="Optional directory containing nsys/ncu (default: use PATH)")
    ap.add_argument("--dataset-base-path", required=True, help="Directory containing ml_tgbl-coin.csv")
    ap.add_argument("--code-path", default=str(Path.home()), help="Path containing temporal-random-walk dirs (default: HOME)")
    ap.add_argument("--results-path", default=str(Path.cwd() / "ablation-results"), help="Output dir (default: ./ablation-results)")
    ap.add_argument("--runs", type=int, default=5, help="Number of repetitions per variant (default: 5)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    args = ap.parse_args()

    cuda_tools_path = args.cuda_tools_path
    dataset_base = Path(args.dataset_base_path).expanduser().resolve()
    code_path = Path(args.code_path).expanduser().resolve()
    results_base = Path(args.results_path).expanduser().resolve()
    runs = int(args.runs)
    dry_run = bool(args.dry_run)

    dataset_csv = dataset_base / "ml_tgbl-coin.csv"
    if not dataset_csv.exists():
        print(f"ERROR: dataset not found: {dataset_csv}", file=sys.stderr)
        return 2

    # Tools
    nsys = tool_path(cuda_tools_path, "nsys")
    ncu = tool_path(cuda_tools_path, "ncu")

    # Define variants (your chosen 4)
    variants: List[Variant] = [
        Variant(
            key="v0_fullwalk_index_inkernel",
            description="in-kernel RNG + full_walk + index (exponential_index)",
            binary_rel="temporal-random-walk/build/bin/ablation_streaming",
            picker="exponential_index",
            kernel_mode="full_walk",
        ),
        Variant(
            key="v1_fullwalk_index_pregen",
            description="pregenerated RNG + full_walk + index (exponential_index)",
            binary_rel="temporal-random-walk-pregenerated-rng/build/bin/ablation_streaming",
            picker="exponential_index",
            kernel_mode="full_walk",
        ),
        Variant(
            key="v2_step_index_inkernel",
            description="in-kernel RNG + step_based + index (exponential_index)",
            binary_rel="temporal-random-walk/build/bin/ablation_streaming",
            picker="exponential_index",
            kernel_mode="step_based",
        ),
        Variant(
            key="v3_fullwalk_weight_inkernel",
            description="in-kernel RNG + full_walk + weight (exponential_weight)",
            binary_rel="temporal-random-walk/build/bin/ablation_streaming",
            picker="exponential_weight",
            kernel_mode="full_walk",
        ),
    ]

    # Validate binaries exist
    for v in variants:
        bin_path = (code_path / v.binary_rel).resolve()
        if not bin_path.exists():
            print(f"ERROR: missing binary for {v.key}: {bin_path}", file=sys.stderr)
            return 2

    results_base.mkdir(parents=True, exist_ok=True)

    # Collect metrics for summary:
    # metrics_by_variant[variant_key][metric_name] -> List[float]
    metrics_by_variant: Dict[str, Dict[str, List[float]]] = {
        v.key: {k: [] for k in SUMMARY_PATTERNS.keys()} for v in variants
    }

    print("========================================")
    print(" Tempest Ablation Runner")
    print("========================================")
    print(f"Dataset: {dataset_csv}")
    print(f"Code path: {code_path}")
    print(f"Results: {results_base}")
    print(f"Runs per variant: {runs}")
    print(f"nsys: {nsys}")
    print(f"ncu:  {ncu}")
    print("")

    # -----------------------------
    # Warmup run (baseline only)
    # -----------------------------
    print("\n========================================")
    print(" GPU WARMUP RUN (baseline configuration)")
    print("========================================")

    # Baseline = first variant (v0_fullwalk_index_inkernel)
    warmup_variant = variants[0]
    warmup_cmd = warmup_variant.command(
        code_path=code_path,
        dataset_csv=dataset_csv
    )

    print("Warmup variant:", warmup_variant.key)
    print("Warmup command:")
    print("  ", " ".join(warmup_cmd))

    if not dry_run:
        warmup_proc = subprocess.run(
            warmup_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            env=os.environ.copy(),
        )

        if warmup_proc.returncode != 0:
            print(
                f"WARNING: warmup run failed with return code "
                f"{warmup_proc.returncode}. Continuing anyway.",
                file=sys.stderr
            )

    print("Warmup completed. Starting measured runs.\n")

    for run_id in range(1, runs + 1):
        run_dir = results_base / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"==============================")
        print(f"Run {run_id}/{runs} -> {run_dir}")
        print(f"==============================")

        for v in variants:
            var_dir = run_dir / v.key
            var_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nVariant: {v.key}")
            print(f"  {v.description}")
            print(f"  Output: {var_dir}")

            # (1) Normal run (stdout)
            stdout_txt = var_dir / "stdout.txt"
            stderr_txt = var_dir / "stderr.txt"
            cmd = v.command(code_path=code_path, dataset_csv=dataset_csv)

            rc = run_cmd(cmd, cwd=None, stdout_path=stdout_txt, stderr_path=stderr_txt, dry_run=dry_run)
            if rc != 0:
                print(f"ERROR: normal run failed (rc={rc}) for {v.key} run-{run_id}", file=sys.stderr)
                return 3

            # (2) nsys
            # output base must be without extension; nsys will create .qdrep
            nsys_base = var_dir / "nsys"
            nsys_cmd = [
                nsys, "profile",
                "-o", str(nsys_base),
                "--trace=cuda,nvtx,osrt",
                "--cuda-memory-usage=true",
                *cmd,
            ]
            rc = run_cmd(nsys_cmd, cwd=None, stdout_path=var_dir / "nsys_stdout.txt", stderr_path=var_dir / "nsys_stderr.txt", dry_run=dry_run)
            if rc != 0:
                print(f"ERROR: nsys failed (rc={rc}) for {v.key} run-{run_id}", file=sys.stderr)
                return 4

            # (3) ncu
            # output base must be without extension; ncu will create .ncu-rep
            ncu_base = var_dir / "ncu"
            ncu_cmd = [
                ncu,
                "--target-processes", "all",
                "--replay-mode", "kernel",
                "-o", str(ncu_base),
                "--kernel-name", "regex:generate_random_walks_kernel|pick_start_edges_kernel|pick_intermediate_edges_kernel",
                "--kernel-name-base", "demangled",
                "--set", "speedOfLight",
                *cmd,
            ]
            rc = run_cmd(ncu_cmd, cwd=None, stdout_path=var_dir / "ncu_stdout.txt", stderr_path=var_dir / "ncu_stderr.txt", dry_run=dry_run)
            if rc != 0:
                print(f"ERROR: ncu failed (rc={rc}) for {v.key} run-{run_id}", file=sys.stderr)
                return 5

            # Parse summary from stdout
            if not dry_run:
                try:
                    parsed = parse_stdout_summary(stdout_txt)
                except Exception as e:
                    print(f"ERROR: failed to parse stdout summary for {v.key} run-{run_id}: {e}", file=sys.stderr)
                    return 6

                for k, val in parsed.items():
                    # stored as float; total_walks is float(int) for convenience
                    metrics_by_variant[v.key][k].append(float(val))

    # -----------------------------
    # Final summary
    # -----------------------------
    print("\n========================================")
    print(" ABLATION SUMMARY (mean ± std)")
    print(f" Over {runs} runs per variant")
    print("========================================")

    def pm(m: float, s: float, fmt: str = "{:.6f}") -> str:
        return f"{fmt.format(m)} ± {fmt.format(s)}"

    for v in variants:
        m = metrics_by_variant[v.key]

        ingest_mean, ingest_std = format_mean_std(m["total_ingestion_time_sec"])
        walk_mean, walk_std = format_mean_std(m["total_walk_time_sec"])
        thr_mean, thr_std = format_mean_std(m["throughput_walks_per_sec"])
        len_mean, len_std = format_mean_std(m["final_avg_walk_length"])

        print(f"\n{v.key}")
        print(f"  {v.description}")
        print(f"  Total ingestion time (sec): {pm(ingest_mean, ingest_std, '{:.3f}')}")
        print(f"  Total walk time (sec):      {pm(walk_mean, walk_std, '{:.3f}')}")
        print(f"  Throughput (walks/sec):     {pm(thr_mean, thr_std, '{:.3e}')}")
        print(f"  Final avg walk length:      {pm(len_mean, len_std, '{:.3f}')}")

    # Optional: write a machine-readable CSV summary
    summary_csv = results_base / "summary.csv"
    with summary_csv.open("w") as f:
        f.write("variant,total_ingest_mean,total_ingest_std,total_walk_mean,total_walk_std,thr_mean,thr_std,avg_len_mean,avg_len_std\n")
        for v in variants:
            m = metrics_by_variant[v.key]
            ingest_mean, ingest_std = format_mean_std(m["total_ingestion_time_sec"])
            walk_mean, walk_std = format_mean_std(m["total_walk_time_sec"])
            thr_mean, thr_std = format_mean_std(m["throughput_walks_per_sec"])
            len_mean, len_std = format_mean_std(m["final_avg_walk_length"])
            f.write(f"{v.key},{ingest_mean},{ingest_std},{walk_mean},{walk_std},{thr_mean},{thr_std},{len_mean},{len_std}\n")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
