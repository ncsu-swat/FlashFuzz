#!/usr/bin/env python3
"""
Scan a fuzz result directory recursively and compute per‑API validity stats.

For each directory that contains one or more `fuzz-*.log` files, compute:
  - rounds: total executed units (sum of `stat::number_of_executed_units` across logs;
            if absent, sum of `Done X runs in ...` counts)
  - invalid: number of invalid inputs (lines containing 'Exception caught:' or 'CPU Execution error')
  - valid: rounds - invalid (not below 0)
  - validity_ratio: valid / rounds (0 if rounds == 0)

Outputs:
  - Prints a summary table to stdout
  - Writes per‑API `stat.txt` files under each API dir (unless disabled)
  - Writes aggregate `stats.csv` and `summary.txt` under the given base dir

Usage:
  python tools/collect_fuzz_stats.py /path/to/_fuzz_result/<dll><ver>-fuzz-<secs> [--no-write-per-api]
"""

import argparse
import csv
import os
import sys
import glob
from typing import Iterator, List, Tuple, Dict


INVALID_MARKERS = ("Exception caught:", "CPU Execution error", "failed")


def find_api_dirs(base: str) -> Iterator[str]:
    """Yield directories under `base` that contain at least one `fuzz-*.log`."""
    for root, dirs, files in os.walk(base):
        if any(fn.startswith("fuzz-") and fn.endswith(".log") for fn in files):
            yield root


def parse_api_dir(api_dir: str) -> Tuple[int, int]:
    """Return (rounds, invalid_count) for a single API directory.

    Sums across all `fuzz-*.log` files found in the directory.
    """
    rounds_sum = 0
    done_sum = 0
    invalid = 0

    log_files = sorted(glob.glob(os.path.join(api_dir, "fuzz-*.log")))
    if not log_files:
        return 0, 0

    for log_path in log_files:
        try:
            with open(log_path, "r", errors="ignore") as f:
                for line in f:
                    # invalid markers
                    if any(m in line for m in INVALID_MARKERS):
                        invalid += 1

                    if "stat::number_of_executed_units:" in line:
                        try:
                            val = int(line.strip().split(":", 1)[1])
                            rounds_sum += val
                        except Exception:
                            pass
                    elif line.startswith("Done ") and " runs in " in line:
                        # Example: Done 541361 runs in 602 second(s)
                        try:
                            parts = line.split()
                            done_sum += int(parts[1])
                        except Exception:
                            pass
        except Exception:
            # Skip unreadable file
            continue

    rounds = rounds_sum if rounds_sum > 0 else done_sum
    return rounds, invalid


def write_stat_txt(api_dir: str, api_label: str, rounds: int, invalid: int) -> None:
    valid = max(0, rounds - invalid)
    ratio = (valid / rounds) if rounds > 0 else 0.0
    lines = [
        f"api: {api_label}",
        f"rounds: {rounds}",
        f"invalid: {invalid}",
        f"valid: {valid}",
        f"validity_ratio: {ratio:.6f}",
    ]
    path = os.path.join(api_dir, "stat.txt")
    try:
        with open(path, "w") as sf:
            sf.write("\n".join(lines) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Aggregate fuzz stats for all APIs under a directory.")
    ap.add_argument("--base", help="Base directory to scan (e.g., _fuzz_result/torch2.7-fuzz-600s)")
    ap.add_argument("--no-write-per-api", action="store_true", help="Do not write per-API stat.txt files")
    ap.add_argument("--csv", default="stats.csv", help="CSV filename to write under base (default: stats.csv)")
    ap.add_argument("--summary", default="summary.txt", help="Summary filename to write under base (default: summary.txt)")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        print(f"Base path not found or not a directory: {base}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, object]] = []
    total_rounds = 0
    total_invalid = 0

    api_dirs = list(sorted(find_api_dirs(base)))
    if not api_dirs:
        print(f"No API directories with fuzz logs found under: {base}")
        sys.exit(0)

    for api_dir in api_dirs:
        rounds, invalid = parse_api_dir(api_dir)
        valid = max(0, rounds - invalid)
        ratio = (valid / rounds) if rounds > 0 else 0.0
        # API label as path relative to base for clarity
        api_label = os.path.relpath(api_dir, base)

        if not args.no_write_per_api:
            write_stat_txt(api_dir, api_label, rounds, invalid)

        results.append({
            "api": api_label,
            "rounds": rounds,
            "invalid": invalid,
            "valid": valid,
            "validity_ratio": ratio,
            "path": api_dir,
        })

        total_rounds += rounds
        total_invalid += invalid

    # Sort for display: by api name
    results.sort(key=lambda r: r["api"])  # type: ignore

    # Print a short table
    print(f"Scanned base: {base}")
    print(f"APIs found: {len(results)}")
    for r in results:
        print(f"- {r['api']}: rounds={r['rounds']} invalid={r['invalid']} valid={r['valid']} ratio={r['validity_ratio']:.6f}")

    overall_valid = max(0, total_rounds - total_invalid)
    overall_ratio = (overall_valid / total_rounds) if total_rounds > 0 else 0.0
    print("Overall:")
    print(f"- rounds={total_rounds} invalid={total_invalid} valid={overall_valid} ratio={overall_ratio:.6f}")

    # Write CSV and summary under base
    csv_path = os.path.join(base, args.csv)
    try:
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=["api", "rounds", "invalid", "valid", "validity_ratio", "path"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)  # type: ignore
        print(f"Wrote: {csv_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")

    summary_path = os.path.join(base, args.summary)
    try:
        with open(summary_path, "w") as sf:
            sf.write(f"Base: {base}\n")
            sf.write(f"APIs: {len(results)}\n")
            sf.write(f"Total rounds: {total_rounds}\n")
            sf.write(f"Total invalid: {total_invalid}\n")
            sf.write(f"Total valid: {overall_valid}\n")
            sf.write(f"Overall validity ratio: {overall_ratio:.6f}\n")
        print(f"Wrote: {summary_path}")
    except Exception as e:
        print(f"Failed to write summary: {e}")


if __name__ == "__main__":
    main()

