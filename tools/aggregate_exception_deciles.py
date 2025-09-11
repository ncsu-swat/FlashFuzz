#!/usr/bin/env python3
import os
import sys
from math import ceil


def iter_fuzz_logs(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("fuzz-") and fn.endswith(".log"):
                yield os.path.join(dirpath, fn)


def count_lines(path: str) -> int:
    # Efficient enough for large files; avoids reading into memory
    n = 0
    with open(path, 'rb', buffering=1024 * 1024) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def analyze_file(path: str):
    """
    Returns:
      total_lines: int
      decile_exc: list of length 10 with exception counts per decile
      decile_lens: list of length 10 with line counts per decile
    """
    total_lines = count_lines(path)
    if total_lines <= 0:
        return 0, [0] * 10, [0] * 10

    binsize = ceil(total_lines / 10)
    # Compute decile lengths deterministically
    decile_lens = []
    for i in range(10):
        start = i * binsize + 1
        end = min((i + 1) * binsize, total_lines)
        if start > total_lines:
            start = total_lines
            end = total_lines
        decile_lens.append(max(0, end - start + 1))

    # Second pass: count exceptions into deciles
    decile_exc = [0] * 10
    target = b"INVALID_ARGUMENT:"
    with open(path, 'rb', buffering=1024 * 1024) as f:
        lineno = 0
        for raw in f:
            lineno += 1
            if target in raw:
                bin_idx = (lineno - 1) // binsize
                if bin_idx < 0:
                    bin_idx = 0
                elif bin_idx > 9:
                    bin_idx = 9
                decile_exc[bin_idx] += 1

    return total_lines, decile_exc, decile_lens


def main():
    if len(sys.argv) != 2:
        print("Usage: aggregate_exception_deciles.py <root_dir>")
        sys.exit(2)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Directory not found: {root}")
        sys.exit(1)

    files = sorted(iter_fuzz_logs(root))
    if not files:
        print(f"No fuzz-*.log files found under {root}")
        return

    total_lines_all = 0
    total_exc_all = 0
    agg_exc = [0] * 10
    agg_lens = [0] * 10

    print(f"Found {len(files)} log files under {root}")
    for idx, path in enumerate(files, 1):
        tl, dec_exc, dec_len = analyze_file(path)
        total_lines_all += tl
        total_exc_all += sum(dec_exc)
        for i in range(10):
            agg_exc[i] += dec_exc[i]
            agg_lens[i] += dec_len[i]
        if idx % 50 == 0:
            print(f"Processed {idx} files...", file=sys.stderr)

    def rate_per_1k(exc: int, lines: int) -> float:
        return (exc * 1000.0 / lines) if lines > 0 else 0.0

    overall_rate = rate_per_1k(total_exc_all, total_lines_all)
    print(f"Total lines (all logs): {total_lines_all}")
    print(f"Total \"Exception caught\" (all logs): {total_exc_all}")
    print(f"Overall rate: {overall_rate:.2f} per 1k lines\n")

    print("Aggregated per-file deciles (1..10):")
    print("Decile\tAggLines\tExceptions\tRate(/1k lines)")
    for i in range(10):
        lines = agg_lens[i]
        exc = agg_exc[i]
        rate = rate_per_1k(exc, lines)
        print(f"{i+1}\t{lines}\t{exc}\t{rate:.2f}")

    lines_first = sum(agg_lens[:5])
    exc_first = sum(agg_exc[:5])
    lines_last = sum(agg_lens[5:])
    exc_last = sum(agg_exc[5:])
    rate_first = rate_per_1k(exc_first, lines_first)
    rate_last = rate_per_1k(exc_last, lines_last)
    print(f"\nFirst half rate (/1k): {rate_first:.2f}")
    print(f"Second half rate (/1k): {rate_last:.2f}")


if __name__ == "__main__":
    main()

