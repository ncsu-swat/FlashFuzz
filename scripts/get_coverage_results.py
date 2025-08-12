#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import re
from typing import List, Tuple, cast

from bs4 import BeautifulSoup


def extract_coverage_data(html_content: str,
        required_substrings: List[str]) -> Tuple[List[Tuple[str, int, int]], int, int]:
    """
    Parse llvm-cov HTML index and collect covered lines for files whose path
    contains ALL required_substrings (logical AND). Returns:
        - list of (path, covered_lines, total_lines)
        - sum_covered
        - sum_total
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('tr', class_='light-row')

    results: List[Tuple[str, int, int]] = []
    sum_cov = 0
    sum_tot = 0

    for row in rows:
        tds = row.find_all('td')
        # Need at least 5 columns because we use index 4
        if len(tds) < 5:
            continue

        link = tds[0].find('a')
        if not link:
            continue

        path = link.get_text(strip=True)
        if required_substrings and not all(sub in path for sub in required_substrings):
            continue

        # Column 5 (index 4) has "X.YY% (covered/total)"
        pre_tag = tds[4].find('pre')
        if not pre_tag:
            continue
        coverage_text = pre_tag.get_text(strip=True)

        # Match (covered/total)
        m = re.search(r'\((\d+)/(\d+)\)', coverage_text)
        if not m:
            continue

        covered = int(m.group(1))
        total = int(m.group(2))
        results.append((path, covered, total))
        sum_cov += covered
        sum_tot += total

    return results, sum_cov, sum_tot


def main():
    parser = argparse.ArgumentParser(description="Extract kernel coverage (HTML parsing path).")
    parser.add_argument("--coverage_file", required=True, help="Path to merged .profdata.")
    parser.add_argument(
        "--binary",
        action="append",
        required=True,
        help="Instrumented binary / shared object (repeat for multiple)."
    )
    parser.add_argument(
        "--require",
        action="append",
        default=["tensorflow/core/kernels"],
        help="Substring that must appear in file path (repeatable, default: tensorflow/core/kernels)."
    )
    parser.add_argument(
        "--html-dir",
        default="coverage_html",
        help="Directory to emit llvm-cov HTML (default: coverage_html)."
    )
    parser.add_argument(
        "--out",
        default="coverage_results.txt",
        help="Output summary text file (default: coverage_results.txt)."
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="Keep existing HTML if already present (skip regeneration)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.coverage_file):
        print(f"Error: coverage file not found: {args.coverage_file}", file=sys.stderr)
        sys.exit(1)

    for b in args.binary:
        if not os.path.isfile(b):
            print(f"Error: binary not found: {b}", file=sys.stderr)
            sys.exit(1)

    # Generate HTML (index.html needed for current parsing approach)
    index_html = os.path.join(args.html_dir, "index.html")
    if not (args.keep_html and os.path.isfile(index_html)):
        os.makedirs(args.html_dir, exist_ok=True)
        binaries = cast(List[str], args.binary)
        cmd: List[str] = [
            "llvm-cov",
            "show",
            *binaries,
            "--show-branches=count",
            f"--instr-profile={args.coverage_file}",
            "-format=html",
            f"-output-dir={args.html_dir}"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            print(f"Error: llvm-cov show command failed: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(e.returncode)

    if not os.path.isfile(index_html):
        print(f"Error: HTML index not found at {index_html}", file=sys.stderr)
        sys.exit(1)

    with open(index_html, "r", encoding="utf-8") as fh:
        html_content = fh.read()

    per_file, covered_sum, total_sum = extract_coverage_data(html_content, args.require)
    percent = (covered_sum / total_sum * 100) if total_sum else 100.0

    # Write summary
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Filters (ALL required): {args.require}\n")
        f.write(f"Files matched: {len(per_file)}\n")
        f.write(f"Covered lines: {covered_sum}\n")
        f.write(f"Total lines: {total_sum}\n")
        f.write(f"Line coverage: {percent:.2f}%\n")
        f.write("\nPer-file (covered/total pct path):\n")
        for path, cov, tot in sorted(per_file, key=lambda x: (-x[1], x[0])):
            pct = (cov / tot * 100) if tot else 100.0
            f.write(f"{cov}/{tot} {pct:6.2f}% {path}\n")

    print(f"Line coverage (filtered): {covered_sum}/{total_sum} = {percent:.2f}%")
    print(f"Wrote summary to {args.out}")

if __name__ == "__main__":
    main()
