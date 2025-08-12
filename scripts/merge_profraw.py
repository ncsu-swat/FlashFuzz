import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Merge profraw files for coverage fuzzing.")
    parser.add_argument("--dll", type=str, required=True, help="DLL name for the coverage fuzzing. (e,g, tf or torch)")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing profraw files.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .profdata file path. Defaults to <dir>/<dll>.profdata",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of threads to use for llvm-profdata merge (default: CPU count)",
    )
    parser.add_argument(
        "--previous",
        type=str,
        default=None,
        required=False,
        help="Path to a previous .profdata file to include in the merge",
    )
    args = parser.parse_args()

    # Validate llvm-profdata availability
    if not shutil.which("llvm-profdata"):
        print("Error: 'llvm-profdata' not found in PATH.", file=sys.stderr)
        sys.exit(127)

    dir_path = os.path.abspath(args.dir)
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)

    # Collect .profraw files explicitly to avoid shell globbing issues
    profraw_files: List[str] = sorted(glob.glob(os.path.join(dir_path, "*.profraw")))
    if not profraw_files:
        print(f"No .profraw files found in {dir_path}. Nothing to merge.")
        

    # Determine output file
    output_path: str = os.path.abspath(args.out) if args.out else os.path.join(dir_path, "merged.profdata")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.previous and  os.path.isfile(args.previous):
        cmd: List[str] = [
            "llvm-profdata",
            "merge",
            "-failure-mode=all",
            "-sparse",
            "-j",
            str(args.threads),
            "-o",
            output_path,
            args.previous,
            *profraw_files,
        ]
    else:
        cmd: List[str] = [
            "llvm-profdata",
            "merge",
            "-failure-mode=all",
            "-sparse",
            "-j",
            str(args.threads),
            "-o",
            output_path,
            *profraw_files,
        ]



    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error: llvm-profdata merge failed.", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)

    print(f"Merged {len(profraw_files)} files into {output_path}")


if __name__ == "__main__":
    main()

