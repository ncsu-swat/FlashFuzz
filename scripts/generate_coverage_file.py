import argparse
import os
import glob
import multiprocessing
# from functools import partial  # removed: no longer used
# Added imports for performance and robustness
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def _safe_symlink(src: str, dst: str):
    """Create or update a symlink. If not permitted, fall back to copy."""
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            try:
                # If it's already the correct symlink, keep it
                if os.path.islink(dst) and os.readlink(dst) == src:
                    return
                # Otherwise remove and recreate
                os.remove(dst)
            except OSError:
                pass
        os.symlink(src, dst)
    except OSError:
        # Fall back to copy if symlink not allowed
        shutil.copy2(src, dst)
        # Ensure executable bit for binaries
        if os.path.basename(dst) == 'fuzz':
            os.chmod(dst, 0o755)


def prepare_directory(dir_path: str, binary_src: str, lib_paths: list[str]):
    """Ensure required runtime files exist in dir via fast symlinks (or copies)."""
    os.makedirs(dir_path, exist_ok=True)
    # Link binary
    _safe_symlink(binary_src, os.path.join(dir_path, 'fuzz'))
    # Link libraries
    for lib in lib_paths:
        _safe_symlink(lib, os.path.join(dir_path, os.path.basename(lib)))


def run_fuzzer_for_seed(seed: str, dir_path: str, timeout: int | None = None) -> int:
    """Run the fuzzer for a single seed file inside dir_path. Returns exit code."""
    env = os.environ.copy()
    env['LLVM_PROFILE_FILE'] = f"{seed}.profraw"
    # Reduce thread oversubscription from TF/BLAS when running many workers
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('TF_NUM_INTRAOP_THREADS', '1')
    env.setdefault('TF_NUM_INTEROP_THREADS', '1')
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        # Silence stdout/stderr for performance; comment out if debugging
        with open(os.devnull, 'wb') as devnull:
            result = subprocess.run(
                ['./fuzz', seed],
                cwd=dir_path,
                env=env,
                stdout=devnull,
                stderr=devnull,
                timeout=timeout if timeout and timeout > 0 else None,
                check=False,
            )
        return result.returncode
    except subprocess.TimeoutExpired:
        return 124  # common code for timeout
    except Exception:
        return 1


def worker_task(task: tuple[str, str, int | None]) -> tuple[str, str, int]:
    """Top-level picklable worker wrapper for ProcessPoolExecutor.
    task: (seed, dir_path, timeout)
    returns: (dir_path, seed, returncode)
    """
    seed, d, timeout = task
    rc = run_fuzzer_for_seed(seed, d, timeout=timeout)
    return (d, seed, rc)


def main():
    parser = argparse.ArgumentParser(description="Generate coverage files.")
    parser.add_argument("--dll", type=str, required=True, help="Input directory containing coverage data.")
    parser.add_argument("--api", type=str, required=True, help="API name")
    parser.add_argument("--itv", type=int, default=60, help="Interval for coverage collection in seconds.")
    # New performance-related options
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: CPU count)")
    parser.add_argument("--timeout", type=int, default=None, help="Per-seed timeout in seconds")
    args = parser.parse_args()

    if args.dll == "tf" and args.api != "all":
        # Ensure TF shared libraries are available at repo root once (cheap, idempotent)
        try:
            tf_libs = glob.glob("/root/tensorflow/bazel-bin/tensorflow/libtensorflow_*.so*")
            for lib in tf_libs:
                dst = os.path.join(os.getcwd(), os.path.basename(lib))
                if not os.path.exists(dst):
                    shutil.copy2(lib, dst)
        except Exception as e:
            print(f"Warning: failed to copy TensorFlow libs once: {e}")

        seeds_dir = f"corpus_itv_{args.itv}"

        # Absolute paths
        binary_src = os.path.realpath("./fuzz")
        libtensorflow_path = os.path.realpath("libtensorflow_cc.so.2")
        libtensorflow_framework_path = os.path.realpath("libtensorflow_framework.so.2")

        # Validate required files exist
        missing = [p for p in [binary_src, libtensorflow_path, libtensorflow_framework_path] if not os.path.exists(p)]
        if missing:
            print(f"Missing required files: {missing}")
            return

        # Discover directories
        dirs = [d for d in glob.glob(os.path.join(seeds_dir, "*")) if os.path.isdir(d)]
        if not dirs:
            print(f"No directories found in {seeds_dir}")
            return

        # Prepare all directories quickly using threads (I/O bound)
        lib_paths = [libtensorflow_path, libtensorflow_framework_path]
        with ThreadPoolExecutor(max_workers=min(32, len(dirs) or 1)) as tpool:
            for d in dirs:
                tpool.submit(prepare_directory, d, binary_src, lib_paths)

        # Build global task list across all dirs (avoid nested pools)
        tasks: list[tuple[str, str, int | None]] = []  # (seed, dir, timeout)
        seeds_per_dir: dict[str, list[str]] = {}
        for d in dirs:
            all_files = os.listdir(d)
            seeds = [
                f for f in all_files
                if f not in ['fuzz', 'libtensorflow_cc.so.2', 'libtensorflow_framework.so.2']
                and not os.path.isdir(os.path.join(d, f))
                and not (f.endswith('.profraw') or f.endswith('.profdata'))
            ]
            if not seeds:
                continue
            seeds_per_dir[d] = seeds
            for s in seeds:
                tasks.append((s, d, args.timeout))

        if not tasks:
            print("No input seeds found across directories, nothing to do.")
            return

        # Determine process parallelism
        max_workers = args.workers if args.workers and args.workers > 0 else multiprocessing.cpu_count()
        print(f"Running fuzzer in parallel on {len(tasks)} seeds across {len(seeds_per_dir)} dirs with {max_workers} workers...")

        # Compute a reasonable chunksize to reduce overhead
        chunksize = max(1, len(tasks) // (max_workers * 4) if max_workers else 1)

        results: list[tuple[str, str, int]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for res in pool.map(worker_task, tasks, chunksize=chunksize):
                results.append(res)

        # Optional: basic summary
        failures = [(d, s, rc) for (d, s, rc) in results if rc != 0]
        if failures:
            print(f"Completed with {len(failures)} failures out of {len(results)} runs.")
        else:
            print(f"All {len(results)} runs completed successfully.")

        # Merge profraw per directory (can be done in parallel, but typically fast)
        def _merge_dir(d: str):
            profraws = [f for f in os.listdir(d) if f.endswith('.profraw')]
            if not profraws:
                return
            merged = os.path.join(d, 'merged.profdata')
            cmd = ['llvm-profdata', 'merge', '-sparse', '-o', merged] + [os.path.join(d, f) for f in profraws]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Merge failed in {d}: {e}")

        with ThreadPoolExecutor(max_workers=min(16, len(seeds_per_dir) or 1)) as tpool:
            for d in seeds_per_dir.keys():
                tpool.submit(_merge_dir, d)

        # Clean up only .profraw files to preserve seeds and binaries
        for d in seeds_per_dir.keys():
            for f in os.listdir(d):
                if f.endswith('.profraw'):
                    try:
                        os.remove(os.path.join(d, f))
                    except OSError:
                        pass

if __name__ == "__main__":
    # This check is important for robust multiprocessing on all platforms
    main()
