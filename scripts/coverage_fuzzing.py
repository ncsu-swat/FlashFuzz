import os
import sys
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional, Sequence, List, Dict
from types import FrameType


class IntervalFuzzRunner:
    def __init__(
        self,
        corpus_dir: str,
        coverage_dir: str,
        interval_sec: int,
        max_time_sec: int,
        fuzz_args: Optional[Sequence[str]] = None,
        api_name: str = "unamed"
    ) -> None:
        self.corpus_dir = corpus_dir
        self.coverage_dir = Path(coverage_dir)
        self.interval_sec = max(1, interval_sec)
        self.max_time_sec = max(1, max_time_sec)
        self.fuzz_args = list(fuzz_args) if fuzz_args else []
        self._stop = False

        self.coverage_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.coverage_dir / ".work"
        self.work_dir.mkdir(exist_ok=True)

        # Paths to scripts/binaries
        self.build_sh = "./build.sh"
        self.fuzz_bin = "./fuzz"

        # API name for profraw file
        self.api_name = api_name

    def build(self) -> None:
        print("Building with build.sh ...")
        if not os.path.exists(self.build_sh):
            raise FileNotFoundError("build.sh not found in current directory")

        subprocess.run(["bash", str(self.build_sh)], check=True)
        print("✓ Build completed")

        # Ensure fuzz binary exists
        if not os.path.exists(self.fuzz_bin):
            raise FileNotFoundError("fuzz binary not found after build")
        # Make sure it's executable
        os.chmod(self.fuzz_bin, os.stat(self.fuzz_bin).st_mode | 0o111)

    def _fuzz_command(self, duration: int) -> List[str]:
        cmd: List[str] = [str(self.fuzz_bin), self.corpus_dir]

        env = os.environ
        jobs = env.get("JOBS")
        workers = env.get("WORKERS")
        max_len = env.get("MAX_LEN")
        rss_limit = env.get("RSS_LIMIT")

        def add_opt(flag: str, val: Optional[str]) -> None:
            if val not in (None, ""):
                cmd.append(f"{flag}={val}")

        add_opt("-jobs", jobs)
        add_opt("-workers", workers)
        add_opt("-max_len", max_len)
        add_opt("-rss_limit_mb", rss_limit)

        cmd.extend([
            "-prefer_small=0",
            "-use_value_profile=1",
            "-mutate_depth=100",
            "-entropic=1",
            "-use_counters=1",
            "-ignore_crashes=1",
            "-reduce_inputs=0",
            "-len_control=0",
            f"-max_total_time={duration}",
            "-print_final_stats=1",
        ])

        cmd.extend(self.fuzz_args)
        return cmd

    def _run_once(self, idx: int, duration: int) -> None:
        # Use fixed interval directory names: 0-interval, interval-2*interval, ...
        interval_start = idx * self.interval_sec
        interval_end = (idx + 1) * self.interval_sec
        interval_name = f"{interval_start}-{interval_end}"
        interval_dir = self.coverage_dir / interval_name
        interval_dir.mkdir(parents=True, exist_ok=True)

        # Temp location for raw profiles
        tmp_dir = self.work_dir / f"run_{interval_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Use {api}.profraw as the profile file name
        profraw_path = tmp_dir / f"{self.api_name}.profraw"

        env: Dict[str, str] = os.environ.copy()
        env["LLVM_PROFILE_FILE"] = str(profraw_path.resolve())

        ld_path = env.get("LD_LIBRARY_PATH", "")
        cwd_path = os.getcwd()
        env["LD_LIBRARY_PATH"] = f"{cwd_path}:{ld_path}" if ld_path else cwd_path

        cmd = self._fuzz_command(duration)
        print(f"\nInterval {interval_name}s: starting fuzzing")
        print(f"Command: {' '.join(cmd)}")
        print(f"LLVM_PROFILE_FILE: {env['LLVM_PROFILE_FILE']}")

        grace = max(15, duration // 10)
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line.rstrip())
            proc.wait(timeout=duration + grace)
        except subprocess.TimeoutExpired:
            print("✗ Interval watchdog timeout exceeded, attempting graceful termination...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing process...")
                proc.kill()
                proc.wait()
        except KeyboardInterrupt:
            self._stop = True
            print("\nInterrupted by user. Stopping after current interval...")
            proc.terminate()
            proc.wait()
        finally:
            rc = proc.returncode
            print(f"Fuzzer exited with code: {rc}")

        # Move non-empty profraw file into the interval dir, always as {api}.profraw
        moved_count = 0
        if profraw_path.is_file() and profraw_path.stat().st_size > 0:
            shutil.move(str(profraw_path), interval_dir / f"{self.api_name}.profraw")
            moved_count = 1

        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        print(f"Interval {interval_name}s: moved {moved_count} .profraw files to {interval_dir}")

        # self._merge_profraw(interval_dir, interval_name)

    def _merge_profraw(self, interval_dir: Path, interval_name: str) -> None:
        profraw_file = interval_dir / f"{self.api_name}.profraw"
        if not profraw_file.exists():
            print("  ℹ No .profraw file to merge for this interval")
            return

        merged = interval_dir / f"merged_{interval_name}.profdata"
        cmd = ["llvm-profdata", "merge", "-sparse", str(profraw_file), "-o", str(merged)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ Created merged profdata: {merged}")
        except FileNotFoundError:
            print("  ✗ llvm-profdata not found, skipping merge")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to merge profiles: {e}\n  stdout: {e.stdout}\n  stderr: {e.stderr}")

    def run(self) -> None:
        self.build()
        total = self.max_time_sec
        step = self.interval_sec
        num_intervals = (total + step - 1) // step  # ceil division

        for idx in range(num_intervals):
            if self._stop:
                break
            # Last interval may be shorter
            remaining = total - idx * step
            duration = min(step, remaining)
            self._run_once(idx, duration)

        print("\nAll intervals completed.")

    def stop(self) -> None:
        self._stop = True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interval fuzzing with coverage collection")
    parser.add_argument("--corpus", default="corpus", help="Corpus directory")
    parser.add_argument("--coverage-dir", default="coverage_data", help="Coverage data output directory")
    parser.add_argument("--interval", type=int, required=True, help="Interval duration in seconds")
    parser.add_argument("--max-time", type=int, required=True, help="Total fuzzing time in seconds")
    parser.add_argument("--fuzz-args", nargs="*", help="Additional arguments passed to the fuzz binary")
    parser.add_argument("--build-only", action="store_true", help="Only build via build.sh and exit")
    parser.add_argument("--api", type=str, required=True, help="API to fuzz (default: all)")
    args = parser.parse_args()

    runner = IntervalFuzzRunner(
        corpus_dir=args.corpus,
        coverage_dir=args.coverage_dir,
        interval_sec=args.interval,
        max_time_sec=args.max_time,
        fuzz_args=args.fuzz_args,
        api_name=args.api
    )

    # Build only mode
    if args.build_only:
        try:
            runner.build()
            print("Build completed. Exiting as requested.")
            sys.exit(0)
        except Exception as e:
            print(f"Build failed: {e}")
            sys.exit(1)
    # Trap SIGINT/SIGTERM for graceful shutdown between intervals
    def _handle_signal(signum: int, frame: Optional[FrameType]) -> None:
        print(f"\nReceived signal {signum}, will stop after current interval...")
        runner.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        runner.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
