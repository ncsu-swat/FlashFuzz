#!/usr/bin/env python3
"""
Script to run fuzzing with coverage instrumentation and collect .profraw files every 60 seconds.
This allows tracking coverage evolution over time.
"""

import os
import sys
import time
import shutil
import subprocess
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Sequence, Dict, Protocol, runtime_checkable, cast
from subprocess import Popen
from types import FrameType

class CoverageFuzzer:
    # Class-level type annotations for attributes
    corpus_dir: str
    profraw_dir: str
    fuzzing_process: Optional[Popen[str]]
    copy_thread: Optional[threading.Thread]
    stop_copying: bool
    iteration: int
    start_time: Optional[float]

    def __init__(self, corpus_dir: str = "corpus_itv_60", profraw_dir: str = "coverage_data") -> None:
        self.corpus_dir = corpus_dir
        self.profraw_dir = profraw_dir
        self.fuzzing_process = None
        self.copy_thread = None
        self.stop_copying = False
        self.iteration = 0
        self.start_time = None
        
        # Create coverage data directory
        Path(self.profraw_dir).mkdir(exist_ok=True)
        
        # Set environment variables for coverage
        os.environ['LLVM_PROFILE_FILE'] = 'fuzz-%p.profraw'
        
    def build_instrumented_binary(self) -> bool:
        """Build the fuzzer with coverage instrumentation"""
        print("Building instrumented fuzzer binary...")
        
        build_cmd: List[str] = [
            'clang++', 'fuzz.cpp',
            '-std=c++17',
            '-g',
            '-O0',
            '-fsanitize=fuzzer',
            '-fprofile-instr-generate',
            '-fcoverage-mapping',
            '-I', '/root/tensorflow',
            '-I', '/root/tensorflow/bazel-tensorflow',
            '-I', '/root/tensorflow/bazel-bin',
            '-I', '/root/tensorflow/bazel-tensorflow/external/com_google_absl',
            '-I', '/root/tensorflow/bazel-tensorflow/external/com_google_protobuf/src',
            '-I', '/root/tensorflow/bazel-tensorflow/external/eigen_archive',
            '-I', '/root/tensorflow/bazel-tensorflow/external/local_tsl',
            '-I', '/root/tensorflow/bazel-bin/external/local_tsl',
            '-I', '/root/tensorflow/bazel-tensorflow/external/nsync/public',
            '-I', '/root/tensorflow/bazel-tensorflow/external',
            '-L', '/root/tensorflow/bazel-bin/tensorflow',
            '-Wl,-rpath,$ORIGIN',
            '-ltensorflow_cc',
            '-ltensorflow_framework',
            '-lpthread',
            '-o', 'fuzz'
        ]
        
        try:
            subprocess.run(build_cmd, capture_output=True, text=True, check=True)
            print("✓ Binary built successfully with coverage instrumentation")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Build failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    def copy_profraw_files(self) -> None:
        """Copy .profraw files every 60 seconds in a separate thread"""
        while not self.stop_copying:
            time.sleep(60)  # Wait 60 seconds
            
            if self.stop_copying:
                break
                
            self.iteration += 1
            
            # Create interval naming: 0-60, 60-120, 120-180, etc.
            interval_start = (self.iteration - 1) * 60
            interval_end = self.iteration * 60
            interval_name = f"{interval_start}-{interval_end}"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Find all .profraw files in current directory
            profraw_files: List[Path] = list(Path('.').glob('*.profraw'))
            
            if profraw_files:
                # Create timestamped directory with interval naming
                timestamped_dir: Path = Path(self.profraw_dir) / f"{interval_name}_{timestamp}"
                timestamped_dir.mkdir(exist_ok=True)
                
                print(f"\n[{timestamp}] Interval {interval_name}s: Copying {len(profraw_files)} .profraw files to {timestamped_dir}")
                
                for profraw_file in profraw_files:
                    try:
                        shutil.copy2(profraw_file, timestamped_dir)
                        print(f"  ✓ Copied {profraw_file}")
                    except Exception as e:
                        print(f"  ✗ Failed to copy {profraw_file}: {e}")
                
                # Also create a merged profdata file for this interval
                self.create_profdata(timestamped_dir, interval_name)
            else:
                print(f"\n[{timestamp}] Interval {interval_name}s: No .profraw files found")
    
    def create_profdata(self, profraw_dir: Path, interval_name: str) -> None:
        """Create a merged .profdata file from .profraw files"""
        profraw_files: List[Path] = list(profraw_dir.glob('*.profraw'))
        if not profraw_files:
            return
            
        profdata_file: Path = profraw_dir / f"merged_{interval_name}.profdata"
        
        # Use llvm-profdata to merge the raw profiles
        merge_cmd: List[str] = ['llvm-profdata', 'merge', '-sparse'] + [str(f) for f in profraw_files] + ['-o', str(profdata_file)]
        
        try:
            subprocess.run(merge_cmd, capture_output=True, check=True)
            print(f"  ✓ Created merged profdata: {profdata_file}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to create profdata: {e}")
        except FileNotFoundError:
            print(f"  ✗ llvm-profdata not found, skipping profdata creation")
    
    def run_fuzzing(self, max_time: Optional[int] = None, additional_args: Optional[Sequence[str]] = None) -> None:
        """Run the fuzzing process"""
        print(f"Starting fuzzing process...")
        
        # Set up library path
        env: Dict[str, str] = os.environ.copy()
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        if current_ld_path:
            env['LD_LIBRARY_PATH'] = f"{os.getcwd()}:{current_ld_path}"
        else:
            env['LD_LIBRARY_PATH'] = os.getcwd()
        
        # Build fuzzer command
        fuzz_cmd: List[str] = ['./fuzz']
        
        # Add corpus directory if it exists
        if os.path.exists(self.corpus_dir):
            fuzz_cmd.append(self.corpus_dir)
        
        # Add additional arguments
        if additional_args:
            fuzz_cmd.extend(additional_args)
        
        # Add max time if specified
        if max_time:
            fuzz_cmd.extend([f'-max_total_time={max_time}'])
        
        print(f"Command: {' '.join(fuzz_cmd)}")
        print(f"Environment LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH')}")
        
        try:
            self.fuzzing_process = subprocess.Popen(
                fuzz_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start the coverage copying thread
            self.copy_thread = threading.Thread(target=self.copy_profraw_files)
            self.copy_thread.daemon = True
            self.start_time = time.time()  # Record start time for interval calculations
            self.copy_thread.start()
            
            print("✓ Fuzzing started. Press Ctrl+C to stop.")
            print("✓ Coverage files will be copied every 60 seconds.")
            print("-" * 60)
            
            # Stream output in real-time
            assert self.fuzzing_process.stdout is not None
            for line in iter(self.fuzzing_process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
            
            return_code = self.fuzzing_process.wait()
            print(f"\nFuzzing process ended with return code: {return_code}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.stop_fuzzing()
        except Exception as e:
            print(f"Error running fuzzing: {e}")
    
    def stop_fuzzing(self) -> None:
        """Stop the fuzzing process and cleanup"""
        print("Stopping fuzzing...")
        
        self.stop_copying = True
        
        if self.fuzzing_process and self.fuzzing_process.poll() is None:
            print("Terminating fuzzing process...")
            self.fuzzing_process.terminate()
            try:
                self.fuzzing_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing fuzzing process...")
                self.fuzzing_process.kill()
                self.fuzzing_process.wait()
        
        if self.copy_thread and self.copy_thread.is_alive():
            print("Waiting for coverage copying thread to finish...")
            self.copy_thread.join(timeout=5)
        
        # Final copy of any remaining .profraw files
        self.final_profraw_copy()
        
        print("✓ Cleanup completed")
    
    def final_profraw_copy(self) -> None:
        """Final copy of .profraw files when stopping"""
        profraw_files: List[Path] = list(Path('.').glob('*.profraw'))
        if profraw_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Calculate final interval
            if self.start_time:
                elapsed_seconds = int(time.time() - self.start_time)
                final_interval = f"final-{elapsed_seconds}s"
            else:
                final_interval = "final"
                
            final_dir: Path = Path(self.profraw_dir) / f"{final_interval}_{timestamp}"
            final_dir.mkdir(exist_ok=True)
            
            print(f"Final copy ({final_interval}): {len(profraw_files)} .profraw files to {final_dir}")
            
            for profraw_file in profraw_files:
                try:
                    shutil.copy2(profraw_file, final_dir)
                except Exception as e:
                    print(f"Failed to copy {profraw_file}: {e}")
            
            self.create_profdata(final_dir, final_interval)


def main() -> None:
    import argparse

    @runtime_checkable
    class _Args(Protocol):
        corpus: str
        coverage_dir: str
        max_time: Optional[int]
        build_only: bool
        fuzz_args: Optional[Sequence[str]]
    
    parser = argparse.ArgumentParser(description='Run fuzzing with coverage tracking')
    parser.add_argument('--corpus', default='corpus_itv_60', help='Corpus directory')
    parser.add_argument('--coverage-dir', default='coverage_data', help='Coverage data output directory')
    parser.add_argument('--max-time', type=int, help='Maximum fuzzing time in seconds')
    parser.add_argument('--build-only', action='store_true', help='Only build the binary, do not run fuzzing')
    parser.add_argument('--fuzz-args', nargs='*', help='Additional arguments to pass to the fuzzer')
    
    args = cast(_Args, parser.parse_args())
    
    fuzzer: CoverageFuzzer = CoverageFuzzer(corpus_dir=args.corpus, profraw_dir=args.coverage_dir)
    
    # Build the instrumented binary
    if not fuzzer.build_instrumented_binary():
        print("Failed to build binary, exiting.")
        sys.exit(1)
    
    if args.build_only:
        print("Build completed. Exiting as requested.")
        sys.exit(0)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        print(f"\nReceived signal {signum}")
        fuzzer.stop_fuzzing()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run fuzzing
    try:
        fuzzer.run_fuzzing(max_time=args.max_time, additional_args=args.fuzz_args)
    except Exception as e:
        print(f"Error: {e}")
        fuzzer.stop_fuzzing()
        sys.exit(1)

if __name__ == "__main__":
    main()
