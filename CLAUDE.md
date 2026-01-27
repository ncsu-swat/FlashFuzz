# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashFuzz is a coverage-guided fuzzing framework for testing Deep Learning APIs (TensorFlow and PyTorch) at scale. It orchestrates fuzzing experiments in Docker containers, manages parallel execution, and collects coverage data.

## Common Commands

### Setup
```bash
bash build_docker.sh  # Build all Docker images
```

### Running Experiments

```bash
# Fuzz TensorFlow APIs (default 180s per API)
python3 -u run.py --dll tf --version 2.16 --mode fuzz

# Fuzz with parallelism and custom time budget
python3 -u run.py --dll tf --version 2.16 --mode fuzz --num_parallel 40 --time_budget 300

# Check test harness validity
python3 -u run.py --dll tf --version 2.16 --mode fuzz --check_valid

# Collect coverage
python3 -u run.py --dll tf --version 2.16 --mode cov

# PyTorch equivalent
python3 -u run.py --dll torch --version 2.2 --mode fuzz

# Run with baseline comparison (e.g., pathfinder, titanfuzz, acetest)
python3 -u run.py --dll tf --version 2.16 --mode fuzz --vs pathfinder

# GPU mode (requires nvidia container toolkit)
python3 -u run.py --dll tf --version 2.19 --mode fuzz --gpu

# SLURM cluster mode
python3 -u run.py --dll tf --version 2.16 --mode fuzz --slurm
```

## Architecture

### Entry Points
- `run.py` - Main CLI entry point, parses arguments and creates experiments
- `expmanager.py` - Core experiment orchestration: `Experiment` class manages Docker lifecycle, `Scheduler` class handles parallel execution

### Test Harnesses
- `testharness/tf_cpu/` - Pre-generated C++ libFuzzer harnesses for TensorFlow APIs (one directory per `tf.raw_ops.*` API)
- `testharness/torch_cpu/` - Pre-generated C++ libFuzzer harnesses for PyTorch APIs (one directory per `torch.*` API)
- Each harness directory contains `fuzz.cpp` (TF) or `main.cpp` (PyTorch) with `LLVMFuzzerTestOneInput`

### Scripts (run inside Docker containers)
- `scripts/build_test_harness.py` - Builds all harnesses in parallel, checks build status
- `scripts/coverage_fuzzing.py` - Runs fuzzing with coverage instrumentation
- `scripts/merge_profraw.py` - Merges LLVM profraw coverage files
- `scripts/get_coverage_results.py` - Extracts coverage metrics from profdata

### Docker Images
Docker images are tagged as `ncsuswat/flashfuzz:<dll><version>-<mode>[-gpu]`:
- Base images: `tf2.16-base`, `tf2.19-base`, `torch2.2-base`, `torch2.7-base`
- Fuzz images: `tf2.16-fuzz`, `tf2.19-fuzz`, `torch2.2-fuzz`, `torch2.7-fuzz`
- Coverage images: `tf2.16-cov`, `torch2.2-cov`

### API Lists
- `api_list/` contains text files listing APIs to fuzz for each configuration
- Naming: `<dll><version>-<tool>.txt` (e.g., `tf2.16-flashfuzz.txt`, `torch2.2-pathfinder.txt`)

### Results
- Fuzzing results: `_fuzz_result/<dll><version>-fuzz-<time>s[-<baseline>]/<api>/`
- Coverage results: `_cov_result/<dll><version>-cov-<time>s[-<baseline>]/`

## Key Classes (expmanager.py)

- `Experiment` - Represents a single API fuzzing run; handles Docker container lifecycle (create, start, execute, copy results, stop, remove)
- `Scheduler` - Manages parallel execution of experiments using ThreadPoolExecutor; includes watchdog to kill hung containers at 2x time budget

## Ablation Studies
- `ablation/torch/` - LLM-driven ablation pipeline for generating PyTorch fuzz targets with variants (original, no_doc, no_helper)
- `ablation/tf/` - Similar ablation for TensorFlow

## Test Harness Generation
- `testharness_generation/torch_cpu/` - LLM-based generation of PyTorch fuzz harnesses using `llm_gptoss.py`


## Fixing Testharness for PyTorch APIs

### Automated Fixing
Use `fix_harness.py` to automatically fix test harnesses using Claude:
```bash
# Check status
python3 fix_harness.py --status

# Fix a single API
python3 fix_harness.py --api torch.abs

# Fix multiple APIs
python3 fix_harness.py --limit 10

# Dry run (analyze only)
python3 fix_harness.py --dry-run --api torch.add
```

### Manual Fixing Workflow
1. Navigate to the specific API directory in `testharness/torch_cpu/`
2. Run the corresponding docker image, and copy the testharness to the specific path
3. Execute in the docker
4. First check if the API exist or used correctly. If not, try to fix the test harness or leave a note "API not found in backend" in the api.json
5. Second, try to compile the testharness, if there is compilation error, try to fix it.
6. If both above steps are successful, run the testharness with one minute, and check the log. If it crashes or have concernings, try to fix the testharness or make it stronger.
7. Finally, document all the changes in the api.json file

## Manual Docker Debugging

### Directory Structure Inside Docker
- **PyTorch**: `/root/fuzz/` contains all API harnesses
- **TensorFlow**: `/root/tensorflow/fuzz/` contains all API harnesses

Each API directory contains:
- `main.cpp` (PyTorch) or `fuzz.cpp` (TensorFlow) - the harness source
- `fuzzer_utils.cpp`, `fuzzer_utils.h` - utility functions for tensor creation
- `build.sh` - compilation script
- `fuzz.sh` - fuzzing execution script
- `fuzz` - compiled binary (after build)
- `build.log` - compilation output
- `fuzz-0.log` - fuzzing runtime log
- `corpus/` - generated test inputs
- `artifacts/` - crash-inducing inputs

### Start Interactive Docker Session
```bash
# PyTorch 2.7
docker run -it --rm \
  -v $(pwd)/testharness/torch_cpu:/root/testharness:rw \
  ncsuswat/flashfuzz:torch2.7-fuzz \
  bash

# TensorFlow 2.16
docker run -it --rm \
  -v $(pwd)/testharness/tf_cpu:/root/testharness:rw \
  ncsuswat/flashfuzz:tf2.16-fuzz \
  bash
```

### Copy Harness Files to Container Working Directory
Inside Docker:
```bash
# Copy your harness to the working directory (PyTorch example)
API_NAME="torch.abs"
cp /root/testharness/$API_NAME/main.cpp /root/fuzz/$API_NAME/main.cpp
cd /root/fuzz/$API_NAME
```

### Compile Test Harness
Inside Docker, in the API directory:
```bash
# Run the build script (logs to build.log)
bash build.sh

# Or compile manually for PyTorch:
clang++ -fsanitize=fuzzer \
    -fno-omit-frame-pointer \
    -O0 -g \
    -I/root/pytorch/build-fuzz/include \
    -I/root/pytorch/aten/src \
    -I/root/pytorch/c10/core \
    -I/root/pytorch \
    -I/root/pytorch/build-fuzz \
    -I/root/pytorch/build-fuzz/aten/src \
    -I/root/pytorch/torch/csrc/api/include \
    -std=c++17 \
    main.cpp fuzzer_utils.cpp \
    -Wl,-rpath,/root/pytorch/build-fuzz/lib \
    -L/root/pytorch/build-fuzz/lib \
    -ltorch -ltorch_cpu -lc10 \
    -o fuzz

# Check compilation log if build fails
cat build.log
```

### Run Fuzzing
Inside Docker, in the API directory:
```bash
# Run for 60 seconds using fuzz.sh
bash fuzz.sh

# Or run manually with custom settings
./fuzz ./corpus \
    -max_total_time=60 \
    -max_len=128 \
    -jobs=1 \
    -workers=2 \
    -timeout=2 \
    -rss_limit_mb=2048 \
    -print_final_stats=1

# Run with specific input (for debugging crashes)
./fuzz path/to/crash_input
```

### Check Runtime Logs
```bash
# View fuzzing log
cat fuzz-0.log

# Check for crashes
grep -E "SUMMARY|SEGV|heap-buffer-overflow|stack-buffer-overflow" fuzz-0.log

# Check for exceptions (may indicate invalid inputs)
grep "Exception caught:" fuzz-0.log | wc -l

# View crash artifacts
ls -la artifacts/
```

### Copy Results Back to Host
Inside Docker or from host:
```bash
# From host - copy modified harness back
docker cp <container_id>:/root/fuzz/torch.abs/main.cpp testharness/torch_cpu/torch.abs/

# Or if using volume mount, changes are already synced
```

### Using expmanager.py for Debugging
You can also use the `Experiment` class programmatically:
```python
from expmanager import Experiment

# Create experiment
exp = Experiment(
    dll="torch",
    mode="fuzz",
    ver="2.7",
    api="torch.abs",
    time_budget=60,
    debug=True  # Keeps intermediate files
)

# Start container
exp.start_docker_container()

# Execute commands
exp.execute_command("cd /root/fuzz/torch.abs && bash build.sh")
exp.execute_command("cd /root/fuzz/torch.abs && cat build.log")

# Copy files
exp.copy_files_to_container("testharness/torch_cpu/torch.abs/main.cpp", "/root/fuzz/torch.abs/main.cpp")
exp.copy_results_from_container("/root/fuzz/torch.abs/build.log", "./debug/")

# Cleanup
exp.stop_docker_container()
exp.remove_docker_container()
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| API not found | `error: no member named 'xxx' in namespace 'torch'` | Check PyTorch C++ API docs; API may be Python-only |
| Missing includes | `fatal error: 'xxx.h' file not found` | Add appropriate `-I` flags to build command |
| Linker errors | `undefined reference to 'xxx'` | Add appropriate `-l` library flags |
| Tensor shape mismatch | Runtime exception about shapes | Fix tensor creation in harness |
| Memory issues | `rss_limit_mb exceeded` | Reduce `max_len` or increase `rss_limit_mb` |
| Timeout | Fuzzer hangs | Check for infinite loops; add timeout guards |
