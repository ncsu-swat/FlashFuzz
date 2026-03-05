# FlashFuzz — Artifact Evaluation

FlashFuzz is a framework that employs coverage-guided fuzzing to test Deep Learning APIs at scale. This document provides instructions for reproducing the experiments and figures from the paper.

## Table of Contents

- [Requirements](#requirements)
- [Kick-the-Tires (Short Run, ~30 min)](#kick-the-tires-short-run-30-min)
- [Full Evaluation](#full-evaluation)
  - [E1: Coverage Comparison (RQ1)](#e1-coverage-comparison-rq1)
  - [E2: Ablation Study (RQ2)](#e2-ablation-study-rq2)
  - [E3: Input Validity (RQ3)](#e3-input-validity-rq3)
- [Reproducing Figures](#reproducing-figures)
- [Project Structure](#project-structure)

---

## Requirements

### Hardware

| Resource | Minimum | Recommended |
|---|---|---|
| CPU cores | 8 | 50+ (for `--num_parallel 50`) |
| RAM | 16 GB | 64 GB+ |
| Disk | 50 GB free | 200 GB+ (multiple Docker images) |
| GPU | Not required | NVIDIA GPU + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional) |

### Software

- **Docker** — Required for running all experiments. Each experiment runs inside a Docker container with the DL library compiled from source.
- **[Pixi](https://pixi.sh)** (recommended) — Package manager used to set up the Python environment. Install it with:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
  Then activate the environment:
  ```bash
  pixi install
  pixi shell
  ```
  Alternatively, install the Python dependencies manually: Python 3.12, `tqdm`, `bs4`, `regex`.

### Building Docker Images

Build the Docker images required for experiments. This step takes several hours due to compiling TensorFlow/PyTorch from source with coverage instrumentation.

```bash
# Build all images (TF 2.16, TF 2.19, PyTorch 2.2, PyTorch 2.7)
bash build_docker.sh
```

To build only the images needed for a specific experiment, run the relevant `docker build` commands from `build_docker.sh` individually.

---

## Kick-the-Tires (Short Run, ~30 min)

This section verifies that the artifact works end-to-end. We fuzz a small subset of PyTorch APIs with a short time budget.

### Step 1: Build the PyTorch 2.2 fuzz image

```bash
bash build_docker_kicktires.sh
```

### Step 2: Check that test harnesses compile

```bash
python3 -u run.py --dll torch --version 2.2 --mode fuzz --check_valid
```

Expected output (numbers may vary slightly):
```
Build Summary: Build status: XXX/1164 PyTorch APIs built successfully.
```

### Step 3: Fuzz 5 APIs with a 60-second budget

```bash
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 60 --apis torch.abs torch.add torch.argmax torch.concat torch.matmul
```

Results are stored in `_fuzz_result/`. Each API directory contains:
- `execution.log` — stdout from the fuzzing run
- `fuzz-0.log` — libFuzzer log with coverage and execution stats
- `artifacts/` — crash-triggering inputs (if any)

### Step 4: (Optional) Short coverage measurement

Build the coverage image and run coverage collection on the same APIs:

```bash
docker build -t ncsuswat/flashfuzz:torch2.2-cov -f docker/torch-2.2-cov.Dockerfile .
python3 -u run.py --dll torch --version 2.2 --mode cov --time_budget 60 --itv 30 --apis torch.abs torch.add torch.argmax torch.concat torch.matmul
```

Results are stored in `_cov_result/`.

---

## Full Evaluation

The full evaluation reproduces the main experiments from the paper. Each experiment fuzzes hundreds of APIs with a 600-second (10-minute) time budget per API. With `--num_parallel 50`, the approximate wall-clock times are:

| Experiment | Approx. Time |
|---|---|
| FlashFuzz fuzzing (TF) | ~2 hours |
| FlashFuzz fuzzing (PyTorch) | ~4 hours |
| Coverage collection (per tool per library) | ~3-5 hours |
| Ablation study (4 variants x 2 libraries) | ~5-6 hours |

### Common Flags

| Flag | Description |
|---|---|
| `--dll` | Target library: `tf` or `torch` |
| `--version` | Library version (e.g., `2.16`, `2.2`) |
| `--mode` | `fuzz` (fuzzing) or `cov` (coverage measurement) |
| `--time_budget` | Per-API time budget in seconds (default: 180) |
| `--num_parallel` | Number of parallel Docker containers (default: 1) |
| `--vs` | Baseline name (e.g., `titanfuzz`, `pathfinder`, `acetest`) |
| `--itv` | Coverage collection interval in seconds (default: 60) |
| `--slurm` | Pin Docker CPUs to SLURM-allocated cores |

### E1: Coverage Comparison (RQ1)

This experiment compares FlashFuzz against three baselines (TitanFuzz, PathFinder, ACETest) on TensorFlow 2.16 and PyTorch 2.2.

#### FlashFuzz (ours)

```bash
# TensorFlow 2.16
python3 -u run.py --dll tf --version 2.16 --mode fuzz --time_budget 600 --num_parallel 50
python3 -u run.py --dll tf --version 2.16 --mode cov  --time_budget 600 --num_parallel 50 --itv 60

# PyTorch 2.2
python3 -u run.py --dll torch --version 2.2 --mode fuzz --time_budget 600 --num_parallel 50
python3 -u run.py --dll torch --version 2.2 --mode cov  --time_budget 600 --num_parallel 50 --itv 60
```

#### Baselines

Each baseline uses `--vs <name>` to select the corresponding API list and test harnesses:

```bash
# TitanFuzz
python3 -u run.py --dll tf    --version 2.16 --mode fuzz --time_budget 600 --num_parallel 50 --vs titanfuzz
python3 -u run.py --dll tf    --version 2.16 --mode cov  --time_budget 600 --num_parallel 50 --vs titanfuzz --itv 60
python3 -u run.py --dll torch --version 2.2  --mode fuzz --time_budget 600 --num_parallel 50 --vs titanfuzz
python3 -u run.py --dll torch --version 2.2  --mode cov  --time_budget 600 --num_parallel 50 --vs titanfuzz --itv 60

# PathFinder
python3 -u run.py --dll tf    --version 2.16 --mode fuzz --time_budget 600 --num_parallel 50 --vs pathfinder
python3 -u run.py --dll tf    --version 2.16 --mode cov  --time_budget 600 --num_parallel 50 --vs pathfinder --itv 60
python3 -u run.py --dll torch --version 2.2  --mode fuzz --time_budget 600 --num_parallel 50 --vs pathfinder
python3 -u run.py --dll torch --version 2.2  --mode cov  --time_budget 600 --num_parallel 50 --vs pathfinder --itv 60

# ACETest
python3 -u run.py --dll tf    --version 2.16 --mode fuzz --time_budget 600 --num_parallel 50 --vs acetest
python3 -u run.py --dll tf    --version 2.16 --mode cov  --time_budget 600 --num_parallel 50 --vs acetest --itv 60
python3 -u run.py --dll torch --version 2.2  --mode fuzz --time_budget 600 --num_parallel 50 --vs acetest
python3 -u run.py --dll torch --version 2.2  --mode cov  --time_budget 600 --num_parallel 50 --vs acetest --itv 60
```

Coverage results are stored in `_cov_result/<dll><ver>-cov-<budget>s[-<baseline>]/all/`. Each interval directory (e.g., `0-60/`, `60-120/`, ...) contains a `.txt` file with branch coverage counts.

### E2: Ablation Study (RQ2)

The ablation study evaluates four variants of test harness generation by swapping the harness directory before rebuilding Docker images. The four variants are:

| Variant | Description |
|---|---|
| `original` | Full FlashFuzz (with helper functions + documentation) |
| `no_helper` | Without helper functions |
| `no_doc` | Without API documentation |
| `no_helper_no_doc` | Without helper functions or documentation |

Run the ablation scripts:

```bash
# TensorFlow ablation
bash abb_tf.sh

# PyTorch ablation
bash abb_torch.sh
```

Each variant produces its own `_fuzz_result_<variant>/` and `_cov_result_<variant>/` directories.

### E3: Input Validity (RQ3)

After fuzzing completes, use the helper scripts in `tools/` to aggregate results.

#### Collect validity statistics

```bash
python3 tools/collect_fuzz_stats.py --base _fuzz_result/torch2.2-fuzz-600s
```

This scans all `fuzz-*.log` files under the result directory and produces:
- Per-API `stat.txt` files with rounds, invalid count, valid count, and validity ratio
- `stats.csv` — aggregate CSV of all APIs
- `summary.txt` — overall summary

#### Collect crash reports

```bash
python3 tools/collect_fuzz_crashes.py _fuzz_result/tf2.19-fuzz-600s reports
```

This parses fuzz logs for crash info and writes a JSON report and Markdown summary under `reports/`.

#### Aggregate exception deciles (for validity plots)

```bash
python3 tools/aggregate_exception_deciles.py _fuzz_result/torch2.2-fuzz-600s
```

This computes per-decile exception rates across all fuzz logs, which feeds into the validity plots in `plots/validity.ipynb`.

> **Note:** The validity tools may produce inaccurate results for the TensorFlow ablation study. The TensorFlow fuzz logs use different error markers across harness variants, which can cause the heuristic-based counters to under- or over-count invalid inputs. PyTorch results are not affected.

---

## Reproducing Figures

The `plots/` directory contains Jupyter notebooks that generate the paper's figures from the collected data.

The notebooks have their own Pixi environment under `plots/`. Set it up before running:

```bash
cd plots
pixi install
pixi shell
```

This installs Python, Jupyter, and matplotlib. Alternatively, install them manually (`pip install jupyter matplotlib`).

### Coverage Plots (Figures in RQ1)

```bash
jupyter notebook coverage.ipynb
```

This notebook generates coverage-over-time comparison plots for:
- TensorFlow: FlashFuzz vs PathFinder, ACETest, TitanFuzz
- PyTorch: FlashFuzz vs PathFinder, ACETest, TitanFuzz

Output plots are saved as PNG and PDF in `plots/`.

To use your own experimental data, update the coverage arrays (e.g., `flashfuzz_tf`, `pathfinder_tf`) in the notebook cells with the branch coverage numbers from your `_cov_result/` directories.

### Validity Plots (Figures in RQ3)

```bash
cd plots
jupyter notebook validity.ipynb
```

This notebook generates per-decile exception rate plots for TensorFlow and PyTorch. Output is saved as `tf_decile_exception_rate.pdf` and `torch_decile_exception_rate.pdf`.

---

## Project Structure

```
FlashFuzz/
  run.py                  # Main entry point for fuzzing and coverage
  expmanager.py           # Experiment and scheduler classes
  build_docker.sh         # Build all Docker images
  docker/                 # Dockerfiles for TF/PyTorch (base, fuzz, cov)
  testharness/            # Test harnesses (C++ fuzz targets)
  testharness_generation/ # Harness generation tooling
  api_list/               # Per-library, per-tool API lists
  baselines/              # Baseline Dockerfiles and configs
  ablation/               # Ablation study harness variants
  plots/                  # Jupyter notebooks for figure reproduction
  tools/                  # Post-processing scripts (stats, crashes, deciles)
  scripts/                # Helper scripts (coverage, merge, etc.)
  reports/                # Bug reports
  paper/                  # Paper PDF
```
