# FlashFuzz Torch Ablation

This folder contains an LLM-driven ablation pipeline that generates C++ libFuzzer targets for PyTorch operators and modules. It lets you study how two factors influence fuzzer quality:

- Access to API docs in the prompt (on/off)
- Access to a reusable helper skeleton (`fuzzer_utils.*`) in the generated target (on/off)

The generator produces three variants per API:

- `original`: Docs provided to the LLM + helper skeleton available
- `no_doc`: Docs withheld + helper skeleton available
- `no_helper`: Docs provided + helper skeleton removed (target must be self-contained)

Use this to generate and fuzz many targets quickly and compare results across variants.

## Layout

- `llm.py`: Orchestrates prompt construction, LLM calls (Anthropic), and file layout per variant
- `api.txt`: List of PyTorch APIs to target (one per line)
- `pixi.toml`: Minimal Python environment (Python 3.11, `anthropic`, `torch==2.2.0`)
- `torch_cpu_helper/`: Helper skeleton copied into generated targets when helpers are enabled
  - `main.cpp`: Entry that includes `fuzzer_utils.h` and defines `LLVMFuzzerTestOneInput`
  - `fuzzer_utils.h/.cpp`: Utilities to parse tensors from fuzzer bytes, logging, comparisons
  - `build.sh`: Example build script using Clang + libFuzzer and a locally built PyTorch C++
  - `fuzz.sh`: Example fuzz invocation wrapper (libFuzzer flags)
  - `random_seed.py`: Generates a seed corpus under `corpus/`
  - `copy.py`: Utility to replicate helper files across many per-API directories (optional)

## Requirements

- Linux, Clang with libFuzzer support (`clang++ -fsanitize=fuzzer`) and C++17
- A PyTorch C++ build and headers available locally (see Build section)
- Python 3.11 with packages: `anthropic (>=0.66,<0.67)`, `torch==2.2.0`
- Anthropic API key: export `ANTHROPIC_API_KEY` in your shell

Note: The helper `build.sh` references a PyTorch source/build at `/root/pytorch/...`. You will very likely need to edit include and library paths for your environment.

## Environment Setup

Using pixi (recommended for Python deps):

1) Install pixi (if needed) and then:
   - `cd ablation/torch`
   - `pixi install`
   - `pixi shell`

2) Provide your Anthropic key:
   - `export ANTHROPIC_API_KEY="<your-key>"`
   - Optional overrides: `ANTHROPIC_MODEL` (default `claude-3-5-sonnet-latest`), `LLM_MAX_TOKENS` (default `4000`), `LLM_TEMPERATURE` (default `0.0`)

Without pixi (via venv):

- `python -m venv .venv && source .venv/bin/activate`
- `pip install 'anthropic>=0.66,<0.67' torch==2.2.0`
- `export ANTHROPIC_API_KEY=...`

Security note: `.env` is not automatically consumed by `llm.py`. Do not commit secrets. Prefer exporting `ANTHROPIC_API_KEY` in your shell/session manager.

## Generate Targets

1) Edit `ablation/torch/api.txt` to include the APIs you want (one per line). Examples in the repo target a mix of `torch.*` symbols.

2) Run the generator:

- With all variants into `out/`:
  - `python ablation/torch/llm.py --api-file ablation/torch/api.txt --out-dir out`

- Restrict to specific variants (names: `original`, `no_doc`, `no_helper`):
  - `python ablation/torch/llm.py --api-file ablation/torch/api.txt --out-dir out --variants original no_doc`

- Overwrite existing outputs:
  - `python ablation/torch/llm.py --api-file ablation/torch/api.txt --out-dir out --overwrite`

The script creates:

- `out/original/<api>/main.cpp` (+ helper files)
- `out/no_doc/<api>/main.cpp` (+ helper files)
- `out/no_helper/<api>/main.cpp` (no helpers; self-contained)

Internally, `llm.py` inspects Python-level docstrings for each API using your installed `torch` version to optionally include them in the prompt.

## Build

Each generated directory that includes helpers contains a `build.sh` you can adapt. It assumes:

- Clang with libFuzzer is available
- You have a local PyTorch C++ build (includes and libs). The defaults point at `/root/pytorch/...` and CUDA includes; update these to your paths or remove CUDA includes if you’re on CPU-only.

Example (adjust paths, then run):

- `cd out/original/torch.asin`
- `bash build.sh`

If compilation succeeds, a `fuzz` binary will be created in the same directory. If you generated a `no_helper` target, you’re expected to compile the single `main.cpp` yourself with appropriate include/library paths for your environment.

## Fuzz

- Seed corpus: `python3 random_seed.py` (writes `corpus/seed*.bin`)
- Run the fuzzer:
  - `bash fuzz.sh`

Notes:

- `fuzz.sh` currently includes a placeholder for `-max_total_time`. Edit it to a numeric value (e.g., `600`) or modify the script to read from an env var, e.g. `-max_total_time=${TIME_BUDGET:-600}`.
- The helper utilities write logs like `error.log`, `error_inputs.log` and, when mismatches are detected by `compareTensors`, a `diff_inputs/` folder with inputs that caused differences.

## Prompt Variants and What They Change

- `original`: LLM sees Python docstrings for the API and can use `fuzzer_utils.*` in the C++ harness.
- `no_doc`: LLM does not see docstrings but can still use helper utilities.
- `no_helper`: LLM sees docstrings but must produce a fully self-contained `main.cpp` and must not include or reference `fuzzer_utils.*`.

All variants always receive a minimal `main.cpp` skeleton in the prompt for consistent structure.

## Tips and Troubleshooting

- PyTorch C++ paths: If you don’t keep PyTorch at `/root/pytorch/...`, update `torch_cpu_helper/build.sh` after generation (or adjust the template before generation) to include headers like `torch/csrc/api/include`, `aten/src`, and link against `libtorch`/`libtorch_cpu`/`libc10` from your build or a libtorch distribution.
- Version mismatches: The docstrings come from your installed Python `torch`. If you compile against a different PyTorch C++ version, symbols or behavior can differ. Align versions when possible.
- Missing libFuzzer: Ensure your Clang supports `-fsanitize=fuzzer`. On some distros this requires an LLVM tooling package.
- CUDA includes: Remove `-I/usr/local/cuda/include` from `build.sh` if not present in your system.
- `copy.py`: This helper expects directories named like `torch.*` at the current working directory. If you use `llm.py`’s default layout (`out/<variant>/<api>/`), run `copy.py` only if you modify it to match your directory structure; it is optional.
- Network/API limits: The generator calls Anthropic’s API once per API per variant. Large `api.txt` lists can hit rate limits; set smaller batches or add sleeps as needed.

## Related Orchestration

At repo root, `run.py` and the `*-torch.sh` shell wrappers orchestrate full experiments (fuzzing and coverage) across many APIs in Docker/SLURM settings. The ablation generator in this folder is standalone and focuses on templating and producing per-API fuzz targets; you can integrate generated targets into your broader pipelines as needed.

## Example End-to-End (single API)

- `export ANTHROPIC_API_KEY=...`
- `python ablation/torch/llm.py --api-file ablation/torch/api.txt --out-dir out --variants original`
- `cd out/original/torch.asin`
- Edit `build.sh` include/library paths to match your environment
- `bash build.sh`
- Edit `fuzz.sh` to set `-max_total_time=600`
- `python3 random_seed.py`
- `bash fuzz.sh`

---

Questions or hiccups? Open an issue or ping the maintainer. This doc focuses on the ablation generator; it won’t fix build env specifics for your machine, but it should make the moving pieces clear and predictable.

