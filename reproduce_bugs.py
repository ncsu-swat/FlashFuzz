#!/usr/bin/env python3
"""
Reproduce fuzzer-found bugs in the PyTorch Python frontend.

For each API with crash/timeout artifacts from C++ libFuzzer runs:
1. Replay the crash artifact in Docker to get crash output
2. Decode the binary artifact into tensor metadata
3. Use Claude CLI to generate a triage reproduce.py (hex-based)
4. Test the reproduction in Docker, iterate up to N times
5. If reproduced, generate a clean reproduce.py with concrete tensors
6. Generate a bug_report.md suitable for PyTorch GitHub issue submission
7. Save all outputs to _bug_reports/<version>/<api>/

Usage:
    python3 reproduce_bugs.py --crashes-only
    python3 reproduce_bugs.py --api torch.dot
    python3 reproduce_bugs.py --dry-run
    python3 reproduce_bugs.py --limit 10
"""

import os
import sys
import json
import struct
import subprocess
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configuration
DOCKER_IMAGE = "ncsuswat/flashfuzz:torch2.10-fuzz"
PYTORCH_VERSION = "2.10.0"
RESULT_DIR = Path("_fuzz_result/torch2.10-fuzz-43200s")
OUTPUT_DIR = Path("_bug_reports/torch2.10")
TESTHARNESS_DIR = Path("testharness/torch_cpu")
CLAUDE_TIMEOUT = 300  # seconds
DOCKER_TIMEOUT = 30  # seconds for replay
DOCKER_PYTHON_TIMEOUT = 120  # seconds for Python test (includes pip install)
MAX_ATTEMPTS = 3
INTER_API_DELAY = 10  # seconds between APIs to avoid overuse
INTER_ATTEMPT_DELAY = 5  # seconds between Claude attempts

# Dtype mapping matching fuzzer_utils.cpp supported_types
SUPPORTED_DTYPES = [
    "torch.float32", "torch.float64", "torch.float16", "torch.bfloat16",
    "torch.complex64", "torch.complex128",
    "torch.int8", "torch.uint8", "torch.int16", "torch.int32", "torch.int64",
    "torch.bool",
]

DTYPE_SIZES = {
    "torch.float32": 4, "torch.float64": 8, "torch.float16": 2, "torch.bfloat16": 2,
    "torch.complex64": 8, "torch.complex128": 16,
    "torch.int8": 1, "torch.uint8": 1, "torch.int16": 2, "torch.int32": 4,
    "torch.int64": 8, "torch.bool": 1,
}

MIN_RANK = 0
MAX_RANK = 4
MIN_DIM = 0
MAX_DIM = 16


# ---------------------------------------------------------------------------
# Artifact decoding
# ---------------------------------------------------------------------------

def decode_tensor(data: bytes, offset: int) -> tuple[dict, int]:
    """Decode one tensor from binary artifact data, matching fuzzer_utils.cpp."""
    if offset + 2 > len(data):
        raise ValueError(f"Not enough data at offset {offset}")

    dtype_selector = data[offset]; offset += 1
    dtype = SUPPORTED_DTYPES[dtype_selector % len(SUPPORTED_DTYPES)]
    dtype_size = DTYPE_SIZES[dtype]

    rank_byte = data[offset]; offset += 1
    rank = rank_byte % (MAX_RANK - MIN_RANK + 1) + MIN_RANK

    shape = []
    for _ in range(rank):
        if offset + 8 <= len(data):
            dim_raw = struct.unpack_from("<q", data, offset)[0]
            offset += 8
            shape.append(MIN_DIM + (abs(dim_raw) % (MAX_DIM - MIN_DIM + 1)))
        else:
            shape.append(MIN_DIM)
            offset = len(data)

    num_elements = 1
    for d in shape:
        num_elements *= d

    total_bytes_needed = num_elements * dtype_size
    bytes_available = max(0, len(data) - offset)
    bytes_to_copy = min(total_bytes_needed, bytes_available)
    tensor_bytes = data[offset:offset + bytes_to_copy]
    if len(tensor_bytes) < total_bytes_needed:
        tensor_bytes += b'\x00' * (total_bytes_needed - len(tensor_bytes))
    offset += bytes_to_copy

    return {
        "dtype": dtype,
        "dtype_selector": dtype_selector,
        "rank": rank,
        "shape": shape,
        "num_elements": num_elements,
        "tensor_bytes_hex": tensor_bytes[:64].hex(),
        "total_bytes": total_bytes_needed,
    }, offset


def decode_artifact(artifact_path: Path) -> dict:
    """Decode all tensors from a binary artifact file."""
    data = artifact_path.read_bytes()
    offset = 0
    tensors = []
    idx = 0
    while offset + 2 <= len(data) and idx < 5:
        try:
            info, offset = decode_tensor(data, offset)
            info["index"] = idx
            tensors.append(info)
            idx += 1
        except (ValueError, struct.error):
            break
    return {
        "file": artifact_path.name,
        "size": len(data),
        "hex": data.hex(),
        "tensors": tensors,
        "remaining_bytes_at_end": len(data) - offset,
    }


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def check_docker_image() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def run_docker_command(cmd: str, timeout: int = DOCKER_TIMEOUT) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", DOCKER_IMAGE, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, "Docker command timed out"
    except Exception as e:
        return -1, str(e)


def replay_crash_in_docker(api_name: str, artifact_path: Path) -> str:
    hex_data = artifact_path.read_bytes().hex()
    cmd = (
        f"cd /root/fuzz/{api_name} && "
        f"python3 -c \"import sys; sys.stdout.buffer.write(bytes.fromhex('{hex_data}'))\" "
        f"> /tmp/artifact && ./fuzz /tmp/artifact 2>&1 || true"
    )
    _, output = run_docker_command(cmd, timeout=DOCKER_TIMEOUT)
    return output


def test_reproduce_in_docker(reproduce_code: str) -> dict:
    """Run reproduce.py in Docker. Returns structured result dict.

    Keys:
        has_hard_crash: bool — segfault, ASAN, core dump, timeout/kill
        has_runtime_error: bool — a RuntimeError appeared in output
        runtime_error_message: str|None — first RuntimeError line
        exit_code: int|None — parsed from EXIT_CODE=N in output
        output: str — full stdout+stderr
    """
    cmd = f"""
pip3 install torch --break-system-packages --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null
cat > /tmp/reproduce.py << 'PYEOF'
{reproduce_code}
PYEOF
timeout 30 python3 /tmp/reproduce.py 2>&1
echo "EXIT_CODE=$?"
"""
    _, output = run_docker_command(cmd, timeout=DOCKER_PYTHON_TIMEOUT)

    # Parse exit code
    exit_code = None
    m = re.search(r"EXIT_CODE=(\d+)", output)
    if m:
        exit_code = int(m.group(1))

    # Check for hard crashes (always true positives)
    hard_crash_indicators = [
        "Segmentation fault", "SIGSEGV", "core dumped",
        "AddressSanitizer", "heap-buffer-overflow", "stack-buffer-overflow",
        "use-after-free", "double-free", "SIGABRT", "Aborted", "Killed",
        "MemoryError",
    ]
    has_hard_crash = False
    if exit_code in (124, 137):  # timeout / killed
        has_hard_crash = True
    for ind in hard_crash_indicators:
        if ind in output:
            has_hard_crash = True
            break

    # Check for RuntimeError separately (needs classification)
    has_runtime_error = "RuntimeError" in output
    runtime_error_message = None
    if has_runtime_error:
        for line in output.splitlines():
            if "RuntimeError" in line:
                runtime_error_message = line.strip()
                break

    return {
        "has_hard_crash": has_hard_crash,
        "has_runtime_error": has_runtime_error,
        "runtime_error_message": runtime_error_message,
        "exit_code": exit_code,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Claude CLI helpers
# ---------------------------------------------------------------------------

def run_claude(prompt: str, model: str = "sonnet") -> Optional[str]:
    """Call Claude CLI and return stdout."""
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=CLAUDE_TIMEOUT, env=env,
        )
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        print("    Claude CLI timed out")
        return None
    except FileNotFoundError:
        print("    ERROR: claude CLI not found")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from Claude's response."""
    for pattern in [r"```python\s*(.*?)\s*```", r"```\s*(import torch.*?)\s*```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len)
    if "import torch" in response:
        lines = response.split("\n")
        code_lines = []
        started = False
        for line in lines:
            if "import torch" in line:
                started = True
            if started:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)
    return None


def extract_markdown(response: str) -> Optional[str]:
    """Extract markdown from Claude's response, stripping outer fences."""
    # If response is wrapped in ```markdown ... ```, strip it
    m = re.match(r"^```(?:markdown)?\s*\n(.*)\n```\s*$", response, re.DOTALL)
    if m:
        return m.group(1)
    return response


def classify_runtime_error(
    api_name: str,
    reproduce_code: str,
    runtime_error_message: str,
    full_output: str,
    model: str = "sonnet",
) -> tuple[str, str]:
    """Classify a RuntimeError as true_positive, false_positive, or false_positive_repro_bug.

    Returns (classification, reason).
    """
    prompt = f"""You are classifying whether a RuntimeError from a PyTorch bug reproduction script
indicates a REAL bug or a FALSE POSITIVE.

API under test: {api_name}
PyTorch version: {PYTORCH_VERSION}

Reproduction script:
```python
{reproduce_code[:3000]}
```

RuntimeError message:
{runtime_error_message}

Full output:
```
{full_output[:2000]}
```

Classify this into EXACTLY ONE category:

1. **false_positive** — PyTorch is CORRECTLY rejecting invalid input. Examples:
   - "inconsistent tensor size" when tensors have mismatched dimensions
   - "out of range" or "out of bounds" for invalid indices
   - "expected ... but got ..." for wrong dtype/shape
   - "cannot be multiplied" for incompatible matrix shapes
   - "invalid argument" for unsupported parameter values
   - Any error message that says the input is wrong/invalid/unsupported
   This is NOT a bug — PyTorch is working correctly by rejecting bad input.

2. **false_positive_repro_bug** — The error is in the SETUP CODE, not in the target API call.
   The script crashes BEFORE reaching the API under test. Examples:
   - `.t()` called on a 3D tensor (only works on 2D)
   - `torch.reshape()` with incompatible shape in setup
   - Variable referenced before assignment
   - Import error or syntax error
   The API itself was never tested because the reproduction code is broken.

3. **true_positive** — This is an ACTUAL bug in PyTorch. Examples:
   - Error message is wrong or misleading for the given input
   - Error occurs with clearly valid inputs that should work
   - Internal PyTorch error (not input validation)
   - Inconsistent behavior compared to documentation
   - The error indicates memory corruption or internal state issues

Respond with EXACTLY this format (two lines only):
CLASSIFICATION: <one of: true_positive, false_positive, false_positive_repro_bug>
REASON: <one sentence explaining why>
"""
    response = run_claude(prompt, model)
    if not response:
        return "true_positive", "Classification failed, treating as potential bug"

    classification = "true_positive"  # default: err on the side of caution
    reason = "Could not parse classification response"

    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("CLASSIFICATION:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("false_positive", "false_positive_repro_bug", "true_positive"):
                classification = val
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return classification, reason


# ---------------------------------------------------------------------------
# Stage 1: Triage — generate reproduce.py that triggers the bug
# ---------------------------------------------------------------------------

def generate_triage_code(
    api_name: str,
    crash_output: str,
    harness_code: str,
    decoded_info: dict,
    artifact_type: str,
    previous_attempt: Optional[str] = None,
    previous_error: Optional[str] = None,
    model: str = "sonnet",
) -> Optional[str]:
    """Generate a triage reproduce.py (may use hex decoding, whatever works)."""
    tensor_lines = []
    for t in decoded_info.get("tensors", []):
        tensor_lines.append(
            f"  Tensor {t['index']}: dtype={t['dtype']}, shape={t['shape']}, "
            f"num_elements={t['num_elements']}"
        )
    tensor_str = "\n".join(tensor_lines) or "  (none decoded)"

    prompt = f"""You are reproducing a C++ fuzzer bug in the PyTorch Python frontend.

API: {api_name}
PyTorch version: {PYTORCH_VERSION}
Bug type: {"CRASH (memory error / segfault)" if artifact_type == "crash" else "TIMEOUT (hangs > 2s)"}

C++ harness:
```cpp
{harness_code[:4000]}
```

Crash replay output:
```
{crash_output[:3000]}
```

Decoded artifact tensors:
{tensor_str}

Raw artifact hex (first 256 chars): {decoded_info.get('hex', '')[:256]}
"""
    if previous_attempt and previous_error:
        prompt += f"""
PREVIOUS ATTEMPT (did NOT reproduce):
```python
{previous_attempt[:2000]}
```
Output:
```
{previous_error[:2000]}
```
Fix the reproduction to actually trigger the bug.
"""

    prompt += """
Write a minimal Python script that reproduces this bug.
- import torch, create tensor(s) with matching dtype/shape/data, call the API.
- A RuntimeError that correctly rejects invalid input (e.g. "inconsistent tensor size",
  "out of bounds", wrong dtype) is NOT a bug — it means PyTorch is working correctly.
- Real bugs are: segfaults, crashes, hangs, memory corruption, incorrect results,
  or errors that should NOT occur for the given inputs.
- Make sure your setup code is correct before calling the target API — e.g. don't call
  .t() on a 3D tensor, don't reshape to an incompatible shape, etc.
- For CRASH: trigger segfault or memory error. For TIMEOUT: trigger a hang.
- Output ONLY the Python code, no explanations.
"""

    response = run_claude(prompt, model)
    if not response:
        return None
    return extract_python_code(response)


# ---------------------------------------------------------------------------
# Stage 2: Clean — rewrite reproduce.py with concrete, readable tensors
# ---------------------------------------------------------------------------

def generate_clean_code(
    api_name: str,
    triage_code: str,
    test_output: str,
    model: str = "sonnet",
) -> Optional[str]:
    """Rewrite the triage code into clean, human-readable Python with concrete values."""
    prompt = f"""Rewrite this PyTorch bug reproduction into clean, minimal, easy-to-read code.

API: {api_name}
PyTorch version: {PYTORCH_VERSION}

Working triage code (triggers the bug):
```python
{triage_code}
```

Output when running (confirms the bug):
```
{test_output[:2000]}
```

Requirements for the rewritten code:
1. Use concrete tensor values inline — NO hex strings, NO bytes.fromhex(), NO struct.unpack.
   - Use torch.zeros(), torch.ones(), torch.tensor([...]), torch.randn(), etc.
   - Pick the SIMPLEST tensor values that still trigger the bug (e.g. zeros, small integers).
2. Use concrete API calls — NO dynamic dispatch, NO variable API names.
3. Keep it under 30 lines if possible.
4. Add a brief comment at the top: "# Bug: <one-line description of what goes wrong>"
5. The code MUST still trigger the same error when run.

Example of what GOOD output looks like:
```python
# Bug: torch.dot crashes with inconsistent tensor sizes
import torch

a = torch.zeros(0, dtype=torch.int16)
b = torch.zeros(1, dtype=torch.int16)
result = torch.dot(a, b)  # RuntimeError: inconsistent tensor size
```

Output ONLY the Python code.
"""
    response = run_claude(prompt, model)
    if not response:
        return None
    return extract_python_code(response)


# ---------------------------------------------------------------------------
# Stage 3: Bug report — markdown formatted for PyTorch GitHub
# ---------------------------------------------------------------------------

def generate_bug_report_md(
    api_name: str,
    clean_code: str,
    test_output: str,
    artifact_type: str,
    crash_output: str,
    has_hard_crash: bool = True,
    model: str = "sonnet",
) -> Optional[str]:
    """Generate a bug_report.md suitable for filing on PyTorch GitHub."""
    # For non-crash cases (RuntimeError only), add context about C++ vs Python
    non_crash_note = ""
    if not has_hard_crash:
        non_crash_note = """
NOTE: This bug was found via C++ libFuzzer and the Python reproduction shows a RuntimeError
rather than a hard crash. Include a "### Why this matters" section explaining:
1. The Python frontend validates inputs before passing to the C++ backend — so some C++ crashes
   manifest as RuntimeErrors in Python
2. The C++ libFuzzer bypasses Python validation and directly hits the C++ backend, which is why
   it crashes in C++ but raises a RuntimeError in Python
3. The underlying C++ issue may still represent a real bug (e.g. missing bounds check that could
   be exploited if reached through a different code path)
"""

    prompt = f"""Write a PyTorch GitHub issue bug report for this bug.

API: {api_name}
PyTorch version: {PYTORCH_VERSION}
Bug type: {"CRASH / memory error" if artifact_type == "crash" else "TIMEOUT / hang"}
Found by: coverage-guided fuzzing (FlashFuzz)

Minimal reproduction:
```python
{clean_code}
```

Error output:
```
{test_output[:2000]}
```

C++ backend crash output (from libFuzzer):
```
{crash_output[:1500]}
```
{non_crash_note}
Write the bug report in this EXACT format (output raw markdown, no outer code fences):

## Bug

<1-2 sentence description of what is wrong>

### Reproduction

```python
<paste the minimal reproduction code>
```

### Expected behavior

<What should happen instead — e.g. "Should raise a clear error" or "Should complete without crashing">

### Actual behavior

```
<Key lines from the error output>
```

### Environment

- PyTorch version: {PYTORCH_VERSION}
- OS: Ubuntu 24.04 (Docker)
- Python: 3.12
- Build: CPU-only

### Additional context

Found by coverage-guided C++ fuzzing (libFuzzer) of the PyTorch C++ API.
The bug is in the C++ backend and affects both C++ and Python frontends.
<Add any other relevant observations about the bug>
"""
    response = run_claude(prompt, model)
    if not response:
        return None
    return extract_markdown(response)


# ---------------------------------------------------------------------------
# Pixi environment
# ---------------------------------------------------------------------------

def ensure_pixi_env():
    """Create pixi.toml in the output directory for easy reproduction."""
    pixi_path = OUTPUT_DIR / "pixi.toml"
    if pixi_path.exists():
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pixi_path.write_text(f"""[project]
name = "flashfuzz-bug-reports"
version = "0.1.0"
description = "Bug reproductions for PyTorch {PYTORCH_VERSION} found by FlashFuzz"
channels = ["pytorch", "conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.12"
pytorch-cpu = "{PYTORCH_VERSION}.*"

[tasks]
# Run a specific bug reproduction:
#   pixi run reproduce torch.dot
reproduce = {{ cmd = "python3 $1/reproduce.py", description = "Run reproduce.py for an API" }}
""")
    print(f"  Created {pixi_path}")

    # Also write a README
    readme_path = OUTPUT_DIR / "README.md"
    if not readme_path.exists():
        readme_path.write_text(f"""# FlashFuzz Bug Reports — PyTorch {PYTORCH_VERSION}

Bugs found by coverage-guided C++ fuzzing of PyTorch APIs.

## Quick start

```bash
# Using pixi (recommended):
pixi install
pixi run reproduce torch.dot

# Or using pip:
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 torch.dot/reproduce.py
```

## Structure

Each API directory contains:
- `reproduce.py` — clean, minimal Python reproduction with concrete tensor values
- `reproduce_triage.py` — raw triage version (may use hex decoding)
- `bug_report.md` — formatted for PyTorch GitHub issue submission
- `bug_report.json` — machine-readable metadata
""")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def get_harness_code(api_name: str) -> Optional[str]:
    path = TESTHARNESS_DIR / api_name / "main.cpp"
    return path.read_text() if path.exists() else None


def process_api(api_name: str, dry_run: bool = False, model: str = "sonnet") -> dict:
    """Full pipeline for one API."""
    result = {
        "api": api_name,
        "pytorch_version": PYTORCH_VERSION,
        "timestamp": datetime.now().isoformat(),
        "artifact_type": None,
        "decoded": None,
        "crash_output": None,
        "reproduced": False,
        "attempts": 0,
        "triage_code": None,
        "clean_code": None,
        "test_output": None,
        "clean_test_output": None,
        "classification": None,
        "classification_reason": None,
        "status": "pending",
    }

    # Find artifacts
    artifacts_dir = RESULT_DIR / api_name / "artifacts"
    if not artifacts_dir.exists():
        result["status"] = "no_artifacts"
        return result

    artifact_files = sorted(artifacts_dir.iterdir())
    crash_files = [f for f in artifact_files if f.name.startswith("crash-")]
    timeout_files = [f for f in artifact_files if f.name.startswith("timeout-")]

    if crash_files:
        artifact = crash_files[0]
        result["artifact_type"] = "crash"
    elif timeout_files:
        artifact = timeout_files[0]
        result["artifact_type"] = "timeout"
    else:
        result["status"] = "no_artifacts"
        return result

    print(f"  Artifact: {artifact.name} ({artifact.stat().st_size} bytes)")

    # 1. Decode
    try:
        decoded = decode_artifact(artifact)
        result["decoded"] = decoded
        for t in decoded.get("tensors", []):
            print(f"    Tensor {t['index']}: dtype={t['dtype']}, shape={t['shape']}")
    except Exception as e:
        print(f"    Decode failed: {e}")
        result["status"] = "decode_failed"
        return result

    # 2. Replay in Docker
    print("  Replaying in Docker...")
    crash_output = replay_crash_in_docker(api_name, artifact)
    result["crash_output"] = crash_output[:5000]
    print(f"    Replay: {len(crash_output)} chars")

    if dry_run:
        result["status"] = "dry_run"
        return result

    # 3. Get harness code
    harness_code = get_harness_code(api_name)
    if not harness_code:
        result["status"] = "no_harness"
        return result

    # 4. Stage 1: Triage — generate and test reproduce.py
    prev_code, prev_error = None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result["attempts"] = attempt
        print(f"  Triage attempt {attempt}/{MAX_ATTEMPTS}...")

        code = generate_triage_code(
            api_name, crash_output, harness_code, decoded,
            result["artifact_type"], prev_code, prev_error, model,
        )
        if not code:
            print("    No code generated")
            time.sleep(INTER_ATTEMPT_DELAY)
            continue

        result["triage_code"] = code

        print("  Testing in Docker...")
        test_result = test_reproduce_in_docker(code)
        test_output = test_result["output"]
        result["test_output"] = test_output[:5000]

        if test_result["has_hard_crash"]:
            # Hard crash (segfault, ASAN, timeout, kill) — always a true positive
            print("    BUG REPRODUCED (hard crash)!")
            result["reproduced"] = True
            result["status"] = "reproduced"
            result["classification"] = "true_positive"
            result["classification_reason"] = "Hard crash detected (segfault/ASAN/timeout)"
            break

        elif test_result["has_runtime_error"]:
            # RuntimeError — needs classification
            print(f"    RuntimeError: {test_result['runtime_error_message'][:120]}")
            print("    Classifying RuntimeError...")
            time.sleep(INTER_ATTEMPT_DELAY)
            classification, reason = classify_runtime_error(
                api_name, code, test_result["runtime_error_message"] or "",
                test_output, model,
            )
            result["classification"] = classification
            result["classification_reason"] = reason
            print(f"    Classification: {classification} — {reason}")

            if classification == "true_positive":
                print("    BUG REPRODUCED (true positive RuntimeError)!")
                result["reproduced"] = True
                result["status"] = "reproduced"
                break
            elif classification == "false_positive_repro_bug":
                # Reproduction code is broken — retry with hint
                print("    Repro code has a bug, retrying with hint...")
                prev_code = code
                prev_error = (
                    f"YOUR REPRODUCTION CODE HAS A BUG — the error occurs in setup "
                    f"code BEFORE the target API ({api_name}) is called. Fix your setup "
                    f"code first.\n\nError: {test_output}"
                )
                time.sleep(INTER_ATTEMPT_DELAY)
                continue
            else:
                # false_positive — PyTorch correctly rejects invalid input
                print("    False positive: PyTorch correctly rejects invalid input")
                result["reproduced"] = False
                result["status"] = "false_positive"
                break
        else:
            # Clean exit, no error — retry with feedback
            print(f"    Not reproduced (clean exit): {test_output[:200]}")
            prev_code = code
            prev_error = test_output
            time.sleep(INTER_ATTEMPT_DELAY)

    if result["status"] == "pending":
        result["status"] = "not_reproduced"

    if not result["reproduced"]:
        return result

    # 5. Stage 2: Clean — rewrite with concrete tensors
    print("  Generating clean reproduce.py...")
    time.sleep(INTER_ATTEMPT_DELAY)
    clean_code = generate_clean_code(api_name, result["triage_code"],
                                     result["test_output"], model)
    if clean_code:
        # Verify it still reproduces
        print("  Verifying clean code...")
        clean_result = test_reproduce_in_docker(clean_code)
        result["clean_test_output"] = clean_result["output"][:5000]
        if clean_result["has_hard_crash"] or clean_result["has_runtime_error"]:
            result["clean_code"] = clean_code
            print("    Clean code verified!")
        else:
            print(f"    Clean code doesn't reproduce, keeping triage version")
            result["clean_code"] = None

    # 6. Stage 3: Bug report markdown
    print("  Generating bug report...")
    time.sleep(INTER_ATTEMPT_DELAY)
    final_code = result.get("clean_code") or result["triage_code"]
    final_output = result.get("clean_test_output") or result["test_output"]
    bug_md = generate_bug_report_md(
        api_name, final_code, final_output,
        result["artifact_type"], crash_output,
        has_hard_crash=(result["classification"] == "true_positive"
                        and "Hard crash" in (result.get("classification_reason") or "")),
        model=model,
    )
    result["bug_report_md"] = bug_md

    return result


def save_result(api_name: str, result: dict):
    """Save all outputs for an API."""
    api_dir = OUTPUT_DIR / api_name
    api_dir.mkdir(parents=True, exist_ok=True)

    # bug_report.json (machine-readable)
    # Strip large fields for JSON
    json_result = {k: v for k, v in result.items() if k != "bug_report_md"}
    (api_dir / "bug_report.json").write_text(json.dumps(json_result, indent=2, default=str))

    # reproduce_triage.py (raw triage version)
    if result.get("triage_code"):
        (api_dir / "reproduce_triage.py").write_text(result["triage_code"])

    # reproduce.py (clean version, or triage as fallback)
    final_code = result.get("clean_code") or result.get("triage_code")
    if final_code:
        (api_dir / "reproduce.py").write_text(final_code)

    # bug_report.md
    if result.get("bug_report_md"):
        (api_dir / "bug_report.md").write_text(result["bug_report_md"])

    print(f"  Saved to {api_dir}/")


def get_apis_with_artifacts() -> list[tuple[str, str]]:
    apis = []
    if not RESULT_DIR.exists():
        return apis
    for api_dir in sorted(RESULT_DIR.iterdir()):
        if not api_dir.is_dir():
            continue
        artifacts_dir = api_dir / "artifacts"
        if not artifacts_dir.exists():
            continue
        files = list(artifacts_dir.iterdir())
        has_crash = any(f.name.startswith("crash-") for f in files)
        has_timeout = any(f.name.startswith("timeout-") for f in files)
        if has_crash:
            apis.append((api_dir.name, "crash"))
        elif has_timeout:
            apis.append((api_dir.name, "timeout"))
    apis.sort(key=lambda x: (0 if x[1] == "crash" else 1, x[0]))
    return apis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce fuzzer bugs in Python frontend")
    parser.add_argument("--api", type=str, help="Process only this API")
    parser.add_argument("--crashes-only", action="store_true", help="Only crash artifacts")
    parser.add_argument("--timeouts-only", action="store_true", help="Only timeout artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Decode + replay only, no Claude")
    parser.add_argument("--limit", type=int, default=0, help="Max APIs to process")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model")
    parser.add_argument("--delay", type=int, default=INTER_API_DELAY, help="Seconds between APIs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-processed APIs")
    parser.add_argument("--skip-apis-file", type=str, help="File with API names to skip (one per line)")
    parser.add_argument("--dedup-file", type=str, help="File with dedup groups (representative -> dup1, dup2)")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    args = parser.parse_args()

    if not args.status and not args.dry_run:
        if not check_docker_image():
            print(f"ERROR: Docker image '{DOCKER_IMAGE}' not found.")
            print(f"Build: docker build -t {DOCKER_IMAGE} -f docker/torch-2.10-fuzz.Dockerfile .")
            sys.exit(1)

    all_apis = get_apis_with_artifacts()

    if args.status:
        crash_apis = [a for a in all_apis if a[1] == "crash"]
        timeout_apis = [a for a in all_apis if a[1] == "timeout"]
        print(f"APIs with artifacts: {len(all_apis)}")
        print(f"  Crashes: {len(crash_apis)}")
        print(f"  Timeouts: {len(timeout_apis)}")
        if OUTPUT_DIR.exists():
            reproduced = not_reproduced = false_positive = 0
            has_clean = has_md = 0
            for d in OUTPUT_DIR.iterdir():
                rp = d / "bug_report.json"
                if rp.exists():
                    with open(rp) as f:
                        r = json.load(f)
                    if r.get("reproduced"):
                        reproduced += 1
                    elif r.get("status") == "false_positive":
                        false_positive += 1
                    elif r.get("status") not in ("dry_run", "pending"):
                        not_reproduced += 1
                if (d / "reproduce.py").exists() and r.get("clean_code"):
                    has_clean += 1
                if (d / "bug_report.md").exists():
                    has_md += 1
            print(f"\nResults in {OUTPUT_DIR}:")
            print(f"  Reproduced: {reproduced}")
            print(f"  False positives: {false_positive}")
            print(f"  Not reproduced: {not_reproduced}")
            print(f"  Clean reproduce.py: {has_clean}")
            print(f"  Bug reports (md): {has_md}")
        return

    # Create pixi environment
    ensure_pixi_env()

    # Filter
    if args.api:
        all_apis = [(a, t) for a, t in all_apis if a == args.api]
        if not all_apis:
            print(f"Error: API '{args.api}' not found")
            sys.exit(1)
    if args.crashes_only:
        all_apis = [(a, t) for a, t in all_apis if t == "crash"]
    if args.timeouts_only:
        all_apis = [(a, t) for a, t in all_apis if t == "timeout"]
    if args.skip_existing:
        existing = set()
        if OUTPUT_DIR.exists():
            existing = {d.name for d in OUTPUT_DIR.iterdir()
                        if d.is_dir() and (d / "bug_report.json").exists()}
        before = len(all_apis)
        all_apis = [(a, t) for a, t in all_apis if a not in existing]
        if before > len(all_apis):
            print(f"Skipping {before - len(all_apis)} already processed")
    if args.skip_apis_file:
        skip_set = set()
        with open(args.skip_apis_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    skip_set.add(line)
        before = len(all_apis)
        all_apis = [(a, t) for a, t in all_apis if a not in skip_set]
        if before > len(all_apis):
            print(f"Skipping {before - len(all_apis)} APIs from skip list")
    if args.dedup_file:
        # Parse dedup file: "representative -> dup1, dup2"
        # Only the representative is processed; duplicates are skipped.
        dup_set = set()
        with open(args.dedup_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "->" not in line:
                    continue
                _, dups_str = line.split("->", 1)
                for dup in dups_str.split(","):
                    dup = dup.strip()
                    if dup:
                        dup_set.add(dup)
        before = len(all_apis)
        all_apis = [(a, t) for a, t in all_apis if a not in dup_set]
        if before > len(all_apis):
            print(f"Skipping {before - len(all_apis)} duplicate APIs")
    if args.limit > 0:
        all_apis = all_apis[:args.limit]

    print(f"Processing {len(all_apis)} APIs (model={args.model}, delay={args.delay}s)\n")

    stats = {"total": 0, "reproduced": 0, "false_positive": 0,
             "not_reproduced": 0, "failed": 0}

    for i, (api_name, artifact_type) in enumerate(all_apis, 1):
        print(f"\n[{i}/{len(all_apis)}] {api_name} ({artifact_type})")
        stats["total"] += 1

        try:
            result = process_api(api_name, dry_run=args.dry_run, model=args.model)
            save_result(api_name, result)
            if result.get("reproduced"):
                stats["reproduced"] += 1
            elif result.get("status") == "false_positive":
                stats["false_positive"] += 1
            elif result.get("status") not in ("dry_run", "pending"):
                stats["not_reproduced"] += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            stats["failed"] += 1
            save_result(api_name, {
                "api": api_name, "status": "error", "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        if i < len(all_apis) and not args.dry_run:
            print(f"  Waiting {args.delay}s...")
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"SUMMARY: {stats['reproduced']} reproduced, "
          f"{stats['false_positive']} false positive, "
          f"{stats['not_reproduced']} not reproduced, {stats['failed']} failed "
          f"(of {stats['total']} total)")


if __name__ == "__main__":
    main()
