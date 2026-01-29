#!/usr/bin/env python3
"""
Automated test harness fixer for PyTorch APIs using Claude Code CLI.

This script traverses all PyTorch API test harnesses and uses Claude Code to:
1. Check if the API exists and is used correctly
2. Fix compilation errors
3. Improve runtime stability
4. Document all changes in api.json
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configuration
TESTHARNESS_DIR = Path(__file__).parent / "testharness" / "torch_cpu"
FUZZER_UTILS_DIR = Path(__file__).parent / "scripts" / "template" / "torch_cpu"
API_JSON_PATH = Path(__file__).parent / "api.json"
DOCKER_IMAGE = "ncsuswat/flashfuzz:torch2.7-fuzz"
DOCKER_HARNESS_PATH = "/root/testharness"
TIMEOUT_COMPILE = 120  # seconds
TIMEOUT_FUZZ = 60  # seconds
CLAUDE_TIMEOUT = 300  # seconds for claude command


def load_api_json() -> dict:
    """Load the api.json tracking file."""
    if API_JSON_PATH.exists():
        with open(API_JSON_PATH, "r") as f:
            return json.load(f)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": "Tracking document for PyTorch/TensorFlow test harness fixes",
        "metadata": {"created": datetime.now().strftime("%Y-%m-%d"), "last_updated": None},
        "apis": {},
    }


def save_api_json(data: dict):
    """Save the api.json tracking file."""
    data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(API_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_all_apis() -> list[str]:
    """Get all PyTorch API names from testharness directory."""
    if not TESTHARNESS_DIR.exists():
        print(f"Error: {TESTHARNESS_DIR} does not exist")
        sys.exit(1)

    apis = sorted([d.name for d in TESTHARNESS_DIR.iterdir() if d.is_dir()])
    return apis


def read_harness_code(api_name: str) -> Optional[str]:
    """Read the main.cpp file for a given API."""
    harness_path = TESTHARNESS_DIR / api_name / "main.cpp"
    if harness_path.exists():
        with open(harness_path, "r") as f:
            return f.read()
    return None


def write_harness_code(api_name: str, code: str):
    """Write the main.cpp file for a given API."""
    harness_path = TESTHARNESS_DIR / api_name / "main.cpp"
    with open(harness_path, "w") as f:
        f.write(code)


def run_docker_command(cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command inside Docker container."""
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{TESTHARNESS_DIR}:{DOCKER_HARNESS_PATH}:rw",
        DOCKER_IMAGE,
        "bash", "-c", cmd
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def compile_harness(api_name: str) -> tuple[bool, str]:
    """Compile a test harness in Docker."""
    # Copy fuzzer_utils files from container's /root/fuzz/ to the mounted harness directory,
    # then compile. Cleanup happens in cleanup_harness_dir() after fuzzing.
    compile_cmd = f"""
cp /root/fuzz/fuzzer_utils.cpp {DOCKER_HARNESS_PATH}/{api_name}/ && \
cp /root/fuzz/fuzzer_utils.h {DOCKER_HARNESS_PATH}/{api_name}/ && \
cd {DOCKER_HARNESS_PATH}/{api_name} && \
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
    -I/usr/local/cuda/include \
    -std=c++17 \
    -I/. \
    main.cpp fuzzer_utils.cpp \
    -Wl,-rpath,/root/pytorch/build-fuzz/lib \
    -L/root/pytorch/build-fuzz/lib \
    -ltorch -ltorch_cpu -lc10 \
    -o fuzz_target 2>&1
"""
    returncode, stdout, stderr = run_docker_command(compile_cmd, TIMEOUT_COMPILE)
    output = stdout + stderr
    return returncode == 0, output


def cleanup_harness_dir(api_name: str):
    """Remove build artifacts from the harness directory, keeping only main.cpp."""
    api_dir = TESTHARNESS_DIR / api_name
    for pattern in ["fuzzer_utils.cpp", "fuzzer_utils.h", "fuzz_target", "*.profraw"]:
        import glob as g
        for f in g.glob(str(api_dir / pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def run_harness(api_name: str, duration: int = 60) -> tuple[bool, str]:
    """Run a test harness for fuzzing in Docker."""
    run_cmd = f"""
cd {DOCKER_HARNESS_PATH}/{api_name} && \
timeout {duration} ./fuzz_target -max_total_time={duration} 2>&1 || true
"""
    returncode, stdout, stderr = run_docker_command(run_cmd, duration + 30)
    output = stdout + stderr

    # Check for crashes or serious errors
    has_crash = "SUMMARY: AddressSanitizer" in output or "SEGV" in output or "heap-buffer-overflow" in output
    return not has_crash, output


class HarnessFixer:
    """Uses Claude Code CLI to analyze and fix test harnesses."""

    def __init__(self, model: str = "opus"):
        self.model = model
        self.system_prompt = """You are an expert C++ developer specializing in libFuzzer test harnesses for PyTorch APIs.

Your task is to analyze and fix test harnesses for PyTorch C++ APIs. The harnesses use:
- LibFuzzer for coverage-guided fuzzing
- PyTorch C++ frontend (libtorch)
- Custom fuzzer_utils.h for tensor creation from fuzzer data

When analyzing code:
1. Check if the API is used correctly according to PyTorch documentation
2. Identify compilation errors and fix them
3. Ensure proper error handling to avoid false positive crashes
4. Make sure the fuzzer explores the API thoroughly

When providing fixes, output ONLY the complete fixed main.cpp code wrapped in ```cpp``` code blocks.
If the API doesn't exist in PyTorch C++ frontend, respond with: API_NOT_FOUND

Important guidelines:
- Use torch:: namespace for all PyTorch calls
- Don't use features not available in C++ frontend
- Ensure tensors have appropriate shapes for the operation
- Use fuzzer_utils::createTensor() for input generation

CRITICAL - Exception handling for validity tracking:
- The outer try-catch block MUST log exceptions and return -1 to track input validity
- Use this exact pattern for the outer exception handler:
```cpp
catch (const std::exception &e)
{
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return -1;  // Tell libFuzzer to discard invalid input
}
```
- Inner try-catch blocks (for expected failures like shape mismatches) should NOT log, just catch silently
- Return 0 for successful execution, return -1 for exceptions

CRITICAL - Progress tracking:
- Add a static counter at the beginning of the function to track iterations
- Print progress every 10000 iterations to verify the API is being exercised
- Use this exact pattern at the START of LLVMFuzzerTestOneInput:
```cpp
static uint64_t iteration_count = 0;
iteration_count++;
if (iteration_count % 10000 == 0) {
    std::cout << "Iterations: " << iteration_count << std::endl;
}
```
"""

    def _run_claude(self, prompt: str) -> str:
        """Run claude command and return the response."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        try:
            result = subprocess.run(
                ["claude", "-p", full_prompt, "--model", self.model],
                capture_output=True,
                text=True,
                timeout=CLAUDE_TIMEOUT,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "ERROR: Claude command timed out"
        except FileNotFoundError:
            return "ERROR: claude command not found. Make sure Claude Code CLI is installed."
        except Exception as e:
            return f"ERROR: {str(e)}"

    def analyze_and_fix(
        self,
        api_name: str,
        current_code: str,
        compile_error: Optional[str] = None,
        runtime_error: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        """
        Analyze the harness and return fixed code if needed.

        Returns:
            tuple: (fixed_code or None, explanation)
        """
        user_message = f"""API: {api_name}

Current test harness code:
```cpp
{current_code}
```
"""

        if compile_error:
            user_message += f"""
Compilation failed with error:
```
{compile_error[:3000]}
```

Please fix the compilation errors.
"""
        elif runtime_error:
            user_message += f"""
Runtime/fuzzing output (may contain crashes or concerns):
```
{runtime_error[:3000]}
```

Please analyze and fix any issues that could cause false positive crashes or instability.
"""
        else:
            user_message += """
Please analyze this harness and check:
1. Is the API used correctly?
2. Are there any potential issues?
3. Can the fuzzing coverage be improved?

If the code looks good, respond with: CODE_OK
If the API doesn't exist in PyTorch C++ frontend, respond with: API_NOT_FOUND
Otherwise, provide the fixed code.
"""

        response_text = self._run_claude(user_message)

        # Check for errors
        if response_text.startswith("ERROR:"):
            return None, response_text

        # Check for special responses
        if "API_NOT_FOUND" in response_text:
            return None, "API not found in PyTorch C++ backend"

        if "CODE_OK" in response_text:
            return None, "Code looks good, no changes needed"

        # Extract code from response
        fixed_code = self._extract_code(response_text)
        if fixed_code:
            return fixed_code, "Fixed by Claude"

        return None, response_text[:500]

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract C++ code from Claude's response."""
        import re

        # Look for ```cpp ... ``` blocks
        pattern = r"```cpp\s*(.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return the longest match (likely the complete code)
            return max(matches, key=len)

        # Try without language specifier
        pattern = r"```\s*(#include.*?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return max(matches, key=len)

        return None


def process_api(
    api_name: str,
    fixer: HarnessFixer,
    api_data: dict,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Process a single API: analyze, fix, compile, and test."""

    # Check if already processed
    if skip_existing and api_name in api_data["apis"]:
        existing = api_data["apis"][api_name]
        if existing.get("status") in ["fixed", "skipped"]:
            print(f"  Skipping {api_name} (already processed)")
            return existing

    result = {
        "name": api_name,
        "dll": "torch",
        "version": "2.7",
        "status": "pending",
        "api_exists": True,
        "compilation": {"status": "not_attempted", "error": None},
        "runtime": {"status": "not_attempted", "duration_seconds": None, "crashes": None, "concerns": None},
        "changes": [],
        "notes": None,
    }

    # Read current code
    current_code = read_harness_code(api_name)
    if not current_code:
        result["status"] = "skipped"
        result["notes"] = "No main.cpp found"
        return result

    print(f"  Analyzing {api_name}...")

    # Step 1: Initial analysis
    fixed_code, explanation = fixer.analyze_and_fix(api_name, current_code)

    if "API not found" in explanation:
        result["status"] = "skipped"
        result["api_exists"] = False
        result["notes"] = "API not found in PyTorch C++ backend"
        return result

    if fixed_code and not dry_run:
        write_harness_code(api_name, fixed_code)
        result["changes"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Initial fix by Claude",
            "files_modified": ["main.cpp"],
        })
        current_code = fixed_code

    # Step 2: Try to compile
    print(f"  Compiling {api_name}...")
    compile_success, compile_output = compile_harness(api_name)

    if not compile_success:
        result["compilation"]["status"] = "failed"
        result["compilation"]["error"] = compile_output[:1000]

        # Try to fix compilation error
        print(f"  Fixing compilation error for {api_name}...")
        fixed_code, explanation = fixer.analyze_and_fix(
            api_name, current_code, compile_error=compile_output
        )

        if fixed_code and not dry_run:
            write_harness_code(api_name, fixed_code)
            result["changes"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "description": f"Fixed compilation error: {explanation[:100]}",
                "files_modified": ["main.cpp"],
            })

            # Retry compilation
            compile_success, compile_output = compile_harness(api_name)
            if compile_success:
                result["compilation"]["status"] = "success"
                result["compilation"]["error"] = None
            else:
                result["compilation"]["error"] = compile_output[:1000]
    else:
        result["compilation"]["status"] = "success"

    # Step 3: Run fuzzing if compilation succeeded
    if result["compilation"]["status"] == "success":
        print(f"  Running fuzzer for {api_name} ({TIMEOUT_FUZZ}s)...")
        run_success, run_output = run_harness(api_name, TIMEOUT_FUZZ)

        result["runtime"]["duration_seconds"] = TIMEOUT_FUZZ
        result["runtime"]["crashes"] = not run_success

        if not run_success:
            result["runtime"]["status"] = "failed"
            result["runtime"]["concerns"] = run_output[:1000]

            # Try to fix runtime issues
            print(f"  Fixing runtime issues for {api_name}...")
            current_code = read_harness_code(api_name)
            fixed_code, explanation = fixer.analyze_and_fix(
                api_name, current_code, runtime_error=run_output
            )

            if fixed_code and not dry_run:
                write_harness_code(api_name, fixed_code)
                result["changes"].append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "description": f"Fixed runtime issue: {explanation[:100]}",
                    "files_modified": ["main.cpp"],
                })

                # Recompile and retest
                compile_success, _ = compile_harness(api_name)
                if compile_success:
                    run_success, run_output = run_harness(api_name, TIMEOUT_FUZZ)
                    result["runtime"]["crashes"] = not run_success
                    if run_success:
                        result["runtime"]["status"] = "success"
                        result["runtime"]["concerns"] = None
        else:
            result["runtime"]["status"] = "success"

    # Determine final status
    if result["compilation"]["status"] == "success" and result["runtime"]["status"] == "success":
        result["status"] = "fixed"
    elif result["compilation"]["status"] == "failed":
        result["status"] = "broken"
    else:
        result["status"] = "broken"

    # Clean up build artifacts, keep only main.cpp
    cleanup_harness_dir(api_name)

    return result


def main():
    parser = argparse.ArgumentParser(description="Fix PyTorch test harnesses using Claude")
    parser.add_argument("--api", type=str, help="Process only this specific API")
    parser.add_argument("--start-from", type=str, help="Start processing from this API (alphabetically)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of APIs to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just analyze")
    parser.add_argument("--force", action="store_true", help="Reprocess already completed APIs")
    parser.add_argument("--model", type=str, default="opus", help="Claude model to use (sonnet, opus, haiku)")
    parser.add_argument("--list", action="store_true", help="List all APIs and exit")
    parser.add_argument("--status", action="store_true", help="Show processing status and exit")
    args = parser.parse_args()

    # Get all APIs
    all_apis = get_all_apis()
    print(f"Found {len(all_apis)} PyTorch APIs")

    if args.list:
        for api in all_apis:
            print(api)
        return

    # Load tracking data
    api_data = load_api_json()

    if args.status:
        total = len(all_apis)
        processed = len(api_data.get("apis", {}))
        fixed = sum(1 for a in api_data.get("apis", {}).values() if a.get("status") == "fixed")
        skipped = sum(1 for a in api_data.get("apis", {}).values() if a.get("status") == "skipped")
        broken = sum(1 for a in api_data.get("apis", {}).values() if a.get("status") == "broken")
        print(f"Total APIs: {total}")
        print(f"Processed: {processed}")
        print(f"  Fixed: {fixed}")
        print(f"  Skipped: {skipped}")
        print(f"  Broken: {broken}")
        print(f"Remaining: {total - processed}")
        return

    # Filter APIs
    apis_to_process = all_apis

    if args.api:
        if args.api in all_apis:
            apis_to_process = [args.api]
        else:
            print(f"Error: API '{args.api}' not found")
            sys.exit(1)
    elif args.start_from:
        try:
            start_idx = all_apis.index(args.start_from)
            apis_to_process = all_apis[start_idx:]
        except ValueError:
            print(f"Error: API '{args.start_from}' not found")
            sys.exit(1)

    if args.limit > 0:
        apis_to_process = apis_to_process[: args.limit]

    print(f"Processing {len(apis_to_process)} APIs...")

    # Initialize fixer
    fixer = HarnessFixer(model=args.model)

    # Process each API
    for i, api_name in enumerate(apis_to_process, 1):
        print(f"\n[{i}/{len(apis_to_process)}] Processing {api_name}")

        try:
            result = process_api(
                api_name,
                fixer,
                api_data,
                dry_run=args.dry_run,
                skip_existing=not args.force,
            )

            if not args.dry_run:
                api_data["apis"][api_name] = result
                save_api_json(api_data)

            status_icon = {"fixed": "✓", "skipped": "⊘", "broken": "✗", "pending": "?"}
            print(f"  Result: {status_icon.get(result['status'], '?')} {result['status']}")

        except Exception as e:
            print(f"  Error processing {api_name}: {e}")
            if not args.dry_run:
                api_data["apis"][api_name] = {
                    "name": api_name,
                    "dll": "torch",
                    "version": "2.7",
                    "status": "broken",
                    "api_exists": None,
                    "compilation": {"status": "not_attempted", "error": None},
                    "runtime": {"status": "not_attempted", "duration_seconds": None, "crashes": None, "concerns": None},
                    "changes": [],
                    "notes": f"Error during processing: {str(e)[:200]}",
                }
                save_api_json(api_data)

        # Small delay to avoid rate limiting
        time.sleep(1)

    print("\nDone!")

    # Print summary
    if not args.dry_run:
        api_data = load_api_json()
        fixed = sum(1 for a in api_data["apis"].values() if a.get("status") == "fixed")
        skipped = sum(1 for a in api_data["apis"].values() if a.get("status") == "skipped")
        broken = sum(1 for a in api_data["apis"].values() if a.get("status") == "broken")
        print(f"\nSummary: {fixed} fixed, {skipped} skipped, {broken} broken")


if __name__ == "__main__":
    main()
