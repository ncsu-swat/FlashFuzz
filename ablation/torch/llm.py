import argparse
import dataclasses
import inspect
import io
import contextlib
import os
import re
import shutil
import time
from typing import Optional

import anthropic
import torch  # required for eval(api_name)

# Paths and configuration
API_FILE = "api.txt"
HELPER_DIR = "torch_cpu_helper"


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    use_docs: bool
    use_helpers: bool


VARIANTS: list[Variant] = [
    Variant("original", use_docs=True, use_helpers=True),
    Variant("no_doc", use_docs=False, use_helpers=True),
    Variant("no_helper", use_docs=True, use_helpers=False),
    Variant("no_helper_no_doc", use_docs=False, use_helpers=False),
]


def read_api_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_helper_skeleton(dst_dir: str) -> None:
    if not os.path.isdir(HELPER_DIR):
        raise FileNotFoundError(
            f"Helper directory '{HELPER_DIR}' not found next to llm.py"
        )
    shutil.copytree(HELPER_DIR, dst_dir, dirs_exist_ok=True)


def load_helper_texts() -> tuple[str, str, str]:
    main_cpp = open(os.path.join(HELPER_DIR, "main.cpp"), "r").read()
    fuzz_cpp = open(os.path.join(HELPER_DIR, "fuzzer_utils.cpp"), "r").read()
    fuzz_h = open(os.path.join(HELPER_DIR, "fuzzer_utils.h"), "r").read()
    return main_cpp, fuzz_cpp, fuzz_h


def minimal_main_skeleton() -> str:
    return (
        "#include <cstdint>\n"
        "#include <torch/torch.h>\n"
        "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n"
        "    try {\n"
        "        size_t offset = 0;\n"
        "    } catch (const std::exception& e) {\n"
        "        // Handle exceptions thrown during fuzzing\n"
        "        std::cout << \"Exception caught: \" << e.what() << std::endl;\n"
        "        return -1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )


def get_api_docstring(api_name: str) -> Optional[str]:
    """Best-effort to obtain documentation text for a given API name.

    Attempts, in order:
    - Call `obj.docs()` and capture its stdout, if available.
    - Fallback to `inspect.getdoc(obj)`.
    """
    try:
        obj = eval(api_name)
    except Exception:
        return None

    # Try calling `.docs()` and capture printed output
    try:
        docs_attr = getattr(obj, "docs", None)
        if callable(docs_attr):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    docs_attr()
                except Exception:
                    pass
            out = buf.getvalue().strip()
            if out:
                return out
    except Exception:
        pass

    # Fallback to Python docstring
    try:
        doc = inspect.getdoc(obj)
        return doc
    except Exception:
        return None


def build_prompt(api_name: str, variant: Variant) -> str:
    """Build the exact prompt text as requested, allowing only variant-driven changes.

    - Keeps the instruction wording identical to the user's specification.
    - Inserts API docs when `use_docs` is True; otherwise omits the docs block.
    - Includes fuzzer_utils in the prompt only when `use_helpers` is True.
    """
    main_cpp, fuzzer_cpp, fuzzer_h = load_helper_texts()

    dep_line = (
        '    * The utility functions from `fuzzer_utils.h` are available via `#include "fuzzer_utils.h"`.'
        if variant.use_helpers
        else ''
    )

    Instructions = f"""
    **Instructions for `main.cpp` content:**

1.  **Complete `LLVMFuzzerTestOneInput`:** Implement the `LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)` function.
2.  **Tensor Creation:**
    * Create input tensor(s) for the `{api_name}`.
    * Explore diverse tensor properties: different ranks, dimensions (including 0 and 1), shapes, and data types.
3.  **Operation Application:** Apply the `{api_name}` operation to the created tensor(s).
4.  **Focus on Coverage and Crashes:**
    * Prioritize inputs that might lead to crashes or uncover edge cases.
    * Consider "dangerous" inputs: negative values where positive are expected (e.g., dimensions), very large or small numbers leading to overflow/underflow, empty tensors, tensors with conflicting shapes for the operation.
5.  **No Premature Sanity Checks:**
    * **Crucially, do NOT add overly defensive sanity checks** in the fuzzer code that would prevent testing edge cases (e.g., checking `if (dim_size > 0)` before creating a tensor). Let the PyTorch API handle invalid inputs. The fuzzer aims to find how the API behaves with such inputs.
    * If the PyTorch C++ API itself throws an exception for invalid setup *before* the operation can be called (e.g. `torch::tensor` creation with invalid parameters), that's acceptable. The focus is on testing the `{api_name}` operation itself.
6.  **Self-Contained `main.cpp`:** The generated code should be the complete content for `main.cpp`.
7.  **No Comments or Prints:** Remove ALL C-style (`/* ... */`, `// ...`) and C++-style comments, and any `std::cout` or `printf` statements from the final C++ code.
8.  **Dependencies:**
{dep_line}
"""

    docs_text = ''
    if variant.use_docs:
        dt = get_api_docstring(api_name)
        if dt:
            docs_text = dt

    # Prepare variant-dependent insertions first to avoid complex f-string expressions
    header_block = (
        f"""Here is the header file you can use:

    ```fuzzer_utils.cpp
    {fuzzer_cpp}
    ```

    ```fuzzer_utils.h
    {fuzzer_h}
    ```
"""
        if variant.use_helpers
        else ""
    )

    # Base prompt prefix and instructions (exact phrasing)
    prompt_prefix = f"""
I need to write a C++ testharness for the PyTorch C++ frontend operation `{api_name}`.
The testharness will be compiled with a fuzzer like libFuzzer.
Your primary goal is to generate C++ code for the `main.cpp` file.

The document of the API is as follows:

{docs_text if docs_text else ''}



{Instructions if variant.use_helpers else ''}


    Here is the skeleton of the file:

    ```main.cpp
    {main_cpp}
    ```

    {header_block}


Please complete the implementation with C++ code that properly tests the {api_name} functionality. Answer in pure string quote with ```cpp ```

    """

    print(prompt_prefix)

    # Prompt is fully constructed via f-string above
    return prompt_prefix


def extract_cpp_from_response(response_text: str) -> Optional[str]:
    patterns = [
        re.compile(r"```cpp\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```c\+\+\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```\s*(.*?)```", re.DOTALL),
    ]
    for pat in patterns:
        m = pat.search(response_text)
        if m:
            return m.group(1).strip()
    return None


def anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
    return anthropic.Anthropic(api_key=api_key)


def call_llm(
    api_name: str,
    variant: Variant,
    *,
    retries: int = 0,
    retry_backoff: float = 2.0,
) -> Optional[str]:
    client = anthropic_client()
    prompt = build_prompt(api_name, variant)
    # Exact system message per request
    system_msg = (
        "You are an expert C++ programmer specializing in writing fuzz targets for PyTorch C++ frontend operations. "
        "Your task is to generate complete, compilable `main.cpp` files based on the provided skeletons and instructions. "
        "Focus on creating code that maximizes test coverage and identifies potential crashes by exploring edge cases. "
        "Adhere strictly to output format requirements."
    )

    attempts = retries + 1
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=15000,
                temperature=0,
                system=system_msg,
                messages=[{"role": "user", "content": prompt}],
            )
            time.sleep(1)

            # Concatenate all text blocks
            content_blocks = getattr(resp, "content", [])
            content_text = "\n".join(
                (
                    block.text
                    if hasattr(block, "text")
                    else (block.get("text", "") if isinstance(block, dict) else str(block))
                )
                for block in content_blocks
            )
            cpp = extract_cpp_from_response(content_text)
            if not cpp:
                raise RuntimeError("Could not extract cpp code block from response")
            return cpp
        except Exception as e:
            last_err = e
            print(
                f"[ERROR] LLM call failed for {api_name} ({variant.name}) attempt {attempt}/{attempts}: {e}"
            )
            if attempt < attempts:
                sleep_s = retry_backoff ** (attempt - 1)
                time.sleep(sleep_s)
            else:
                break
    # All attempts failed
    if last_err is not None:
        print(
            f"[ERROR] Exhausted retries for {api_name} ({variant.name}): {last_err}"
        )
    return None


def write_variant_output(base_out_dir: str, api_name: str, variant: Variant, cpp_code: str, *, overwrite: bool) -> None:
    """Write only a single file main.cpp for each generated target.

    Regardless of the variant's helper setting, the output directory is
    cleaned and only main.cpp is kept to match the requested minimal layout.
    """
    variant_dir = os.path.join(base_out_dir, variant.name, api_name)
    if os.path.exists(variant_dir) and not overwrite:
        print(f"[SKIP] {variant_dir} exists (overwrite disabled)")
        return
    ensure_dir(variant_dir)

    # Clean directory contents so we only keep the generated main.cpp
    for entry in os.listdir(variant_dir):
        p = os.path.join(variant_dir, entry)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    main_path = os.path.join(variant_dir, "main.cpp")
    with open(main_path, "w") as f:
        f.write(cpp_code)
    print(f"[OK] Wrote {main_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C++ fuzz targets for PyTorch APIs with ablations.")
    parser.add_argument("--api-file", default=API_FILE, help="Path to API list file")
    parser.add_argument("--out-dir", default=".", help="Base output directory for generated variants")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--variants", nargs="*", choices=[v.name for v in VARIANTS], help="Subset of variants to run")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 15000)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE",1)))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="LLM call retry attempts on failure")
    args = parser.parse_args()

    apis = read_api_list(args.api_file)
    selected_variants = [v for v in VARIANTS if (args.variants is None or v.name in args.variants)]

    # Create the three variant directories under out-dir
    for v in selected_variants:
        ensure_dir(os.path.join(args.out_dir, v.name))

    for api_name in apis:
        for v in selected_variants:
            # Skip generation if output already exists and overwrite is disabled
            variant_dir = os.path.join(args.out_dir, v.name, api_name)
            main_cpp_path = os.path.join(variant_dir, "main.cpp")
            if not args.overwrite and os.path.isfile(main_cpp_path):
                print(f"[SKIP] {variant_dir} exists (overwrite disabled); skipping LLM generation")
                continue

            print(f"[GEN] {api_name} -> {v.name}")
            cpp = call_llm(
                api_name,
                v,
                retries=args.retries,
            )
            if not cpp:
                print(
                    f"[WARN] Skipping write for {api_name} ({v.name}) due to missing code after retries"
                )
                continue
            write_variant_output(args.out_dir, api_name, v, cpp, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
