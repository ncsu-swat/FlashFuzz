import argparse
import dataclasses
import inspect
import os
import re
import shutil
import time
from typing import Optional

import anthropic
import tensorflow as tf  # required for eval(api_name)

# Paths and configuration
API_FILE = "api.txt"
TEMPLATE_DIR = "template"  # Provides data_type.cpp, rank.cpp, shape.cpp, note


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


def load_helper_texts() -> tuple[str, str, str, str]:
    dt = open(os.path.join(TEMPLATE_DIR, "data_type.cpp"), "r").read()
    rk = open(os.path.join(TEMPLATE_DIR, "rank.cpp"), "r").read()
    sh = open(os.path.join(TEMPLATE_DIR, "shape.cpp"), "r").read()
    note = open(os.path.join(TEMPLATE_DIR, "note"), "r").read()
    return dt, rk, sh, note


def minimal_fuzz_skeleton() -> str:
    return (
        "#include <cstdint>\n"
        "#include <iostream>\n"
        "#include <cstring>\n"
        "#include <tensorflow/core/framework/tensor.h>\n"
        "#include <tensorflow/core/framework/tensor_shape.h>\n"
        "#include <tensorflow/core/framework/types.h>\n"
        "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {\n"
        "    try {\n"
        "        size_t offset = 0;\n"
        "    } catch (const std::exception& e) {\n"
        "        // print Exception to stderr, do not remove this\n"
        "        std::cout << \"Exception caught: \" << e.what() << std::endl;\n"
        "        return -1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )


def get_api_docstring(api_name: str) -> Optional[str]:
    try:
        obj = eval(api_name)
    except Exception:
        return None
    try:
        doc = inspect.getdoc(obj)
        return doc
    except Exception:
        return None


def build_prompt(api_name: str, variant: Variant) -> str:
    """Use the exact TensorFlow prompt style and content provided by the user."""
    # Load helper templates
    type_selector, rank_cpp, shape_cpp, note_txt = load_helper_texts()

    # Docs
    docs = ""
    if variant.use_docs:
        d = get_api_docstring(api_name)
        docs = d if d else "current API does not have docstring"

    # Try to read a skeleton from cpp_cpu/fuzz.cpp; fallback to minimal skeleton
    try:
        with open("cpp_cpu/fuzz.cpp", "r") as f:
            fuzz_skeleton = f.read()
    except Exception:
        fuzz_skeleton = minimal_fuzz_skeleton()

    instructions =  f"""
    **Instructions for `fuzz.cpp` content:**

    1.  **Complete `LLVMFuzzerTestOneInput`:** Implement the `LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)` function.
    2.  **Tensor Creation:**
        * Parse data type. Strictly follow the data type specified in the document. For example:
        ```cpp
            {type_selector}
        ```
        * determine the rank
        ```cpp
            {rank_cpp}
        ```
        * determine the shape
        ```cpp
            {shape_cpp}
        ```

        * Create input tensor(s) for the `{api_name}`. 
    3.  **Operation Application:** Apply the `{api_name}` operation to the created tensor(s).
    4.  **Self-Contained `fuzz.cpp`:** The generated code should be the complete content for `fuzz.cpp`.
    5.  **No Comments:** Remove ALL C-style (`/* ... */`, `// ...`) and C++-style comments.



"""

    # Build prompt (exact wording)
    prompt = f"""
    I need to write a C++ testharness for the TensorFlow C++ frontend operation `{api_name}`.

    The document of the API is as follows:
            {docs}

    The testharness will be compiled with libFuzzer.
    Your primary goal is to generate C++ code for the `fuzz.cpp` file.

    Please complete the implementation with C++ code that properly tests the {api_name} functionality. Answer with 

    ```cpp
    ```

    and only the full C++ code that can be compiled. Do not include any other text or explanations.

    Note: {note_txt if variant.use_helpers else ""}
    
    
    {instructions if variant.use_helpers else ""}
    
        Here is the skeleton of the file:

    ```fuzz.cpp
    {fuzz_skeleton}
    ```
    """

    print(prompt)
    return prompt


def extract_cpp_from_response(response_text: str) -> Optional[str]:
    patterns = [
        re.compile(r"```cpp\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```c\+\+\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```code\s*language=cpp\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```\s*(#include[\s\S]*?)```", re.DOTALL | re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(response_text)
        if m:
            return m.group(1).strip()
    return None


def call_llm(
    api_name: str,
    variant: Variant,
    *,
    retries: int = 3,
) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] Missing ANTHROPIC_API_KEY")
        return None
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_prompt(api_name, variant)
    # Exact system message from the provided config
    system_msg = (
        "You translate Python TensorFlow API calls to their C++ equivalents with appropriate arguments."
    )

    attempts = retries + 1
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            print(prompt)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16384,
                temperature=0,
                system=system_msg,
                messages=[{"role": "user", "content": prompt}],
            )
            time.sleep(1)

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
                sleep_s = 1.5 ** (attempt - 1)
                time.sleep(sleep_s)
            else:
                break
    if last_err is not None:
        print(
            f"[ERROR] Exhausted retries for {api_name} ({variant.name}): {last_err}"
        )
    return None


def write_variant_output(
    base_out_dir: str,
    api_name: str,
    variant: Variant,
    cpp_code: str,
    *,
    overwrite: bool,
) -> None:
    variant_dir = os.path.join(base_out_dir, variant.name, api_name)
    if os.path.exists(variant_dir) and not overwrite:
        print(f"[SKIP] {variant_dir} exists (overwrite disabled)")
        return
    ensure_dir(variant_dir)

    # Clean directory contents so we only keep the generated fuzz.cpp
    for entry in os.listdir(variant_dir):
        p = os.path.join(variant_dir, entry)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    fuzz_path = os.path.join(variant_dir, "fuzz.cpp")
    with open(fuzz_path, "w") as f:
        f.write(cpp_code)
    print(f"[OK] Wrote {fuzz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C++ fuzz targets for TensorFlow APIs with ablations.")
    parser.add_argument("--api-file", default=API_FILE, help="Path to API list file")
    parser.add_argument("--out-dir", default=".", help="Base output directory for generated variants")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--variants", nargs="*", choices=[v.name for v in VARIANTS], help="Subset of variants to run")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 8000)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", 1)))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="LLM call retry attempts on failure")
    args = parser.parse_args()

    apis = read_api_list(args.api_file)
    selected_variants = [v for v in VARIANTS if (args.variants is None or v.name in args.variants)]

    for v in selected_variants:
        ensure_dir(os.path.join(args.out_dir, v.name))

    for api_name in apis:
        for v in selected_variants:
            variant_dir = os.path.join(args.out_dir, v.name, api_name)
            fuzz_cpp_path = os.path.join(variant_dir, "fuzz.cpp")
            if not args.overwrite and os.path.isfile(fuzz_cpp_path):
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
