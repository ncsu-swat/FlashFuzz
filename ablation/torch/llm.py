import argparse
import dataclasses
import inspect
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
        "    } catch (const std::exception&) {\n"
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
    # Always provide main.cpp skeleton for context across variants
    main_cpp, fuzzer_cpp, fuzzer_h = load_helper_texts()

    doc_section = ""
    if variant.use_docs:
        doc = get_api_docstring(api_name)
        if doc:
            doc_section = (
                "\nAPI Reference (for context):\n\n" + doc + "\n"
            )
        else:
            doc_section = "\nAPI Reference (not available).\n"

    if variant.use_helpers:
        skeleton_block = (
            "```main.cpp\n" + main_cpp + "\n```\n\n"
            "```fuzzer_utils.cpp\n" + fuzzer_cpp + "\n```\n\n"
            "```fuzzer_utils.h\n" + fuzzer_h + "\n```\n"
        )
        constraints = (
            "- Include `\"fuzzer_utils.h\"` and you may use its APIs.\n"
        )
        helper_note = "The utility helpers declared in `fuzzer_utils.h` are available.\n"
    else:
        # Keep the main.cpp skeleton as structural context, but forbid using helpers.
        skeleton_block = (
            "```main.cpp\n" + main_cpp + "\n```\n"
        )
        constraints = (
            "- Do not include or reference `fuzzer_utils.*`. Implement without helpers.\n"
            "- The provided main.cpp skeleton references helpers; treat it as layout only and remove such dependencies.\n"
        )
        helper_note = "Do NOT rely on project-specific helpers; re-implement needed logic inline.\n"

    prompt = (
        f"Generate a complete C++ fuzz target (main.cpp) for the PyTorch C++ op `{api_name}`.\n"
        "The target is compiled with libFuzzer.\n"
        + doc_section
        + "\nOutput requirements:\n"
        "- Provide only one fenced code block with language `cpp`.\n"
        "- The code must be self-contained in a single `main.cpp`.\n"
        + constraints
        + "\nImplementation guidance:\n"
        "- Implement `extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)`.\n"
        "- Construct diverse tensors from raw fuzzer bytes: shapes (including 0/1 dims), dtypes, strides.\n"
        f"- Invoke `{api_name}`.\n"
        "- Prefer breadth over validation; avoid early checks that prevent edge cases.\n"
        "- Catch exceptions narrowly; keep the harness running. Return 0/-1 suitably.\n"
        + "\n\nStarter skeleton(s):\n\n"
        + skeleton_block
        + "\n" + helper_note
    )
    return prompt


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


def call_llm(api_name: str, variant: Variant, *, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    client = anthropic_client()
    prompt = build_prompt(api_name, variant)
    system_msg = (
        "You are a senior C++ engineer specializing in libFuzzer targets for"
        " PyTorch C++ frontend ops. Produce a single, compilable main.cpp that"
        " maximizes path coverage and robustness, adhering strictly to format."
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        time.sleep(0.5)

        # Concatenate all text blocks
        content_blocks = getattr(resp, "content", [])
        content_text = "\n".join(
            (block.text if hasattr(block, "text") else (block.get("text", "") if isinstance(block, dict) else str(block)))
            for block in content_blocks
        )
        cpp = extract_cpp_from_response(content_text)
        if not cpp:
            raise RuntimeError("Could not extract cpp code block from response")
        return cpp
    except Exception as e:
        print(f"[ERROR] LLM call failed for {api_name} ({variant.name}): {e}")
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
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-1-20250805"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 3000)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", 0.3)))
    args = parser.parse_args()

    apis = read_api_list(args.api_file)
    selected_variants = [v for v in VARIANTS if (args.variants is None or v.name in args.variants)]

    # Create the three variant directories under out-dir
    for v in selected_variants:
        ensure_dir(os.path.join(args.out_dir, v.name))

    for api_name in apis:
        for v in selected_variants:
            print(f"[GEN] {api_name} -> {v.name}")
            cpp = call_llm(api_name, v, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
            if not cpp:
                print(f"[WARN] Skipping write for {api_name} ({v.name}) due to missing code")
                continue
            write_variant_output(args.out_dir, api_name, v, cpp, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
