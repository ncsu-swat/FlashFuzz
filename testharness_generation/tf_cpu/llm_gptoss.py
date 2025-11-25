"""
LLM-driven test harness generator for TensorFlow C++ (CPU).

Uses Google Gemini to create libFuzzer targets.

- Loads helper snippets from `tf_cpu_helper/` (dtype/rank/shape/note/skeleton)
- Builds a prompt per-API with optional Python docstrings
- Writes per-API directories with a generated `fuzz.cpp`
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import shutil
import time
from typing import Optional

import google.generativeai as genai
import tensorflow as tf  # required so eval("tf.*") resolves for docstrings


# Defaults and paths
DEFAULT_API_FILE = "api.txt"
HELPER_DIR_NAME = "tf_cpu_helper"


def here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def read_api_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_helper_skeleton(src_dir: str, dst_dir: str) -> None:
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def load_helper_texts(helper_dir: str) -> tuple[str, str, str, str, str]:
    fuzz_cpp = open(os.path.join(helper_dir, "fuzz.cpp"), "r").read()
    data_type_cpp = open(os.path.join(helper_dir, "data_type.cpp"), "r").read()
    rank_cpp = open(os.path.join(helper_dir, "rank.cpp"), "r").read()
    shape_cpp = open(os.path.join(helper_dir, "shape.cpp"), "r").read()
    note_txt = open(os.path.join(helper_dir, "note"), "r").read()
    return fuzz_cpp, data_type_cpp, rank_cpp, shape_cpp, note_txt


def get_api_docstring(api_name: str) -> Optional[str]:
    try:
        obj = eval(api_name)
    except Exception:
        return None
    try:
        return inspect.getdoc(obj)
    except Exception:
        return None


def build_prompt(api_name: str, helper_dir: str, include_docs: bool = True) -> str:
    fuzz_cpp, data_type_cpp, rank_cpp, shape_cpp, note_txt = load_helper_texts(helper_dir)

    doc_section = ""
    if include_docs:
        doc = get_api_docstring(api_name)
        if doc:
            doc_section = "\nAPI Reference (Python docstring):\n\n" + doc + "\n"
        else:
            doc_section = "\nAPI Reference not available.\n"

    prompt = (
        f"Generate a complete C++ fuzz target (`fuzz.cpp`) for the TensorFlow C++ op `{api_name}`.\n"
        "The target is compiled with libFuzzer and should explore edge cases.\n"
        + doc_section
        + "\nOutput requirements:\n"
        "- Provide only one fenced code block with language `cpp` containing the entire fuzz.cpp file.\n"
        "- Implement `extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)`.\n"
        "- Keep the code self-contained (no extra source files) and avoid comments or additional prose.\n"
        "- Prefer minimal validation; let the TensorFlow op surface errors.\n"
        "- Use TensorFlow C++ APIs (e.g., `tensorflow::ops` with a `Scope`) to exercise the op.\n"
        "\nHelper snippets you may reuse verbatim:\n"
        "```cpp\n"
        + data_type_cpp
        + "\n```\n"
        "```cpp\n"
        + rank_cpp
        + "\n```\n"
        "```cpp\n"
        + shape_cpp
        + "\n```\n"
        "\nNotes for handling tensor data:\n"
        + note_txt
        + "\n\nStarter skeleton for context:\n"
        "```fuzz.cpp\n"
        + fuzz_cpp
        + "\n```\n"
    )
    return prompt


def extract_cpp_from_response(response_text: str) -> Optional[str]:
    patterns = [
        re.compile(r"```cpp\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```c\+\+\s*(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```\s*(#include[\s\S]*?)```", re.DOTALL | re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(response_text)
        if m:
            return m.group(1).strip()
    return None


def call_gemini(
    api_name: str,
    *,
    helper_dir: str,
    model: str,
    max_tokens: int,
    temperature: float,
    include_docs: bool = True,
    retries: int = 3,
    retry_delay: float = 4.0,
) -> Optional[str]:
    """Call Gemini with retries."""
    prompt = build_prompt(api_name, helper_dir, include_docs=include_docs)

    model_client = genai.GenerativeModel(
        model_name=model,
        system_instruction=(
            "You are a senior C++ engineer specializing in libFuzzer targets for TensorFlow C++ ops. "
            "Return only one C++ code block for fuzz.cpp."
        ),
    )

    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = model_client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            content = getattr(resp, "text", "") or ""
            cpp = extract_cpp_from_response(content)
            if not cpp:
                raise RuntimeError("Could not extract cpp code block from response")
            return cpp
        except Exception as e:
            if attempt < max(1, retries):
                print(
                    f"[RETRY] {api_name}: {type(e).__name__}, attempt {attempt}/{retries}; "
                    f"sleeping {retry_delay:.1f}s"
                )
                time.sleep(retry_delay)
                retry_delay *= 1.6
                continue
            print(f"[ERROR] Gemini call failed for {api_name}: {e}")
            break

    return None


def write_output(
    out_base: str,
    api_name: str,
    cpp_code: str,
    *,
    overwrite: bool,
    copy_helpers: bool,
    helper_dir: str,
) -> None:
    api_dir = os.path.join(out_base, api_name)
    if os.path.exists(api_dir) and not overwrite:
        print(f"[SKIP] {api_dir} exists (overwrite disabled)")
        return
    ensure_dir(api_dir)

    if copy_helpers:
        copy_helper_skeleton(helper_dir, api_dir)

    fuzz_path = os.path.join(api_dir, "fuzz.cpp")
    with open(fuzz_path, "w") as f:
        f.write(cpp_code)
    print(f"[OK] Wrote {fuzz_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate C++ test harnesses for TensorFlow via Gemini")
    p.add_argument("--api-file", default=DEFAULT_API_FILE, help="Path to API list file")
    p.add_argument("--out-dir", default=".", help="Base output directory for generated harnesses")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-helpers", action="store_true", help="Do not copy helper files; write only fuzz.cpp")
    p.add_argument("--no-docs", action="store_true", help="Do not include Python docstrings in prompts")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-002"))
    p.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), help="Gemini API key")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 8000)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", 0.2)))
    p.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="Retries on failure")
    p.add_argument("--retry-delay", type=float, default=float(os.environ.get("LLM_RETRY_DELAY", 10.0)), help="Initial backoff seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    helper_dir = here(HELPER_DIR_NAME)
    if not os.path.isdir(helper_dir):
        raise FileNotFoundError(f"Helper directory not found: {helper_dir}")

    api_key = args.api_key
    if not api_key:
        raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY or pass --api-key.")
    genai.configure(api_key=api_key)

    apis = read_api_list(args.api_file)

    for api_name in apis:
        api_name = api_name.strip()
        print(f"[GEN] {api_name}")
        cpp = call_gemini(
            api_name,
            helper_dir=helper_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            include_docs=not args.no_docs,
            retries=max(1, int(args.retries)),
            retry_delay=float(args.retry_delay),
        )
        if not cpp:
            print(f"[WARN] Skipping write for {api_name} due to missing code")
            continue
        write_output(
            args.out_dir,
            api_name,
            cpp,
            overwrite=args.overwrite,
            copy_helpers=not args.no_helpers,
            helper_dir=helper_dir,
        )
        time.sleep(0.5)


if __name__ == "__main__":
    main()
