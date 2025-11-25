"""
LLM-driven test harness generator for TensorFlow C++ (CPU) using the official Gemini API.

- Loads helper snippets from `tf_cpu_helper/`
- Builds a prompt per-API with optional docstrings
- Writes per-API directories and a generated `fuzz.cpp`
- Optionally copies helper files alongside `fuzz.cpp`
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import shutil
import time
from typing import Optional

import requests
import tensorflow as tf  # required so eval("tf.xxx") resolves for docstrings


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
        f"Generate a complete C++ fuzz target (fuzz.cpp) for the TensorFlow C++ op `{api_name}`.\n"
        "The target is compiled with libFuzzer and should explore edge cases.\n"
        + doc_section
        + "\nOutput requirements:\n"
        "- Provide only one fenced code block with language `cpp`.\n"
        "- The file must be self-contained as a single `fuzz.cpp`.\n"
        "- Avoid excessive validation; let the API handle invalid inputs.\n"
        "- Focus on tensor construction variety (ranks, 0/1 dims, dtypes, shapes).\n"
        "- Implement `extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)`.\n"
        "- Catch exceptions narrowly and keep the harness running.\n"
        "- Keep the exception print so we can observe crashes.\n"
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
    api_bases: list[str],
    api_key: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    include_docs: bool = True,
    retries: int = 3,
    retry_delay: float = 6.0,
    timeout: float = 180.0,
    log_prompts: bool = True,
) -> Optional[str]:
    """Call Gemini with retries and endpoint fallback using HTTP requests."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or --api-key) is required")

    prompt = build_prompt(api_name, helper_dir, include_docs=include_docs)
    system_instruction = (
        "You are a senior C++ engineer specializing in libFuzzer targets for TensorFlow C++ ops. "
        "Return only one C++ code block for fuzz.cpp."
    )

    if log_prompts:
        print(f"[PROMPT-BEGIN] {api_name}\n{prompt}\n[PROMPT-END] {api_name}")

    def is_retriable(exc: Exception) -> bool:
        status = getattr(exc, "response", None)
        code = None
        if status is not None:
            code = getattr(status, "status_code", None)
        if code in (408, 409, 425, 429, 499):
            return True
        if code and 500 <= int(code) <= 599:
            return True
        from requests import exceptions as req_exc
        return isinstance(exc, (req_exc.Timeout, req_exc.ConnectionError))

    for base in api_bases:
        base = base.rstrip("/")
        url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        delay = retry_delay
        for attempt in range(1, max(1, retries) + 1):
            try:
                body = {
                    "systemInstruction": {"role": "system", "parts": [{"text": system_instruction}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_output_tokens,
                    },
                }
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            delay = max(delay, float(ra))
                        except Exception:
                            pass
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", []) or []
                content_text = ""
                for cand in candidates:
                    parts = cand.get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text")
                        if text:
                            content_text = text
                            break
                    if content_text:
                        break
                if not content_text:
                    raise RuntimeError("No text returned from Gemini response")
                cpp = extract_cpp_from_response(content_text)
                if not cpp:
                    raise RuntimeError("Could not extract cpp code block from response")
                return cpp
            except Exception as e:
                retriable = is_retriable(e)
                if retriable and attempt < max(1, retries):
                    print(
                        f"[RETRY] {api_name}: {type(e).__name__} at {base}, attempt {attempt}/{retries}; "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay *= 1.6
                    continue
                print(f"[ERROR] Gemini call failed for {api_name}: {e}")
                break

        print(f"[WARN] Falling back from {base} to next API base (if any)")

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
    p = argparse.ArgumentParser(description="Generate C++ test harnesses via Gemini API (TensorFlow)")
    p.add_argument("--api-file", default=DEFAULT_API_FILE, help="Path to API list file")
    p.add_argument("--out-dir", default=".", help="Base output directory for generated harnesses")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-helpers", action="store_true", help="Do not copy helper files; write only fuzz.cpp")
    p.add_argument("--no-docs", action="store_true", help="Do not include Python docstrings in prompts")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest"))
    p.add_argument(
        "--api-base",
        default=os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"),
        help="Base URL/endpoint for Gemini API (host or https://host). Comma-separated to allow fallback.",
    )
    p.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), help="API key for Gemini")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 65536)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", 1)))
    p.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="Retries per base on failure")
    p.add_argument("--retry-delay", type=float, default=float(os.environ.get("LLM_RETRY_DELAY", 10.0)), help="Initial backoff seconds")
    p.add_argument("--timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", 180.0)), help="Request timeout seconds")
    p.add_argument("--no-prompt-log", dest="log_prompts", action="store_false", help="Disable printing prompts")
    p.set_defaults(log_prompts=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    helper_dir = here(HELPER_DIR_NAME)
    if not os.path.isdir(helper_dir):
        raise FileNotFoundError(f"Helper directory not found: {helper_dir}")

    apis = read_api_list(args.api_file)

    api_bases = [b.strip() for b in str(args.api_base).split(",") if b.strip()]

    for api_name in apis:
        api_name = api_name.strip()
        print(f"[GEN] {api_name}")
        cpp = call_gemini(
            api_name,
            helper_dir=helper_dir,
            api_bases=api_bases,
            api_key=args.api_key,
            model=args.model,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            include_docs=not args.no_docs,
            retries=max(1, int(args.retries)),
            retry_delay=float(args.retry_delay),
            timeout=float(args.timeout),
            log_prompts=bool(args.log_prompts),
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
