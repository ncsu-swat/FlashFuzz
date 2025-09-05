"""
LLM-driven test harness generator for PyTorch C++ (CPU).

Rewritten to use an OpenAI-compatible Chat Completions endpoint for GPT-OSS
served at https://ollama1.aaaab3n.moe with model `gpt-oss:120b`.

- Loads helper skeletons from `torch_cpu_helper/`
- Builds a prompt per-API with optional docstrings
- Writes per-API directories and a generated `main.cpp`
- Optionally copies helper files alongside `main.cpp`
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import time
from typing import Optional

import requests
import torch  # required so eval("torch.xxx") resolves for docstrings


# Defaults and paths
DEFAULT_API_FILE = "api.txt"
HELPER_DIR_NAME = "torch_cpu_helper"


def here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def read_api_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_helper_skeleton(src_dir: str, dst_dir: str) -> None:
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def load_helper_texts(helper_dir: str) -> tuple[str, str, str]:
    main_cpp = open(os.path.join(helper_dir, "main.cpp"), "r").read()
    fuzz_cpp = open(os.path.join(helper_dir, "fuzzer_utils.cpp"), "r").read()
    fuzz_h = open(os.path.join(helper_dir, "fuzzer_utils.h"), "r").read()
    return main_cpp, fuzz_cpp, fuzz_h


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
    main_cpp, fuzzer_cpp, fuzzer_h = load_helper_texts(helper_dir)

    doc_section = ""
    if include_docs:
        doc = get_api_docstring(api_name)
        if doc:
            doc_section = "\nAPI Reference (Python docstring):\n\n" + doc + "\n"
        else:
            doc_section = "\nAPI Reference not available.\n"

    prompt = (
        f"Generate a complete C++ fuzz target (main.cpp) for the PyTorch C++ op `{api_name}`.\n"
        "The target is compiled with libFuzzer and should explore edge cases.\n"
        + doc_section
        + "\nOutput requirements:\n"
        "- Provide only one fenced code block with language `cpp`.\n"
        "- The file must be self-contained as a single `main.cpp`.\n"
        "- You may include and use `\"fuzzer_utils.h\"`.\n"
        "- Avoid excessive validation; let the API handle invalid inputs.\n"
        "- Focus on tensor construction variety (ranks, 0/1 dims, dtypes, shapes).\n"
        "- Implement `extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)`.\n"
        "- Catch exceptions narrowly and keep the harness running.\n"
        + "\n\nStarter skeletons for context:\n\n"
        + "```main.cpp\n" + main_cpp + "\n```\n\n"
        + "```fuzzer_utils.cpp\n" + fuzzer_cpp + "\n```\n\n"
        + "```fuzzer_utils.h\n" + fuzzer_h + "\n```\n"
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


def call_gpt_oss(
    api_name: str,
    *,
    helper_dir: str,
    api_bases: list[str],
    api_key: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    include_docs: bool = True,
    retries: int = 3,
    retry_delay: float = 4.0,
    timeout: float = 180.0,
) -> Optional[str]:
    """Call GPT-OSS (OpenAI-compatible) with retries and endpoint fallback.

    Retries on timeouts, connection errors, HTTP 5xx/429, and specific CDN
    gateway timeouts (e.g., 524). Falls back across multiple API bases.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    prompt = build_prompt(api_name, helper_dir, include_docs=include_docs)

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior C++ engineer specializing in libFuzzer targets for PyTorch C++ ops. "
                    "Return only one C++ code block for main.cpp."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    def is_retriable(status_code: Optional[int], exc: Optional[BaseException]) -> bool:
        if exc is not None:
            from requests import exceptions as req_exc
            return isinstance(exc, (req_exc.Timeout, req_exc.ConnectionError))
        if status_code is None:
            return True
        if status_code in (408, 409, 425, 429, 499, 524, 529):
            return True
        return 500 <= status_code <= 599

    for base in api_bases:
        base = base.rstrip("/")
        url = base + "/v1/chat/completions"
        delay = retry_delay
        for attempt in range(1, max(1, retries) + 1):
            try:
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
                content = data["choices"][0]["message"]["content"]
                cpp = extract_cpp_from_response(content)
                if not cpp:
                    raise RuntimeError("Could not extract cpp code block from response")
                return cpp
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if is_retriable(status, None) and attempt < max(1, retries):
                    print(
                        f"[RETRY] {api_name}: HTTP {status} from {base}, attempt {attempt}/{retries}; "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay *= 1.6
                    continue
                try:
                    snippet = e.response.text[:1000] if e.response is not None else ""
                except Exception:
                    snippet = ""
                print(f"[ERROR] GPT-OSS call failed for {api_name}: HTTP {status} at {base}")
                if snippet:
                    print(f"[DEBUG] response: {snippet}")
                break
            except Exception as e:
                retriable = is_retriable(None, e)
                if retriable and attempt < max(1, retries):
                    print(
                        f"[RETRY] {api_name}: {type(e).__name__} at {base}, attempt {attempt}/{retries}; "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay *= 1.6
                    continue
                print(f"[ERROR] GPT-OSS call failed for {api_name}: {e}")
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
        # Copy helper files alongside main.cpp
        copy_helper_skeleton(helper_dir, api_dir)

    # Ensure only main.cpp is overwritten; keep helper files if present
    main_path = os.path.join(api_dir, "main.cpp")
    with open(main_path, "w") as f:
        f.write(cpp_code)
    print(f"[OK] Wrote {main_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate C++ test harnesses via GPT-OSS (OpenAI-compatible endpoint)")
    p.add_argument("--api-file", default=DEFAULT_API_FILE, help="Path to API list file")
    p.add_argument("--out-dir", default=".", help="Base output directory for generated harnesses")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-helpers", action="store_true", help="Do not copy helper files; write only main.cpp")
    p.add_argument("--no-docs", action="store_true", help="Do not include Python docstrings in prompts")
    p.add_argument("--model", default=os.environ.get("GPTOSS_MODEL", "gpt-oss:120b"))
    p.add_argument(
        "--api-base",
        default=os.environ.get("GPTOSS_API_BASE", "https://ollama1.aaaab3n.moe"),
        help=(
            "Base URL for OpenAI-compatible API (no /v1 suffix). "
            "Comma-separate multiple bases to enable fallback."
        ),
    )
    p.add_argument("--api-key", default=os.environ.get("GPTOSS_API_KEY"), help="Bearer token for API (optional)")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", 8000)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", 0.2)))
    p.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="Retries per base on failure")
    p.add_argument("--retry-delay", type=float, default=float(os.environ.get("LLM_RETRY_DELAY", 10.0)), help="Initial backoff seconds")
    p.add_argument("--timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", 180.0)), help="Request timeout seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve helper dir relative to this script
    helper_dir = here(HELPER_DIR_NAME)
    if not os.path.isdir(helper_dir):
        raise FileNotFoundError(f"Helper directory not found: {helper_dir}")

    apis = read_api_list(args.api_file)

    # Support comma-separated fallback API bases
    api_bases = [b.strip() for b in str(args.api_base).split(",") if b.strip()]

    for api_name in apis:
        api_name = api_name.strip()
        print(f"[GEN] {api_name}")
        cpp = call_gpt_oss(
            api_name,
            helper_dir=helper_dir,
            api_bases=api_bases,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            include_docs=not args.no_docs,
            retries=max(1, int(args.retries)),
            retry_delay=float(args.retry_delay),
            timeout=float(args.timeout),
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
    
    
        
    
