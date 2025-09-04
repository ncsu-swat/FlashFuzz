#!/usr/bin/env python3
import sys
import re
import json
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_fuzz_result/tf2.19-fuzz-600s/tf2.19-fuzz-600s")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")

# Patterns to detect crash info and stack traces
ERROR_PATTERNS = [
    re.compile(r"^==\d+==ERROR: .+"),
    re.compile(r"^ERROR: libFuzzer: deadly signal"),
    re.compile(r"^UndefinedBehaviorSanitizer:DEADLYSIGNAL"),
    re.compile(r"^SUMMARY: .+"),
]

STACK_LINE_RE = re.compile(r"^\s*#\d+\s+.*")
def is_stack_frame(line: str) -> bool:
    if not STACK_LINE_RE.match(line):
        return False
    # Filter out fuzzer progress lines like "#101	INITED ..." or "#123	NEW ..." or "#32	pulse ..."
    bad_tokens = ['\tINITED', '\tNEW', '\tpulse']
    if any(tok in line for tok in bad_tokens):
        return False
    # Heuristic: real frames usually contain " in " or a shared object path
    return (' in ' in line) or ('(/' in line) or ('.so' in line) or ('.cc' in line) or ('.h:' in line)

FALSE_POS_UBSAN_A0 = "UndefinedBehaviorSanitizer: SEGV on unknown address 0x0000000000a0"

def is_false_positive(lines):
    # Rule (2): UBSan SEGV @ 0x...00a0
    for ln in lines:
        if FALSE_POS_UBSAN_A0 in ln:
            return True
    # Rule (1): FPE inside fuzz.cpp
    # Look for SIGFPE/FPE/floating point exception near stack frames that include fuzz.cpp
    has_fpe = any(
        s in ln for ln in lines for s in ["SIGFPE", "FPE", "Floating point", "floating point"]
    )
    if has_fpe:
        for ln in lines:
            if "fuzz.cpp" in ln:
                return True
    return False

def parse_log_for_crash(log_path: Path):
    if not log_path.exists():
        return None
    crash_lines = []
    frames = []
    pre_frames = []  # buffer frames seen before error marker
    summary = None
    crash_started = False
    # Read line by line; keep a sliding window to check FPE rule
    try:
        with log_path.open('r', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                # capture summary as we see it
                if summary is None and line.startswith('SUMMARY: '):
                    summary = line
                # detect error line
                if any(pat.match(line) for pat in ERROR_PATTERNS):
                    crash_started = True
                    crash_lines.append(line)
                    # If we have collected frames before the error marker, prepend them
                    if pre_frames and not frames:
                        frames.extend(pre_frames[:25])
                    continue
                # collect a few more informative lines after crash start
                if crash_started:
                    if is_stack_frame(line):
                        frames.append(line)
                        continue
                    # Also collect register dump lines and terminating lines
                    if line.startswith("==") or line.startswith("MS:") or line.startswith("artifact_prefix=") or line.startswith("stat::"):
                        crash_lines.append(line)
                        continue
                    # Also collect register dump header and a few context lines
                    if "Register values" in line or "The signal is caused" in line or "Hint: address" in line:
                        crash_lines.append(line)
                        continue
                    # Stop if we've captured enough frames
                    if len(frames) >= 25:
                        break
                else:
                    # Before crash marker: capture the first stack frames if they appear
                    if is_stack_frame(line):
                        if len(pre_frames) < 30:
                            pre_frames.append(line)
    except Exception:
        return None

    if not crash_lines and not frames and summary is None:
        return None
    # Reconstruct crash type from earliest interesting line
    crash_type = None
    first_error = None
    if crash_lines:
        first_error = crash_lines[0]
    elif summary is not None:
        first_error = summary
    # Keep a few leading error lines for context
    context = []
    if crash_lines:
        context = crash_lines[:5]

    if first_error:
        crash_type = first_error
    else:
        crash_type = "Crash detected (see stack)"

    all_for_filter = []
    all_for_filter.extend(context)
    all_for_filter.extend(frames[:10])
    if is_false_positive(all_for_filter):
        return {"false_positive": True}

    return {
        "crash_type": crash_type,
        "summary": summary,
        "context": context,
        "stack": frames[:20],
        "false_positive": False,
    }

def main():
    apis = []
    base = BASE
    if not base.exists():
        print(f"Base path not found: {base}", file=sys.stderr)
        sys.exit(1)
    for api_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        api_name = api_dir.name
        artifacts = api_dir / "artifacts"
        has_crash_artifact = any(artifacts.glob("crash-*") ) if artifacts.exists() else False
        log_path = api_dir / "fuzz-0.log"
        crash_info = parse_log_for_crash(log_path)
        # If there are crash artifacts but no parsed crash info, still record minimally
        if has_crash_artifact or crash_info:
            entry = {
                "api": api_name,
                "log": str(log_path) if log_path.exists() else None,
                "artifacts": [str(p) for p in artifacts.glob("crash-*")] if artifacts.exists() else [],
            }
            if crash_info:
                entry.update(crash_info)
            else:
                entry.update({
                    "crash_type": "Crash artifact present (details not parsed)",
                    "summary": None,
                    "context": [],
                    "stack": [],
                    "false_positive": False,
                })
            apis.append(entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "tf2.19-fuzz-600s-crashes.json"
    md_path = OUT_DIR / "tf2.19-fuzz-600s-crashes.md"
    with json_path.open('w') as jf:
        json.dump(apis, jf, indent=2)

    # Write a quick Markdown summary for human reading
    with md_path.open('w') as mf:
        mf.write(f"Crash Report for {base}\n\n")
        for e in apis:
            if e.get("false_positive"):
                # Skip false positives in the Markdown output
                continue
            mf.write(f"## {e['api']}\n\n")
            mf.write(f"- Crash: {e.get('crash_type','(unknown)')}\n")
            if e.get('summary'):
                mf.write(f"- Summary: {e['summary']}\n")
            if e.get('artifacts'):
                mf.write("- Artifacts:\n")
                for a in e['artifacts']:
                    mf.write(f"  - {a}\n")
            if e.get('log'):
                mf.write(f"- Log: {e['log']}\n")
            if e.get('context'):
                mf.write("- Context:\n")
                for ln in e['context']:
                    mf.write(f"  - {ln}\n")
            if e.get('stack'):
                mf.write("- Stack (top):\n")
                for ln in e['stack']:
                    mf.write(f"  - {ln}\n")
            mf.write("\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

if __name__ == "__main__":
    main()
