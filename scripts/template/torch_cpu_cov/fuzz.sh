#!/bin/bash
set -euo pipefail

# Defaults; allow overrides via environment
JOBS="${JOBS:-1}"           # Prefer small per-container parallelism; scale via your scheduler
WORKERS="${WORKERS:-1}"
MAX_LEN="${MAX_LEN:-8192}"  # Start moderate; raise if shapes need bigger payloads
RSS_LIMIT="${RSS_LIMIT:-4096}"
TIMEOUT="${TIMEOUT:-10}"    # Seconds per input
SLOW_UNIT="${SLOW_UNIT:-10}" # Report inputs slower than this (seconds)
ART_PREFIX="${ART_PREFIX:-./artifacts/}"
DICT_PATH="${DICT_PATH:-}"  # Optional: path to a dictionary, if you create one

if [ ! -f "fuzz" ]; then
  echo "Error: Fuzzer executable not found!"
  exit 1
fi

mkdir -p corpus "${ART_PREFIX}"
python3 random_seed.py

# Optional sanitizer tweaks for stability in coverage mode
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0,allocator_may_return_null=1,abort_on_error=1,handle_abort=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1,halt_on_error=1}"

FUZZ_ARGS=(
  ./fuzz ./corpus
  -jobs="$JOBS"
  -workers="$WORKERS"
  -reload=1
  -max_len="$MAX_LEN"
  -prefer_small=0
  -rss_limit_mb="$RSS_LIMIT"
  -timeout="$TIMEOUT"
  -report_slow_units="$SLOW_UNIT"
  -use_value_profile=1
  -use_counters=1
  -entropic=1
  -cross_over=1
  -len_control=1
  -mutate_depth=50
  -reduce_inputs=0            # Keep variety in coverage runs
  -ignore_crashes=1
  -ignore_timeouts=1
  -ignore_ooms=1
  -artifact_prefix="$ART_PREFIX"
  -shuffle=1
  -keep_seed=1
  -max_total_time={time_budget}
  -print_final_stats=1
)

# Add dict if provided
if [ -n "$DICT_PATH" ] && [ -f "$DICT_PATH" ]; then
  FUZZ_ARGS+=(-dict="$DICT_PATH")
fi

echo "Launching libFuzzer with args:"
printf ' %q' "${FUZZ_ARGS[@]}"; echo
"${FUZZ_ARGS[@]}"

echo "Fuzzing completed."
