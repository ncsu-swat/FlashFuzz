#!/bin/bash

JOBS=1
WORKERS=2
MAX_LEN=128
RSS_LIMIT=2048

if [ ! -f "fuzz" ]; then
  echo "Error: Fuzzer executable not found!" 
  exit 1
fi

python3 random_seed.py
mkdir -p artifacts corpus

LOG=fuzz-0.log
exec > >(stdbuf -oL -eL tee -a "$LOG") 2>&1


export OMP_NUM_THREADS=1           # OpenMP threads for ops :contentReference[oaicite:1]{index=1}
export MKL_NUM_THREADS=1           # MKL threads (takes precedence over OMP) :contentReference[oaicite:2]{index=2}
export OPENBLAS_NUM_THREADS=1      # if OpenBLAS is used
export BLIS_NUM_THREADS=1          # if BLIS is used

export KMP_BLOCKTIME=0             # good perf setting for libiomp :contentReference[oaicite:3]{index=3}
export OMP_WAIT_POLICY=PASSIVE     # yield when idle (optional) :contentReference[oaicite:4]{index=4}
export PYTORCH_JIT=0    
export ASAN_OPTIONS=detect_leaks=0           # skip LSan
export UBSAN_OPTIONS=print_stacktrace=0

./fuzz ./corpus \
  -jobs=$JOBS \
  -workers=$WORKERS \
  -max_len=$MAX_LEN \
  -prefer_small=0 \
  -rss_limit_mb=$RSS_LIMIT \
  -use_value_profile=1 \
  -mutate_depth=8 \
  -entropic=1 \
  -use_counters=1 \
  -timeout=2 \
  -ignore_crashes=1 \
  -reduce_inputs=0 \
  -len_control=0 \
  -prefer_small=1 \
  -max_total_time={time_budget} \
  -print_final_stats=1 \
  -artifact_prefix="./artifacts/"

echo "Fuzzing completed."
