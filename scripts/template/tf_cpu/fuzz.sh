#!/bin/bash

JOBS=1
WORKERS=2
MAX_LEN=5000
RSS_LIMIT=2048

if [ ! -f "/root/tensorflow/bazel-bin/fuzz/{api_name}/fuzz" ]; then
  echo "Error: Fuzzer executable not found!" 
  exit 1
fi

python3 random_seed.py

/root/tensorflow/bazel-bin/fuzz/{api_name}/fuzz ./corpus \
    -jobs=$JOBS \
    -workers=$WORKERS \
    -max_len=$MAX_LEN \
    -prefer_small=0 \
    -rss_limit_mb=$RSS_LIMIT \
    -use_value_profile=1 \
    -mutate_depth=100 \
    -entropic=1 \
    -use_counters=1 \
    -ignore_crashes=1 \
    -reduce_inputs=0 \
    -len_control=0 \
    -max_total_time=180 \
    -print_final_stats=1 

echo "Fuzzing completed."
