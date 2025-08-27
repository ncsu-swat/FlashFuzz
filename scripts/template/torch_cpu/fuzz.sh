#!/bin/bash

JOBS=1
WORKERS=1
MAX_LEN=65536
RSS_LIMIT=2048

if [ ! -f "fuzz" ]; then
  echo "Error: Fuzzer executable not found!" 
  exit 1
fi

python3 random_seed.py

./fuzz ./corpus \
  -jobs=$JOBS \
  -workers=$WORKERS \
  -max_len=$MAX_LEN \
  -prefer_small=0 \
  -rss_limit_mb=$RSS_LIMIT \
  -use_value_profile=1 \
  -mutate_depth=100 \
  -entropic=1 \
  -use_counters=1 \
  -cross_over=1 \
  -ignore_crashes=1 \
  -reduce_inputs=0 \
  -len_control=0 \
  -max_total_time={time_budget} \ 
  -print_final_stats=1

echo "Fuzzing completed."
