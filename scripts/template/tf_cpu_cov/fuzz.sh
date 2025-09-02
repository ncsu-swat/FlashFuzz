#!/bin/bash

JOBS=1
WORKERS=1
MAX_LEN=5000
RSS_LIMIT=2048


if [ ! -f "./fuzz" ]; then
  echo "Error: Fuzzer executable not found!" 
  exit 1
fi
mkdir -p artifacts
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
    -ignore_crashes=1 \
    -reduce_inputs=0 \
    -len_control=0 \
    -max_total_time=60 \
    -print_final_stats=1 

llvm-profdata merge -output=tensorflow_merged.profdata \
  *.profraw

llvm-cov show  /root/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.2.16.1 -object  /root/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.2.16.1 --show-branches=count  --instr-profile=tensorflow_merged.profdata -format=html -output-dir=tf_coverage -path-equivalence=/proc/self/cwd/,/root/tensorflow/

echo "Fuzzing completed."

