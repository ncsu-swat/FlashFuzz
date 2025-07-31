#!/bin/bash

JOBS=1
WORKERS=2
MAX_LEN=5000
RSS_LIMIT=2048

/tensorflow/bazel-bin/fuzz/{api_name}/fuzz ./corpus \
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

llvm-profdata-19 merge -output=tensorflow_merged.profdata \
  *.profraw

llvm-cov-19 show /root/tensorflow/bazel-bin/fuzz/{api_name}/fuzz --show-branches=count  --instr-profile=tensorflow_merged.profdata -format=html -output-dir=tf_coverage
