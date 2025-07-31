#!/bin/bash

bazel build --config=monolithic  \
    --copt=-fsanitize=fuzzer-no-link \
    --copt=-g \
    --copt=-O0 \
    --copt=-fprofile-instr-generate \
    --copt=-fcoverage-mapping \
    --linkopt=-fsanitize=fuzzer-no-link \
    --linkopt=-L/usr/lib/clang/19/lib/linux \
    --linkopt=-lclang_rt.fuzzer-x86_64 \
    --linkopt=-fprofile-instr-generate \
    --linkopt=-fcoverage-mapping \
    --spawn_strategy=standalone \
    //fuzz/{api_name}:fuzz 
