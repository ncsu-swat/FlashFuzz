#!/bin/bash

clang++ fuzz.cpp \
-std=c++17 \
    -g \
    -O0 \
    -fsanitize=fuzzer \
    -fprofile-instr-generate \
    -fcoverage-mapping \
-I /root/tensorflow \
-I /root/tensorflow/bazel-tensorflow \
-I /root/tensorflow/bazel-bin \
-I /root/tensorflow/bazel-tensorflow/external/com_google_absl \
-I /root/tensorflow/bazel-tensorflow/external/com_google_protobuf/src \
-I /root/tensorflow/bazel-tensorflow/external/eigen_archive \
-I /root/tensorflow/bazel-tensorflow/external/local_tsl \
-I /root/tensorflow/bazel-bin/external/local_tsl \
-I /root/tensorflow/bazel-tensorflow/external/nsync/public \
-I /root/tensorflow/bazel-tensorflow/external \
-L /root/tensorflow/bazel-bin/tensorflow \
-Wl,-rpath,'$ORIGIN' \
-ltensorflow_cc \
-ltensorflow_framework \
-lpthread \
-o fuzz

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi
