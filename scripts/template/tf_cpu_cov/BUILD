cc_binary(
    name = "fuzz",
    srcs = [
    "fuzz.cpp",
],
    # Remove the hdrs attribute
    # Fuzzing instrumentation
    copts = [
    "-fsanitize=fuzzer",
    "-g",
    "-fprofile-instr-generate",
    "-fcoverage-mapping",
    "-O0",
],
    linkopts = [
    "-fsanitize=fuzzer",
    "-fprofile-instr-generate",
    "-fcoverage-mapping",
],
    # Fully static TensorFlow dependencies (NO shared libraries)
    deps = [
    "//tensorflow/cc:client_session",
    "//tensorflow/cc:cc_ops",
    "//tensorflow/cc:scope",
    "//tensorflow/core:tensorflow",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:all_kernels",
    "//tensorflow/core:protos_all_cc",
],
)
