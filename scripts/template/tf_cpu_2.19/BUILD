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
    "-O0",
],
    linkopts = [
    "-fsanitize=fuzzer",
],
    # Fully static TensorFlow dependencies (NO shared libraries)
    deps = [
    "//tensorflow/cc:scope",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:tensorflow",
    "//tensorflow/cc:cc_ops",
    "//tensorflow/cc:client_session",
],
)
