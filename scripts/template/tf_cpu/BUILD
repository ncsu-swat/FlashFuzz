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
