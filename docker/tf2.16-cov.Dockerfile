FROM ncsuswat/flashfuzz:tf2.16-base


# Copy the test harness
COPY testharness/tf_cpu /root/tensorflow/fuzz

# Build the test harness
ENV PYTHON_BIN_PATH=/usr/bin/python3
ENV USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_NEED_ROCM=0
ENV TF_NEED_CUDA=0
ENV TF_NEED_CLANG=1
ENV TF_SET_ANDROID_WORKSPACE=0
RUN cd /root/tensorflow && ./configure

WORKDIR /root/tensorflow/fuzz
COPY scripts/ .

RUN  python3 -u build_test_harness.py --dll tf --mode cov

CMD [ "bash" ]

# RUN bazel build \
#     --copt=-fsanitize=fuzzer-no-link \
#     --copt=-g \
#     --copt=-O0 \
#     --copt=-fprofile-instr-generate \
#     --copt=-fcoverage-mapping \
#     --linkopt=-fsanitize=fuzzer-no-link \
#     --linkopt=-L/usr/lib/clang/19/lib/linux \
#     --linkopt=-lclang_rt.fuzzer-x86_64 \
#     --linkopt=-fprofile-instr-generate \
#     --linkopt=-fcoverage-mapping \
#     --spawn_strategy=standalone \
#     --keep_going \
#     //fuzz/... || true

# RUN python3 -u build_test_harness.py --dll tf --mode cov --check_build

# WORKDIR /root

# CMD [ "bash" ]
