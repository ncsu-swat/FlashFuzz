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
ENV CC_OPT_FLAGS=-Wno-sign-compare
RUN cd /root/tensorflow && ./configure

WORKDIR /root/tensorflow 

CMD [ "bash" ]


RUN bazel build  \
    --copt=-O0 --copt=-g \
    --copt=-fsanitize=fuzzer-no-link \
    --copt="-fprofile-instr-generate" \
    --copt="-fcoverage-mapping" \
    --linkopt=-fsanitize=fuzzer-no-link \
    --linkopt="-fprofile-instr-generate" \
    --linkopt="-fcoverage-mapping" \
    --copt="-Wno-error=c23-extensions" \
    --linkopt=-L/usr/lib/clang/19/lib/linux \
    --linkopt=-lclang_rt.fuzzer-x86_64 \
    //tensorflow:tensorflow_cc

RUN  cd /root/tensorflow/bazel-bin/tensorflow && \
    ln -s libtensorflow_cc.so.2.16.1 libtensorflow_cc.so && \
    ln -s libtensorflow_cc.so.2.16.1 libtensorflow_cc.so.2 && \
    ln -s libtensorflow_framework.so.2.16.1 libtensorflow_framework.so.2 && \
    ln -s libtensorflow_framework.so.2.16.1 libtensorflow_framework.so

WORKDIR /root/tensorflow/fuzz
COPY scripts/ .

RUN  python3 -u build_test_harness.py --dll tf --mode cov --no-compile

WORKDIR /root

CMD [ "bash" ]
