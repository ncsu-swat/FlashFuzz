FROM ncsuswat/flashfuzz:tf2.20-base


# Copy the test harness
COPY testharness/tf_cpu /root/tensorflow/fuzz

# Build the test harness
ENV PYTHON_BIN_PATH=/usr/bin/python3
ENV USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_NEED_ROCM=0
ENV TF_NEED_CUDA=0
ENV TF_NEED_CLANG=1
ENV TF_SET_ANDROID_WORKSPACE=0

# Disable pywrap rules to build traditional shared library
# Must be done BEFORE ./configure so python_version_repo is generated correctly
# Note: We remove the entire line because bool("False") == True in Starlark
RUN sed -i '/USE_PYWRAP_RULES/d' /root/tensorflow/.bazelrc

RUN cd /root/tensorflow && ./configure

WORKDIR /root/tensorflow

RUN bazel build \
    --strip=always \
    --copt=-g0 \
    --copt=-O0 \
    --copt="-Wno-error=c23-extensions" \
    --copt=-fsanitize=fuzzer-no-link \
    --linkopt=-fsanitize=fuzzer-no-link \
    --linkopt=-L/usr/lib/clang/20/lib/linux \
    --linkopt=-lclang_rt.fuzzer-x86_64 \
    --define=tsl_protobuf_header_only=false \
    //tensorflow:tensorflow_cc


RUN  cd /root/tensorflow/bazel-bin/tensorflow && \
        ln -s libtensorflow_cc.so.2.20.0 libtensorflow_cc.so && \
        ln -s libtensorflow_cc.so.2.20.0 libtensorflow_cc.so.2 && \
        ln -s libtensorflow_framework.so.2.20.0 libtensorflow_framework.so.2 && \
        ln -s libtensorflow_framework.so.2.20.0 libtensorflow_framework.so

WORKDIR /root/tensorflow/fuzz
COPY scripts/ .

RUN  python3 -u build_test_harness.py --dll tf --mode fuzz --ver 2.20

WORKDIR /root

CMD [ "bash" ]
