FROM ncsuswat/flashfuzz:torch2.2-base

WORKDIR /root/fuzz


WORKDIR /root/pytorch

RUN pip install -r requirements.txt  && \
    mkdir -p build-fuzz && cd build-fuzz && \
    cmake \
    -DUSE_CPP_CODE_COVERAGE=1 \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEBUG=1 \
    -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link  -fno-omit-frame-pointer  -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error " \
    -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link  -fno-omit-frame-pointer  -Wno-error " \
    -DUSE_NCCL=0 \
    -DUSE_CUDA=0 \
    -DUSE_KINETO=0 \
    -DBUILD_CAFFE2=0 \
    -DUSE_DISTRIBUTED=0 \
    -DBUILD_CAFFE2_OPS=0 -DUSE_TENSORPIPE=0 -DUSE_QNNPACK=0 -DUSE_MIOPEN=0  -DUSE_XNNPACK=0 -DUSE_MKLDNN=0 -DUSE_FBGEMM=0 -DUSE_NNPACK=0  \
    -DBUILD_TEST=0 \
    -G "Unix Makefiles" \
    .. && \
    make -j$(nproc) 

COPY scripts /root/fuzz/

WORKDIR /root/fuzz

# Copy the test harness
COPY testharness/torch_cpu /root/fuzz

RUN  python3 -u build_test_harness.py --dll torch --mode fuzz


WORKDIR /root

CMD ["bash"]
