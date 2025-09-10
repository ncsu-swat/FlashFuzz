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
    -DCMAKE_CXX_FLAGS="-O0 -g -fsanitize=fuzzer-no-link  -fno-omit-frame-pointer  -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error -fprofile-instr-generate -fcoverage-mapping " \
    -DCMAKE_C_FLAGS="-O0 -g -fsanitize=fuzzer-no-link  -fno-omit-frame-pointer  -Wno-error -fprofile-instr-generate -fcoverage-mapping" \
    -DBUILD_TEST=0 -DUSE_MKLDNN=0 -DUSE_OPENMP=0 -DUSE_CUDA=0 -DUSE_NCCL=0 \
    -G "Unix Makefiles" \
    .. && \
    make -j$(nproc) 


COPY scripts /root/fuzz/

WORKDIR /root/fuzz

# Copy the test harness
COPY testharness/torch_cpu /root/fuzz

RUN  python3 -u build_test_harness.py --dll torch --mode fuzz --no-compile

WORKDIR /root

CMD ["bash"]
