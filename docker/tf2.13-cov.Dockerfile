from ncsuswat/flashfuzz:tf2.13-base 


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

RUN  python3 build_test_harness.py --dll tf --mode cov

CMD [ "bash" ]
