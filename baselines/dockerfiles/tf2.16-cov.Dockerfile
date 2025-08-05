# Use Ubuntu 22.04 as a base image for modern toolchains
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies, Python, and LLVM/Clang
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    clang \
    llvm \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install Bazel (version required by TensorFlow 2.16)
ARG BAZEL_VERSION=6.5.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Set working directory
WORKDIR /usr/src

# Clone TensorFlow source code
ARG TF_VERSION=v2.16.1
RUN git clone --depth 1 --branch ${TF_VERSION} https://github.com/tensorflow/tensorflow.git

WORKDIR /usr/src/tensorflow

# Install TensorFlow dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy wheel packaging requests opt_einsum
RUN python -m pip install keras_nightly

# Configure TensorFlow build
# Set CC and CXX to clang and clang++ for the build
ENV CC=clang
ENV CXX=clang++
# Disable jemalloc to avoid potential build issues
ENV TF_SYSTEM_JEMALLOC=0
# Set Python path for configure script
ENV PYTHON_BIN_PATH=/usr/bin/python
# Accept default configurations non-interactively
ENV TF_CONFIGURE_IOS=0

# Run configure, explicitly passing the python path
RUN PYTHON_BIN_PATH=/usr/bin/python3.12 ./configure

# Build TensorFlow pip package with coverage flags
# -fprofile-instr-generate and -fcoverage-mapping are used for clang-based coverage
RUN bazel build \
    --config=opt \
    --config=monolithic \
    --copt="-fprofile-instr-generate" \
    --copt="-fcoverage-mapping" \
    --linkopt="-fprofile-instr-generate" \
    --linkopt="-fcoverage-mapping" \
    //tensorflow/tools/pip_package:build_pip_package

# Build the pip package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# # Install the built package
# RUN python -m pip install /tmp/tensorflow_pkg/tensorflow-*.whl


# Set a default working directory for the final image
WORKDIR /app

# Copy the wheel file to the app directory so it can be easily extracted
RUN cp /tmp/tensorflow_pkg/tensorflow-*.whl /app/

CMD ["bash"]


# # Clean up to reduce image size
# RUN rm -rf /usr/src/tensorflow && \
#     rm -rf /tmp/tensorflow_pkg && \
#     rm -rf /root/.cache/bazel

# # Verify installation
# RUN python -c "import tensorflow as tf; print(tf.__version__)"

# # Default command
