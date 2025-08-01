FROM ubuntu:24.04

# Set non-interactive mode for package installations
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \ 
    gnupg2 \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Configure Ubuntu mirror
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirror.aaaab3n.moe/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirror.aaaab3n.moe/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources

# Install dependencies and required tools in a single RUN block to minimize layers
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    software-properties-common \
    cmake \
    git \
    fzf \
    tmux \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    unzip \
    curl \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Clang 19 and associated libraries
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 && \
    apt-get update && \
    apt-get install -y \
    libfuzzer-19-dev \
    clang-19 \
    lld-19 \
    libclang-19-dev \
    llvm-19 \
    llvm-19-dev \
    libc++-19-dev \
    libc++abi-19-dev \
    clangd-19 \
    clang-format-19 \
    libclang-rt-19-dev \
    libasan6 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/lld-19 100

# Set up Python virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies in virtual environment
RUN pip install --no-cache-dir -U \
    "numpy==2.1.1" \
    packaging \
    "protobuf==4.25.3"

# Add venv activation to .bashrc
RUN echo 'source /venv/bin/activate' >> ~/.bashrc

# Echo environment variables to .bashrc
RUN echo 'export CC=clang' >> ~/.bashrc && \
    echo 'export CXX=clang++' >> ~/.bashrc && \
    echo 'export LDFLAGS="-fuse-ld=lld-19"' >> ~/.bashrc && \
    echo 'export CFLAGS="-fsanitize=fuzzer-no-link"' >> ~/.bashrc && \
    echo 'export CXXFLAGS="-fsanitize=fuzzer-no-link"' >> ~/.bashrc && \
    echo 'export LDFLAGS="-fsanitize=fuzzer-no-link"' >> ~/.bashrc && \
    echo 'export CPLUS_INCLUDE_PATH=/usr/lib/clang/19.1.7/include:$CPLUS_INCLUDE_PATH' >> ~/.bashrc && \
    echo 'export C_INCLUDE_PATH=/usr/lib/clang/19.1.7/include:$C_INCLUDE_PATH'

# Install Bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-x86_64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Clone TensorFlow and checkout version 2.16.1
RUN git clone --branch v2.16.1 https://github.com/tensorflow/tensorflow.git /root/tensorflow --depth 1

WORKDIR /root/tensorflow

# Default command
CMD ["bash"]
