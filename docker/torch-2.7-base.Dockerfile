# Start from an official Ubuntu image
FROM ubuntu:24.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    # PyTorch build dependencies
    libopenblas-dev \
    liblapack-dev \
    ninja-build \
    patch \
    btop \
    fzf \
    # Cleanup intermediate packages
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 

# Install latest LLVM toolchain (clang/lld) via llvm.sh (using 20 for latest)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 20 && \
    apt-get update && apt-get install -y --no-install-recommends \
    clang-20 \
    lld-20 \
    libclang-20-dev \
    llvm-20 \
    llvm-20-dev \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/lld-20 100 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 100 

# Install Python requirements (avoid upgrading apt-managed pip)
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    bs4 "numpy==1.26.4" typing-extensions

# Set the working directory
WORKDIR /root

RUN git clone --recursive https://github.com/pytorch/pytorch.git -b v2.7.0

CMD ["bash"]
