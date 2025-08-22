# Start from an official Ubuntu image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    clang \
    llvm-dev \
    libomp-dev \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libssl-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libboost-all-dev \
    lld \
    lcov \
    tmux \
    btop \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/ld ld /usr/bin/lld 20

# Install Python requirements
RUN python3 -m pip install --upgrade pip setuptools wheel bs4 \
    && pip install numpy==1.23.5 typing-extensions

# Set the working directory
WORKDIR /root

RUN git clone --recursive https://github.com/pytorch/pytorch.git -b v2.2.0

CMD ["bash"]
