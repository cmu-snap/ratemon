
FROM ubuntu:20.04

ENV HOME="/"

# Install tools and dependencies. Clean the apt metadata afterwards.
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y --no-install-recommends install \
        bison \
        build-essential \
        cmake \
        flex \
        git \
        libedit-dev \
        libllvm7 \
        llvm-7-dev \
        libclang-7-dev \
        python \
        zlib1g-dev \
        libelf-dev \
        libfl-dev \
        openssh-client \
        python3-distutils \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Need to set up an SSH key in order to clone with SSH.

RUN git clone https://github.com/cmu-snap/unfair.git -b bpf-dev
WORKDIR /unfair
RUN git submodule update --init

# Prepare python virtualenv
RUN python3 -m venv .venv && \
    source .venv/bin/activate && \
    pip install -r requirements.txt

# Install bcc
WORKDIR /unfair/bcc
RUN mkdir build && cd build && \
    cmake .. && \
    make -j "$(nproc)" && \
    sudo make install && \
    cmake -DPYTHON_CMD="$(which python)" .. && \
    cd src/python && \
    make -j "$(nproc)" && \
    sudo make install
