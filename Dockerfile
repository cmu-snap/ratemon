
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
        libjpeg-dev \
        libllvm7 \
        llvm-7-dev \
        libclang-7-dev \
        zlib1g-dev \
        libelf-dev \
        libfl-dev \
        openssh-client \
        software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install tools and dependencies. Clean the apt metadata afterwards.
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y --no-install-recommends install \
        python3.6 \
        python3.6-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# python3.6-distutils \
# # Need to set up an SSH key in order to clone with SSH.
# RUN mkdir "/.ssh" && ssh-keygen -t rsa -N '' -f "/.ssh/id_rsa"
# RUN ls /.ssh
# RUN git config --global core.sshCommand "ssh -i /.ssh/id_rsa -F /dev/null"

# RUN git submodule update --init

COPY requirements.txt /requirements.txt

# Prepare python virtualenv
RUN python3.6 -m venv .venv && \
    . /.venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Install bcc
RUN git clone https://github.com/iovisor/bcc.git
WORKDIR /bcc
RUN . /.venv/bin/activate && \
    git checkout v0.24.0 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j "$(nproc)" && \
    make install && \
    cmake -DPYTHON_CMD="$(which python)" .. && \
    cd src/python && \
    make -j "$(nproc)" && \
    make install

WORKDIR "$HOME"
