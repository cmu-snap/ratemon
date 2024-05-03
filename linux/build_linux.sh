#! /usr/bin/env bash

set -eoux pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: Illegal number of parameters."
    echo "Usage: ./build_linux.sh <linux dir>"
    exit 255
fi
linux_dir="$1"

sudo apt-get update
DEBIAN_FRONTEND="noninteractive" sudo apt-get -y --no-install-recommends install \
    git \
    fakeroot \
    build-essential \
    ncurses-dev \
    xz-utils \
    libssl-dev \
    bc \
    flex \
    libelf-dev \
    bison \
    pahole

pushd "$linux_dir"
cp -fv /boot/config-$(uname -r) .config
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS
make -j"$(nproc)" menuconfig
make -j"$(nproc)"
sudo make -j "$(nproc)" modules_install
sudo make -j "$(nproc)" install
popd