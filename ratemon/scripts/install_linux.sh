#!/usr/bin/env bash

set -eoux pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: Illegal number of parameters."
    echo "Usage: ./install_linux.sh <main dir>"
    exit 255
fi
main_dir="$1"
pushd "$main_dir"

# Download Linux 5.15.156 and apply get_info() patch. Do not build it yet.
wget https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x/linux-5.15.156.tar.gz
tar xf linux-5.15.156.tar.gz
rm -f linux-5.15.156.tar.gz
pushd linux-5.15.156
git apply "$main_dir/ratemon/linux/get_info.patch"
make -j "$(nproc)" menuconfig
make -j "$(nproc)"
sudo make -j "$(nproc)" modules_install
sudo make -j "$(nproc)" install
popd

printf "Please reboot!\n"
