#!/usr/bin/env bash

set -eo pipefail

if [[ $# -ne 1 ]] && [[ $# -ne 2 ]]; then
	echo "ERROR: Illegal number of parameters."
	echo "Usage: ./install_linux.sh <main dir> <version default=5.15.156>"
	exit 255
fi
main_dir="$1"
version="6.9.12"
if [[ -n $2 ]]; then
	version="$2"
fi

set -ux

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

# Download Linux 5.15.156 and apply get_info() patch. Do not build it yet.
pushd "${main_dir}"
wget "https://mirrors.edge.kernel.org/pub/linux/kernel/v6.x/linux-${version}.tar.gz"
tar xf "linux-${version}.tar.gz"
rm -f "linux-${version}.tar.gz"
pushd "linux-${version}"
git apply "${main_dir}/ratemon/linux/get_info_${version}.patch"
uname="$(uname -r)"
cp -fv "/boot/config-${uname}" .config
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS
nproc="$(nproc)"
make -j "${nproc}" menuconfig
make -j "${nproc}"
sudo make -j "${nproc}" modules_install
sudo make -j "${nproc}" install
popd

printf "Please reboot!\n"
