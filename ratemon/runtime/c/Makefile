# SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

# Set interface with RM_IFACE environment variable.

OUTPUT := $(abspath .output)
CLANG ?= clang
LIBBPF_SRC := $(abspath ../../../libbpf/src)
BPFTOOL_SRC := $(abspath ../../../bpftool/src)
LIBBPF_OBJ := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(OUTPUT)/bpftool)
BPFTOOL ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')
VMLINUX := $(OUTPUT)/vmlinux.h
# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated
INCLUDES := -I$(OUTPUT) -I../../../libbpf/include/uapi -I$(dir $(VMLINUX))
CC := clang
# DEBUG: add -g
CFLAGS := -std=c17 -Wall -Wextra -O3
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)
# If this is not set with an environment variable, then hardcode it.
RM_IFACE ?= eno4
RM_CGROUP ?= /test_cg

CXX := clang++
# DEBUG: add -g
CXXFLAGS := -stdlib=libstdc++ -std=c++23 -Wall -Wextra -O3

APPS := ratemon_main
INTERPS := libratemon_interp

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architecture-specific dirs will be "missing" on some architectures/distros -
# headers such as asm/types.h, asm/byteorder.h, asm/socket.h, asm/sockios.h,
# sys/cdefs.h etc. might be missing.
#
# Use '-idirafter': Don't interfere with include mechanics except where the
# build would have failed anyways.
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

ifeq ($(V),1)
	Q =
	msg =
else
	Q = @
	msg = @printf '  %-8s %s%s\n'					\
		      "$(1)"						\
		      "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))"	\
		      "$(if $(3), $(3))";
	MAKEFLAGS += --no-print-directory
endif

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

$(call allow-override,CC,$(CROSS_COMPILE)cc)
$(call allow-override,LD,$(CROSS_COMPILE)ld)

.PHONY: all
all: $(APPS) $(INTERPS) $(OUTPUT)/ratemon_tc.bpf.o

.PHONY: clean
clean:
	$(call msg,CLEAN)
	rm -rfv $(OUTPUT) $(APPS)
	for id in `sudo bpftool struct_ops list | cut -d":" -f1`; do sudo bpftool struct_ops unregister id $$id; done
	sudo tc filter del dev $(RM_IFACE) egress || true
	sudo tc qdisc del dev $(RM_IFACE) clsact || true
	sudo rm -fv /sys/fs/bpf/flow_to_rwnd
	sudo rm -fv /sys/fs/bpf/flow_to_win_scale
	sudo rm -fv /sys/fs/bpf/flow_to_last_data_time_ns
	sudo rm -fv /sys/fs/bpf/flow_to_keepalive
	sudo rm -fv /sys/fs/bpf/done_flows
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_rwnd
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_win_scale
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_last_data_time_ns
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_keepalive
	sudo rm -fv /sys/fs/bpf/tc/globals/done_flows

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	mkdir -pv $@

# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@) \
		    INCLUDEDIR= LIBDIR= UAPIDIR=              \
		    install

# Build bpftool
$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(MAKE) ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap

# Generate vmlinux.h
$(VMLINUX):	$(BPFTOOL)
	$(call msg,GEN-VMLINUX,$@)
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

# Build BPF code
$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) ratemon_maps.h ratemon.h ratemon.bpf.h $(VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	$(CLANG) -g -O3 -target bpf -D__TARGET_ARCH_$(ARCH)                \
		     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES)                     \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Generate BPF skeletons
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(BPFTOOL) gen skeleton $< > $@

# Build user-space code
$(patsubst %,$(OUTPUT)/%.o,$(APPS)): %_main.o: %_sockops.skel.h %_structops.skel.h %_kprobe.skel.h

$(OUTPUT)/%.o: %.c $(wildcard %.h) | $(OUTPUT)
	$(call msg,CC,$@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

# Build application binary
$(APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

$(OUTPUT)/libratemon_interp.so: libratemon_interp.cpp ratemon.h constant_time_int_queue.h | $(OUTPUT)
	$(CXX) $(CXXFLAGS) -shared -fPIC -I$(OUTPUT) $< -ldl -lbpf -o $@

$(INTERPS): %: $(OUTPUT)/%.so ;

# Get the LD environment variables that will force an application binary to
# use libratemon_interp.
get_ld_var:
	$(Q) echo "LD_PRELOAD=$(OUTPUT)/libratemon_interp.so"

# Run ratemon_main and associated setup/teardown.
# Remove existing struct_ops programs, manually attach RWND tc/egress program,
# run ratemon_main, remove tc/egress program, clean up struct_ops.
attach_tc_and_run: ratemon_main $(OUTPUT)/ratemon_tc.bpf.o
	for id in `sudo bpftool struct_ops list | cut -d":" -f1`; do sudo bpftool struct_ops unregister id $$id; done
	sudo tc qdisc del dev $(RM_IFACE) clsact || true
	sudo tc qdisc add dev $(RM_IFACE) clsact
	sudo tc filter add dev $(RM_IFACE) egress bpf direct-action obj $(OUTPUT)/ratemon_tc.bpf.o sec tc/egress
	sudo bpftool map pin name flow_to_rwnd /sys/fs/bpf/flow_to_rwnd
	sudo bpftool map pin name flow_to_win_sca /sys/fs/bpf/flow_to_win_scale
	sudo bpftool map pin name flow_to_last_da /sys/fs/bpf/flow_to_last_data_time_ns
	sudo bpftool map pin name flow_to_keepali /sys/fs/bpf/flow_to_keepalive
	sudo bpftool map pin name done_flows /sys/fs/bpf/done_flows
	sudo RM_CGROUP=$(RM_CGROUP) ./ratemon_main || true
	for id in `sudo bpftool struct_ops list | cut -d":" -f1`; do sudo bpftool struct_ops unregister id $$id; done
	sudo tc filter del dev $(RM_IFACE) egress
	sudo rm -fv /sys/fs/bpf/flow_to_rwnd
	sudo rm -fv /sys/fs/bpf/flow_to_win_scale
	sudo rm -fv /sys/fs/bpf/flow_to_last_data_time_ns
	sudo rm -fv /sys/fs/bpf/flow_to_keepalive
	sudo rm -fv /sys/fs/bpf/done_flows
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_rwnd
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_win_scale
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_last_data_time_ns
	sudo rm -fv /sys/fs/bpf/tc/globals/flow_to_keepalive
	sudo rm -fv /sys/fs/bpf/tc/globals/done_flows
	sudo tc qdisc del dev $(RM_IFACE) clsact

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY:
