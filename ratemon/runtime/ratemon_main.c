// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <linux/limits.h>
#include <linux/sched.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "ratemon.h"
#include "ratemon.skel.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stdout, format, args);
}

// Adapted from:
// https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/cgroup_helpers.c
static int join_cgroup(const char *cgroup_path) {
  char cgroup_procs_path[PATH_MAX + 1];
  pid_t pid = getpid();
  int fd, rc = 0;

  snprintf(cgroup_procs_path, sizeof(cgroup_procs_path), "%s/cgroup.procs",
           cgroup_path);

  fd = open(cgroup_procs_path, O_WRONLY);
  if (fd < 0) {
    printf("Opening Cgroup Procs: %s\n", cgroup_procs_path);
    return 1;
  }

  if (dprintf(fd, "%d\n", pid) < 0) {
    printf("Joining Cgroup\n");
    rc = 1;
  }

  close(fd);
  return rc;
}

int main(int argc, char **argv) {
  struct ratemon_bpf *skel;
  struct bpf_link *bpf_cubic_link, *skops_link_win_scale;
  int err, ret = 0;
  const char *cgroup_path = "/test_cg";

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Open BPF application */
  skel = ratemon_bpf__open();
  if (!skel) {
    printf("Failed to open BPF skeleton\n");
    return 1;
  }

  /* Load & verify BPF programs */
  err = ratemon_bpf__load(skel);
  if (err) {
    printf("Failed to load and verify BPF skeleton\n");
    goto cleanup;
  }

  // Attach struct_ops.
  bpf_cubic_link = bpf_map__attach_struct_ops(skel->maps.bpf_cubic);
  if (bpf_cubic_link == NULL) {
    printf("Failed to attach bpf_cubic\n");
    err = -1;
    goto cleanup;
  }

  // Join cgroup for sock_ops.
  if (join_cgroup(cgroup_path) < 0) {
    printf("Failed to add PID to cgroup\n");
    err = -1;
    goto cleanup;
  }
  int cg_fd = open(cgroup_path, O_RDONLY);
  if (cg_fd <= 0) {
    printf("failed to open cgroup %s cg_fd: %d errno: %d\n", cgroup_path, cg_fd,
           errno);
    err = -1;
    goto cleanup;
  }
  printf("cgroup: %s, cg_fd: %d, pid: %d\n", cgroup_path, cg_fd, getpid());

  // Manually attach soc_kops.
  skops_link_win_scale =
      bpf_program__attach_cgroup(skel->progs.read_win_scale, cg_fd);
  if (skops_link_win_scale == NULL) {
    printf("Failed to attach read_win_scale\n");
    err = -1;
    goto cleanup;
  }
  skel->links.read_win_scale = skops_link_win_scale;

  // // Disable auto-attach for the read_win_scale sock_ops program, since we
  // // already attached it manually above.
  // bpf_program__set_autoattach(skel->progs.read_win_scale, false);
  // Attach all progs not manually attached above.
  err = ratemon_bpf__attach(skel);
  if (err) {
    printf("Failed to attach BPF skeleton\n");
    goto cleanup;
  }

  printf(
      "Successfully started! Please run `sudo cat "
      "/sys/kernel/debug/tracing/trace_pipe` "
      "to see output of the BPF programs.\n");

  while (true) {
    sleep(1);
  }

cleanup:
  ratemon_bpf__destroy(skel);
  return -err;
}
