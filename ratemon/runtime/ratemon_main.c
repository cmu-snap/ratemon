// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include "ratemon.h"
#include "ratemon.skel.h"

// Signals whether the program should continue running.
static volatile bool run = true;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stdout, format, args);
}

// Catch SIGINT and trigger the main function to end.
void sigint_handler(int dummy) { run = false; }

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
    printf("ERROR in opening cgroup procs file: %s\n", cgroup_procs_path);
    return 1;
  }

  if (dprintf(fd, "%d\n", pid) < 0) {
    printf("ERROR in adding PID %d to cgroup: %s\n", pid, cgroup_path);
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

  // Catch SIGINT to end the program.
  signal(SIGINT, sigint_handler);

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Open BPF application */
  skel = ratemon_bpf__open();
  if (!skel) {
    printf("ERROR when opening BPF skeleton\n");
    return 1;
  }

  /* Load & verify BPF programs */
  err = ratemon_bpf__load(skel);
  if (err) {
    printf("ERROR when loading and verifying BPF skeleton\n");
    goto cleanup;
  }

  // Attach struct_ops.
  bpf_cubic_link = bpf_map__attach_struct_ops(skel->maps.bpf_cubic);
  if (bpf_cubic_link == NULL) {
    printf("ERROR when attaching bpf_cubic\n");
    err = -1;
    goto cleanup;
  }

  // Join cgroup for sock_ops.
  if (join_cgroup(cgroup_path) < 0) {
    printf("ERROR when joining cgroup: %s\n", cgroup_path);
    err = -1;
    goto cleanup;
  }
  int cg_fd = open(cgroup_path, O_RDONLY);
  if (cg_fd <= 0) {
    printf("ERROR when opening cgroup: %s cg_fd: %d errno: %d\n", cgroup_path,
           cg_fd, errno);
    err = -1;
    goto cleanup;
  }
  printf("Opened cgroup: %s, cg_fd: %d, pid: %d\n", cgroup_path, cg_fd,
         getpid());

  // Manually attach soc_kops.
  skops_link_win_scale =
      bpf_program__attach_cgroup(skel->progs.read_win_scale, cg_fd);
  if (skops_link_win_scale == NULL) {
    printf("ERROR when attaching read_win_scale\n");
    err = -1;
    goto cleanup;
  }
  skel->links.read_win_scale = skops_link_win_scale;

  // Attach all progs not manually attached above.
  err = ratemon_bpf__attach(skel);
  if (err) {
    printf("ERROR when attach BPF skeleton\n");
    goto cleanup;
  }

  printf(
      "BPF programs running. "
      "Progress: `sudo cat /sys/kernel/debug/tracing/trace_pipe`. "
      "Ctrl-C to end.\n");

  // Wait for Ctrl-C.
  while (run) sleep(1);

cleanup:
  printf("Destroying BPF programs\n");
  // bpf_link__detach_struct_ops(bpf_cubic_link);
  ratemon_bpf__destroy(skel);
  return -err;
}
