// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE

#include <argp.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <linux/types.h>
#include <net/if.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include "ratemon.h"
#include "ratemon.skel.h"

// Signals whether the program should continue running.
static volatile bool run = true;

// Which interface to attach the 'do_rwnd_at_egress' program to.
// const char *ifname = "eno4";

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
  int err = 0;
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

  // Pin maps so they can be reused by libratemon_interp.
  err = bpf_map__pin(skel->maps.flow_to_rwnd, RM_FLOW_TO_RWND_PIN_PATH);
  if (err) {
    printf("ERROR when pinning map flow_to_rwnd at: %s\n",
           RM_FLOW_TO_RWND_PIN_PATH);
    goto cleanup;
  }
  err =
      bpf_map__pin(skel->maps.flow_to_win_scale, RM_FLOW_TO_WIN_SCALE_PIN_PATH);
  if (err) {
    printf("ERROR when pinning map flow_to_win_scale at: %s\n",
           RM_FLOW_TO_WIN_SCALE_PIN_PATH);
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

  // Manually attach sock_ops.
  skops_link_win_scale =
      bpf_program__attach_cgroup(skel->progs.read_win_scale, cg_fd);
  if (skops_link_win_scale == NULL) {
    printf("ERROR when attaching read_win_scale\n");
    err = -1;
    goto cleanup;
  }
  skel->links.read_win_scale = skops_link_win_scale;

  // // Set up tc egress.
  // // We will attach this program manually.
  // bpf_program__set_autoattach(skel->progs.do_rwnd_at_egress, false);

  // TODO: The following code gives an error in bpf_tc_hook_create(). What is
  // the right parent? Do we need to create a particular qdisc ahead of time?
  // The selftests are not clear. See:
  // https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/prog_tests/tc_bpf.c

  // unsigned int ifindex = if_nametoindex(ifname);
  // // Need to create a hook and attach the tc prog to it
  // // First, create the hook.
  // unsigned int parent = BPF_TC_PARENT(1, 0);
  // DECLARE_LIBBPF_OPTS(bpf_tc_hook, hook, .ifindex = ifindex,
  //                     .attach_point = BPF_TC_EGRESS, .parent = parent);
  // printf("parent: %u\n", parent);
  // err = bpf_tc_hook_create(&hook);
  // if (err) {
  //   printf("ERROR when creating TC hook\n");
  //   goto cleanup;
  // }
  // int fd = bpf_program__fd(skel->progs.do_rwnd_at_egress);
  // if (!fd) {
  //   printf("ERROR when looking up FD for 'do_rwnd_at_egress'\n");
  //   goto cleanup;
  // }
  // DECLARE_LIBBPF_OPTS(bpf_tc_opts, tc_opts, .handle = 1, .priority = 1,
  //                     .prog_fd = fd);
  // struct bpf_prog_info info = {};
  // __u32 info_len = sizeof(info);
  // err = bpf_obj_get_info_by_fd(fd, &info, &info_len);
  // if (err) {
  //   printf("ERROR when getting 'do_rwnd_at_egress' info\n");
  //   goto cleanup;
  // }
  // err = bpf_tc_attach(&hook, &tc_opts);
  // if (err) {
  //   printf("ERROR when attaching 'do_rwnd_at_egress'\n");
  //   goto cleanup;
  // }

  // // Attach all programs that were not manually attached above.
  // err = ratemon_bpf__attach(skel);
  // if (err) {
  //   printf("ERROR when attach BPF skeleton\n");
  //   goto cleanup;
  // }

  printf(
      "BPF programs running. "
      "Progress: `sudo cat /sys/kernel/debug/tracing/trace_pipe`. "
      "Ctrl-C to end.\n");

  // Wait for Ctrl-C.
  while (run) sleep(1);

cleanup:
  printf("Destroying BPF programs\n");
  // bpf_tc_detach(&hook, &tc_opts);
  // bpf_tc_hook_destroy(&hook);
  bpf_map__unpin(skel->maps.flow_to_rwnd, NULL);
  bpf_map__unpin(skel->maps.flow_to_win_scale, NULL);
  ratemon_bpf__destroy(skel);
  return -err;
}
