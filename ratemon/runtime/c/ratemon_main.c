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
#include <string.h>
#include <unistd.h>

#include "ratemon.h"
#include "ratemon_kprobe.skel.h"
#include "ratemon_sockops.skel.h"
#include "ratemon_structops.skel.h"

// Signals whether the program should continue running.
static volatile bool run = true;

struct bpf_link *structops_link;
struct ratemon_sockops_bpf *sockops_skel;
struct ratemon_structops_bpf *structops_skel;
struct ratemon_kprobe_bpf *kprobe_skel;

// Existing signal handler for SIGINT.
struct sigaction oldact;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stdout, format, args);
}

// Catch SIGINT and trigger the main function to end.
void sigint_handler(int signum) {
  switch (signum) {
    case SIGINT:
      printf("INFO: caught SIGINT\n");
      run = false;
      printf("Resetting old SIGINT handler\n");
      sigaction(SIGINT, &oldact, NULL);
      break;
    default:
      printf("ERROR: caught signal %d\n", signum);
      break;
  }
  printf("INFO: re-raising signal %d\n", signum);
  raise(signum);
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

int prepare_sockops(char *cg_path) {
  // Open skeleton and load programs and maps.
  sockops_skel = ratemon_sockops_bpf__open_and_load();
  if (!sockops_skel) {
    printf("ERROR: failed to open/load 'ratemon_sockops' BPF skeleton\n");
    return 1;
  }
  // Join cgroup for sock_ops.
  if (join_cgroup(cg_path) < 0) {
    printf("ERROR: failed to join cgroup: %s\n", cg_path);
    return 1;
  }
  int cg_fd = open(cg_path, O_RDONLY);
  if (cg_fd <= 0) {
    printf("ERROR: failed to open cgroup: %s cg_fd: %d errno: %d\n", cg_path,
           cg_fd, errno);
    return 1;
  }
  printf("INFO: opened cgroup: %s, cg_fd: %d, pid: %d\n", cg_path, cg_fd,
         getpid());
  // Attach sock_ops.
  struct bpf_link *skops_link_win_scale =
      bpf_program__attach_cgroup(sockops_skel->progs.read_win_scale, cg_fd);
  if (skops_link_win_scale == NULL) {
    printf("ERROR: failed to attach 'read_win_scale'\n");
    return 1;
  }
  sockops_skel->links.read_win_scale = skops_link_win_scale;
  return 0;
}

int prepare_structops() {
  // Open skeleton and load programs and maps.
  structops_skel = ratemon_structops_bpf__open_and_load();
  if (!structops_skel) {
    printf("ERROR: failed to open/load 'ratemon_structops' BPF skeleton\n");
    return 1;
  }
  // Attach struct_ops.
  structops_link = bpf_map__attach_struct_ops(structops_skel->maps.bpf_cubic);
  if (structops_link == NULL) {
    printf("ERROR: failed to attach 'bpf_cubic'\n");
    return 1;
  }
  return 0;
}

int prepare_kprobe() {
  // Open skeleton and load programs and maps.
  kprobe_skel = ratemon_kprobe_bpf__open_and_load();
  if (!kprobe_skel) {
    printf("ERROR: failed to open/load 'ratemon_kprobe' BPF skeleton\n");
    return 1;
  }
  // Attach kprobe.
  if (ratemon_kprobe_bpf__attach(kprobe_skel)) {
    printf("ERROR: failed to attach 'ratemon_kprobe'\n");
    return 1;
  }
  return 0;
}

bool read_env_str(const char *key, char *dest) {
  // Read an environment variable a char *.
  char *val_str = getenv(key);
  if (val_str == NULL) {
    printf("ERROR: failed to query environment variable '%s'\n", key);
    return false;
  }
  strcpy(dest, val_str);
  return true;
}

int main(int argc, char **argv) {
  // Catch SIGINT to end the program.
  struct sigaction action;
  action.sa_handler = sigint_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, &oldact);

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  char cg_path[1024];
  if (!read_env_str(RM_CGROUP_KEY, cg_path)) {
    printf("ERROR: failed to read cgroup path\n");
    goto cleanup;
  }

  if (prepare_sockops(cg_path)) {
    printf("ERROR: failed to set up sockops\n");
    goto cleanup;
  }
  if (prepare_structops()) {
    printf("ERROR: failed to set up structops\n");
    goto cleanup;
  }
  if (prepare_kprobe()) {
    printf("ERROR: failed to set up kprobe\n");
    goto cleanup;
  }

  printf(
      "INFO: BPF programs running. "
      "Progress: `sudo cat /sys/kernel/debug/tracing/trace_pipe`. "
      "Ctrl-C to end.\n");

  // Wait for Ctrl-C.
  while (run) sleep(1);

cleanup:
  printf("Destroying BPF programs\n");
  // bpf_tc_detach(&hook, &tc_opts);
  // bpf_tc_hook_destroy(&hook);
  // bpf_map__unpin(skel->maps.flow_to_rwnd, NULL);
  // bpf_map__unpin(skel->maps.flow_to_win_scale, NULL);
  bpf_link__destroy(structops_link);
  ratemon_sockops_bpf__destroy(sockops_skel);
  ratemon_structops_bpf__destroy(structops_skel);
  ratemon_kprobe_bpf__destroy(kprobe_skel);
  return 0;
}
