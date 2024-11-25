// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE

#include "ratemon_test.h"

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

#include "ratemon_test.skel.h"

int cgroup_setup_and_join(const char *path);
int create_and_get_cgroup(const char *path);
unsigned long long get_cgroup_id(const char *path);
int join_cgroup(const char *path);
int setup_cgroup_environment(void);
void cleanup_cgroup_environment(void);
int test__join_cgroup(const char *path);

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stdout, format, args);
}

/*
 * To avoid relying on the system setup, when setup_cgroup_env is called
 * we create a new mount namespace, and cgroup namespace. The cgroupv2
 * root is mounted at CGROUP_MOUNT_PATH. Unfortunately, most people don't
 * have cgroupv2 enabled at this point in time. It's easier to create our
 * own mount namespace and manage it ourselves. We assume /mnt exists.
 */

#define WALK_FD_LIMIT 16

#define CGROUP_MOUNT_PATH "/mnt/cgroup-test-work-dir"

#define format_cgroup_path(buf, path)                                          \
  snprintf(buf, sizeof(buf), "%s/%s", CGROUP_MOUNT_PATH, path)

/**
 * enable_all_controllers() - Enable all available cgroup v2 controllers
 *
 * Enable all available cgroup v2 controllers in order to increase
 * the code coverage.
 *
 * If successful, 0 is returned.
 */
static int enable_all_controllers(char *cgroup_path) {
  char path[PATH_MAX + 1];
  char buf[PATH_MAX];
  char *c, *c2;
  int fd, cfd;
  ssize_t len;

  snprintf(path, sizeof(path), "%s/cgroup.controllers", cgroup_path);
  fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stdout, "Opening cgroup.controllers: %s\n", path);
    return 1;
  }

  len = read(fd, buf, sizeof(buf) - 1);
  if (len < 0) {
    close(fd);
    fprintf(stdout, "Reading cgroup.controllers: %s\n", path);
    return 1;
  }
  buf[len] = 0;
  close(fd);

  /* No controllers available? We're probably on cgroup v1. */
  if (len == 0)
    return 0;

  snprintf(path, sizeof(path), "%s/cgroup.subtree_control", cgroup_path);
  cfd = open(path, O_RDWR);
  if (cfd < 0) {
    fprintf(stdout, "Opening cgroup.subtree_control: %s\n", path);
    return 1;
  }

  for (c = strtok_r(buf, " ", &c2); c; c = strtok_r(NULL, " ", &c2)) {
    if (dprintf(cfd, "+%s\n", c) <= 0) {
      fprintf(stdout, "Enabling controller %s: %s\n", c, path);
      close(cfd);
      return 1;
    }
  }
  close(cfd);
  return 0;
}

/**
 * setup_cgroup_environment() - Setup the cgroup environment
 *
 * After calling this function, cleanup_cgroup_environment should be called
 * once testing is complete.
 *
 * This function will print an error to stderr and return 1 if it is unable
 * to setup the cgroup environment. If setup is successful, 0 is returned.
 */
int setup_cgroup_environment(void) {
  char cgroup_workdir[PATH_MAX - 24];

  format_cgroup_path(cgroup_workdir, "");

  fprintf(stdout, "cgroup_workdir: %s\n", cgroup_workdir);

  if (unshare(CLONE_NEWNS)) {
    fprintf(stdout, "unshare\n");
    return 1;
  }

  if (mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL)) {
    fprintf(stdout, "mount fakeroot\n");
    return 1;
  }

  fprintf(stdout, "CGROUP_MOUNT_PATH: %s\n", CGROUP_MOUNT_PATH);
  if (mount("none", CGROUP_MOUNT_PATH, "cgroup2", 0, NULL) && errno != EBUSY) {
    fprintf(stdout, "mount cgroup2\n");
    return 1;
  }

  /* Cleanup existing failed runs, now that the environment is setup */
  cleanup_cgroup_environment();

  if (mkdir(cgroup_workdir, 0777) && errno != EEXIST) {
    fprintf(stdout, "mkdir cgroup work dir\n");
    return 1;
  }

  if (enable_all_controllers(cgroup_workdir)) {
    fprintf(stdout, "enable_all_controllers\n");
    return 1;
  }

  return 0;
}

static int nftwfunc(const char *filename, const struct stat *statptr,
                    int fileflags, struct FTW *pfwt) {
  if ((fileflags & FTW_D) && rmdir(filename))
    fprintf(stdout, "Removing cgroup: %s\n", filename);
  return 0;
}

static int join_cgroup_from_top(const char *cgroup_path) {
  char cgroup_procs_path[PATH_MAX + 1];
  pid_t pid = getpid();
  int fd, rc = 0;

  snprintf(cgroup_procs_path, sizeof(cgroup_procs_path), "%s/cgroup.procs",
           cgroup_path);

  fd = open(cgroup_procs_path, O_WRONLY);
  if (fd < 0) {
    fprintf(stdout, "Opening Cgroup Procs: %s\n", cgroup_procs_path);
    return 1;
  }

  if (dprintf(fd, "%d\n", pid) < 0) {
    fprintf(stdout, "Joining Cgroup\n");
    rc = 1;
  }

  close(fd);
  return rc;
}

/**
 * join_cgroup() - Join a cgroup
 * @path: The cgroup path, relative to the workdir, to join
 *
 * This function expects a cgroup to already be created, relative to the cgroup
 * work dir, and it joins it. For example, passing "/my-cgroup" as the path
 * would actually put the calling process into the cgroup
 * "/cgroup-test-work-dir/my-cgroup"
 *
 * On success, it returns 0, otherwise on failure it returns 1.
 */
int join_cgroup(const char *path) {
  char cgroup_path[PATH_MAX + 1];

  format_cgroup_path(cgroup_path, path);
  return join_cgroup_from_top(cgroup_path);
}

/**
 * cleanup_cgroup_environment() - Cleanup Cgroup Testing Environment
 *
 * This is an idempotent function to delete all temporary cgroups that
 * have been created during the test, including the cgroup testing work
 * directory.
 *
 * At call time, it moves the calling process to the root cgroup, and then
 * runs the deletion process. It is idempotent, and should not fail, unless
 * a process is lingering.
 *
 * On failure, it will print an error to stderr, and try to continue.
 */
void cleanup_cgroup_environment(void) {
  char cgroup_workdir[PATH_MAX + 1];

  format_cgroup_path(cgroup_workdir, "");
  join_cgroup_from_top(CGROUP_MOUNT_PATH);
  nftw(cgroup_workdir, nftwfunc, WALK_FD_LIMIT, FTW_DEPTH | FTW_MOUNT);
}

/**
 * create_and_get_cgroup() - Create a cgroup, relative to workdir, and get the
 * FD
 * @path: The cgroup path, relative to the workdir, to join
 *
 * This function creates a cgroup under the top level workdir and returns the
 * file descriptor. It is idempotent.
 *
 * On success, it returns the file descriptor. On failure it returns -1.
 * If there is a failure, it prints the error to stderr.
 */
int create_and_get_cgroup(const char *path) {
  char cgroup_path[PATH_MAX + 1];
  int fd;

  format_cgroup_path(cgroup_path, path);

  fprintf(stdout, "Creating cgroup %s\n", cgroup_path);

  if (mkdir(cgroup_path, 0777) && errno != EEXIST) {
    fprintf(stdout, "mkdiring cgroup %s .. %s\n", path, cgroup_path);
    return -1;
  }

  fd = open(cgroup_path, O_RDONLY);
  if (fd < 0) {
    fprintf(stdout, "Opening Cgroup\n");
    return -1;
  }

  return fd;
}

/**
 * get_cgroup_id() - Get cgroup id for a particular cgroup path
 * @path: The cgroup path, relative to the workdir, to join
 *
 * On success, it returns the cgroup id. On failure it returns 0,
 * which is an invalid cgroup id.
 * If there is a failure, it prints the error to stderr.
 */
unsigned long long get_cgroup_id(const char *path) {
  int dirfd, err, flags, mount_id, fhsize;
  union {
    unsigned long long cgid;
    unsigned char raw_bytes[8];
  } id;
  char cgroup_workdir[PATH_MAX + 1];
  struct file_handle *fhp, *fhp2;
  unsigned long long ret = 0;

  format_cgroup_path(cgroup_workdir, path);

  dirfd = AT_FDCWD;
  flags = 0;
  fhsize = sizeof(*fhp);
  fhp = calloc(1, fhsize);
  if (!fhp) {
    fprintf(stdout, "calloc\n");
    return 0;
  }
  err = name_to_handle_at(dirfd, cgroup_workdir, fhp, &mount_id, flags);
  if (err >= 0 || fhp->handle_bytes != 8) {
    fprintf(stdout, "name_to_handle_at\n");
    goto free_mem;
  }

  fhsize = sizeof(struct file_handle) + fhp->handle_bytes;
  fhp2 = realloc(fhp, fhsize);
  if (!fhp2) {
    fprintf(stdout, "realloc\n");
    goto free_mem;
  }
  err = name_to_handle_at(dirfd, cgroup_workdir, fhp2, &mount_id, flags);
  fhp = fhp2;
  if (err < 0) {
    fprintf(stdout, "name_to_handle_at\n");
    goto free_mem;
  }

  memcpy(id.raw_bytes, fhp->f_handle, 8);
  ret = id.cgid;

free_mem:
  free(fhp);
  return ret;
}

int cgroup_setup_and_join(const char *path) {
  int cg_fd;

  if (setup_cgroup_environment()) {
    fprintf(stderr, "Failed to setup cgroup environment\n");
    return -EINVAL;
  }

  cg_fd = create_and_get_cgroup(path);
  if (cg_fd < 0) {
    fprintf(stderr, "Failed to create test cgroup\n");
    cleanup_cgroup_environment();
    return cg_fd;
  }

  if (join_cgroup(path)) {
    fprintf(stderr, "Failed to join cgroup\n");
    cleanup_cgroup_environment();
    return -EINVAL;
  }
  return cg_fd;
}

int test__join_cgroup(const char *path) {
  int fd;

  if (setup_cgroup_environment()) {
    fprintf(stderr, "Failed to setup cgroup environment\n");
    return -1;
  }

  fd = create_and_get_cgroup(path);
  if (fd < 0) {
    fprintf(stderr, "Failed to create cgroup '%s' (errno=%d)\n", path, errno);
    return fd;
  }

  if (join_cgroup(path)) {
    fprintf(stderr, "Failed to join cgroup '%s' (errno=%d)\n", path, errno);
    return -1;
  }

  return fd;
}

int main(int argc, char **argv) {
  struct ratemon_test_bpf *skel;
  int err, ret, iter_fd_1, iter_fd_2, iter_fd_3, iter_fd_4 = 0;
  char buf[1];
  union bpf_iter_link_info linfo1, linfo2, linfo3, linfo4;
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts1);
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts2);
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts3);
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts4);

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Open BPF application */
  skel = ratemon_test_bpf__open();
  if (!skel) {
    fprintf(stdout, "Failed to open BPF skeleton\n");
    return 1;
  }

  /* ensure BPF program only handles write() syscalls from our process */
  skel->bss->my_pid = getpid();

  /* Load & verify BPF programs */
  err = ratemon_test_bpf__load(skel);
  if (err) {
    fprintf(stdout, "Failed to load and verify BPF skeleton\n");
    goto cleanup;
  }

  unsigned long zero = 0;
  unsigned short zero_u16 = 0;
  struct tcp_sock *null_tp = NULL;
  for (unsigned int idx = 0; idx < TEST_MAP_MAX; idx++) {
    bpf_map_update_elem(bpf_map__fd(skel->maps.test_map_2), &idx, &zero,
                        BPF_ANY);
    bpf_map_update_elem(bpf_map__fd(skel->maps.test_map_3), &idx, &null_tp,
                        BPF_ANY);
    bpf_map_update_elem(bpf_map__fd(skel->maps.test_map_3a), &idx, &zero_u16,
                        BPF_ANY);
  }
  bpf_map_update_elem(bpf_map__fd(skel->maps.test_map_2_idx), &zero, &zero,
                      BPF_ANY);
  bpf_map_update_elem(bpf_map__fd(skel->maps.test_map_3_idx), &zero, &zero,
                      BPF_ANY);

  // Important: This is how to indicate which map to attach the iter to.
  // https://github.com/torvalds/linux/blob/0bbac3facb5d6cc0171c45c9873a2dc96bea9680/tools/testing/selftests/bpf/prog_tests/bpf_iter.c#L792
  // https://docs.kernel.org/bpf/bpf_iterators.html
  memset(&linfo1, 0, sizeof(linfo1));
  linfo1.map.map_fd = bpf_map__fd(skel->maps.test_map_1);
  opts1.link_info = &linfo1;
  opts1.link_info_len = sizeof(linfo1);

  // Attach the test_iter_1 program manually because the auto-attach mechanisms
  // used in ratemon_test_bpf__attach() do not work for programs of type
  // "iter/bpf_map_elem".
  struct bpf_link *link1 =
      bpf_program__attach_iter(skel->progs.test_iter_1, &opts1);
  if (link1 == NULL) {
    fprintf(stdout, "Failed to attach test_iter_1\n");
    err = -1;
    goto cleanup;
  }

  memset(&linfo2, 0, sizeof(linfo2));
  linfo2.map.map_fd = bpf_map__fd(skel->maps.test_map_2);
  opts2.link_info = &linfo2;
  opts2.link_info_len = sizeof(linfo2);
  struct bpf_link *link2 =
      bpf_program__attach_iter(skel->progs.test_iter_2, &opts2);
  if (link2 == NULL) {
    fprintf(stdout, "Failed to attach test_iter_2\n");
    err = -1;
    goto cleanup;
  }

  memset(&linfo3, 0, sizeof(linfo3));
  linfo3.map.map_fd = bpf_map__fd(skel->maps.test_map_3);
  opts3.link_info = &linfo3;
  opts3.link_info_len = sizeof(linfo3);
  struct bpf_link *link3 =
      bpf_program__attach_iter(skel->progs.test_iter_3, &opts3);
  if (link3 == NULL) {
    fprintf(stdout, "Failed to attach test_iter_3\n");
    err = -1;
    goto cleanup;
  }

  memset(&linfo4, 0, sizeof(linfo4));
  linfo4.map.map_fd = bpf_map__fd(skel->maps.flow_to_sock);
  opts4.link_info = &linfo4;
  opts4.link_info_len = sizeof(linfo4);
  struct bpf_link *link4 =
      bpf_program__attach_iter(skel->progs.test_iter_set_await, &opts4);
  if (link4 == NULL) {
    fprintf(stdout, "Failed to attach test_iter_set_await\n");
    err = -1;
    goto cleanup;
  }

  struct bpf_link *bpf_cubic_link = 0;
  bpf_cubic_link = bpf_map__attach_struct_ops(skel->maps.bpf_cubic);
  if (bpf_cubic_link == NULL) {
    fprintf(stdout, "Failed to attach bpf_cubic\n");
    err = -1;
    goto cleanup;
  }

  const char *cgroup_path = "/test_cg";
  // int cg_fd = -1;
  // if (test__join_cgroup("test_cg")) {
  //   fprintf(stdout, "Failed to add PID to cgroup\n");
  //   return -1;
  // }
  if (join_cgroup_from_top(cgroup_path) < 0) {
    fprintf(stdout, "Failed to add PID to cgroup\n");
    err = -1;
    goto cleanup;
  }
  int cg_fd = open(cgroup_path, O_RDONLY);
  if (cg_fd <= 0) {
    fprintf(stdout, "failed to open cgroup %s cg_fd: %d errno: %d\n",
            cgroup_path, cg_fd, errno);
    err = -1;
    goto cleanup;
  }
  fprintf(stdout, "cgroup: %s, cg_fd: %d, pid: %d\n", cgroup_path, cg_fd,
          getpid());
  struct bpf_link *skops_link;
  skops_link = bpf_program__attach_cgroup(skel->progs.skops_getsockopt, cg_fd);
  if (skops_link == NULL) {
    fprintf(stdout, "Failed to attach skops_getsockopt\n");
    err = -1;
    goto cleanup;
  }
  skel->links.skops_getsockopt = skops_link;

  // Disable auto-attach for the test_iter_1 program, since we already attached
  // it manually above.
  bpf_program__set_autoattach(skel->progs.test_iter_1, false);
  bpf_program__set_autoattach(skel->progs.test_iter_2, false);
  bpf_program__set_autoattach(skel->progs.test_iter_3, false);
  bpf_program__set_autoattach(skel->progs.test_iter_set_await, false);
  bpf_program__set_autoattach(skel->progs.skops_getsockopt, false);
  // bpf_program__set_autoattach(skel->progs.bpf_cubic_undo_cwnd, false);
  // Attach tracepoint handler
  err = ratemon_test_bpf__attach(skel);
  if (err) {
    fprintf(stdout, "Failed to attach BPF skeleton\n");
    goto cleanup;
  }

  iter_fd_1 = bpf_iter_create(bpf_link__fd(link1));
  if (iter_fd_1 < 0) {
    err = -1;
    fprintf(stdout, "Failed to create test_iter_1\n");
    goto cleanup;
  }
  iter_fd_2 = bpf_iter_create(bpf_link__fd(link2));
  if (iter_fd_2 < 0) {
    err = -1;
    fprintf(stdout, "Failed to create test_iter_2\n");
    goto cleanup;
  }
  iter_fd_3 = bpf_iter_create(bpf_link__fd(link3));
  if (iter_fd_3 < 0) {
    err = -1;
    fprintf(stdout, "Failed to create test_iter_3\n");
    goto cleanup;
  }
  iter_fd_4 = bpf_iter_create(bpf_link__fd(link4));
  if (iter_fd_4 < 0) {
    err = -1;
    fprintf(stdout, "Failed to create test_iter_set_await\n");
    goto cleanup;
  }

  printf("Successfully started! Please run `sudo cat "
         "/sys/kernel/debug/tracing/trace_pipe` "
         "to see output of the BPF programs.\n");

  for (;;) {
    /* trigger our BPF program */
    // fprintf(stdout, ".");

    // fprintf(stdout, "reading\n");
    while (true) {
      sleep(0.1);
      ret = read(iter_fd_1, buf, sizeof(buf));
      if (ret == 0) {
        close(iter_fd_1);
        iter_fd_1 = bpf_iter_create(bpf_link__fd(link1));
        if (iter_fd_1 < 0) {
          err = -1;
          fprintf(stdout, "Failed to recreate iter 1\n");
          goto cleanup;
        } else {
          // fprintf(stdout, "Recreated iter 1\n");
        }
        break;
      } else {
        // fprintf(stdout, "iter 1 ret: %d buf: %s\n", ret, buf);
      }
    }

    while (true) {
      sleep(0.1);
      ret = read(iter_fd_2, buf, sizeof(buf));
      if (ret == 0) {
        close(iter_fd_2);
        iter_fd_2 = bpf_iter_create(bpf_link__fd(link2));
        if (iter_fd_2 < 0) {
          err = -1;
          fprintf(stdout, "Failed to recreate iter 2\n");
          goto cleanup;
        } else {
          // fprintf(stdout, "Recreated iter 2\n");
        }
        break;
      } else {
        // fprintf(stdout, "iter 2 ret: %d buf: %s\n", ret, buf);
      }
    }

    while (true) {
      sleep(0.1);
      ret = read(iter_fd_3, buf, sizeof(buf));
      if (ret == 0) {
        close(iter_fd_3);
        iter_fd_3 = bpf_iter_create(bpf_link__fd(link3));
        if (iter_fd_3 < 0) {
          err = -1;
          fprintf(stdout, "Failed to recreate iter 3\n");
          goto cleanup;
        } else {
          // fprintf(stdout, "Recreated iter 3\n");
        }
        break;
      } else {
        // fprintf(stdout, "iter 3 ret: %d buf: %s\n", ret, buf);
      }
    }

    // while (true) {
    //   sleep(0.1);
    //   ret = read(iter_fd_4, buf, sizeof(buf));
    //   if (ret == 0) {
    //     close(iter_fd_4);
    //     iter_fd_4 = bpf_iter_create(bpf_link__fd(link4));
    //     if (iter_fd_4 < 0) {
    //       err = -1;
    //       fprintf(stdout, "Failed to recreate iter 4\n");
    //       goto cleanup;
    //     } else {
    //       fprintf(stdout, "Recreated iter 4\n");
    //     }
    //     break;
    //   } else {
    //     fprintf(stdout, "iter 4 ret: %d buf: %s\n", ret, buf);
    //   }
    // }
  }

cleanup:
  if (iter_fd_1)
    close(iter_fd_1);
  if (iter_fd_2)
    close(iter_fd_2);
  if (iter_fd_3)
    close(iter_fd_3);
  if (iter_fd_4)
    close(iter_fd_4);
  ratemon_test_bpf__destroy(skel);
  return -err;
}
