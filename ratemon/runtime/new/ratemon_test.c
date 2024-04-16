// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)

#include "ratemon_test.h"

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#include "ratemon_test.skel.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stdout, format, args);
}

int main(int argc, char **argv) {
  struct ratemon_test_bpf *skel;
  int err, ret, iter_fd_1, iter_fd_2, iter_fd_3;
  char buf[1];
  union bpf_iter_link_info linfo1, linfo2, linfo3;
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts1);
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts2);
  DECLARE_LIBBPF_OPTS(bpf_iter_attach_opts, opts3);

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

  // Disable auto-attach for the test_iter_1 program, since we already attached
  // it manually above.
  bpf_program__set_autoattach(skel->progs.test_iter_1, false);
  bpf_program__set_autoattach(skel->progs.test_iter_2, false);
  bpf_program__set_autoattach(skel->progs.test_iter_3, false);
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

  printf(
      "Successfully started! Please run `sudo cat "
      "/sys/kernel/debug/tracing/trace_pipe` "
      "to see output of the BPF programs.\n");

  // while ((err = read(iter_fd, &buf, sizeof(buf))) == -1 &&
  // 		errno == EAGAIN) {
  // 	fprintf(stdout, "reading\n");
  // }

  for (;;) {
    /* trigger our BPF program */
    fprintf(stdout, ".");

    fprintf(stdout, "reading\n");
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
          fprintf(stdout, "Recreated iter 1\n");
        }
        break;
      } else {
        fprintf(stdout, "iter 1 ret: %d buf: %s\n", ret, buf);
      }
      // if (ret < 0) {
      // 	if (errno == EAGAIN) {
      // 		fprintf(stdout, "No more iter\n");
      // 		continue;
      // 	}
      // 	err = -errno;
      // 	break;
      // }
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
          fprintf(stdout, "Recreated iter 2\n");
        }
        break;
      } else {
        fprintf(stdout, "iter 2 ret: %d buf: %s\n", ret, buf);
      }
      // if (ret < 0) {
      // 	if (errno == EAGAIN) {
      // 		fprintf(stdout, "No more iter\n");
      // 		continue;
      // 	}
      // 	err = -errno;
      // 	break;
      // }
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
          fprintf(stdout, "Recreated iter 3\n");
        }
        break;
      } else {
        fprintf(stdout, "iter 3 ret: %d buf: %s\n", ret, buf);
      }
      // if (ret < 0) {
      // 	if (errno == EAGAIN) {
      // 		fprintf(stdout, "No more iter\n");
      // 		continue;
      // 	}
      // 	err = -errno;
      // 	break;
      // }
    }
  }

cleanup:
  close(iter_fd_1);
  close(iter_fd_2);
  close(iter_fd_3);
  ratemon_test_bpf__destroy(skel);
  return -err;
}
