
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>

#include "BPF.h"
#include "bcc_version.h"

const std::string BPF_PROGRAM = R"(
#include <linux/bpf.h>

BPF_TABLE_PINNED("hash", u32, u32, test_map, 1024, "/sys/fs/bpf/test_map");

BPF_ITER(bpf_map_elem) {
  struct bpf_map *map = ctx->map;
  u32 *key = ctx->key;
  u32 *val = ctx->value;
  if (key == (void *)0 || val == (void *)0) return 0;
  bpf_trace_printk("Iterating %d %d\n", key, val);
  return 0;
}
)";

int main() {
  ebpf::BPF bpf;
  auto res = bpf.init(BPF_PROGRAM);
  if (!res.ok()) {
    std::cerr << res.msg() << std::endl;
    return 1;
  }

  int prog_fd;
  res = bpf.load_func("bpf_iter__bpf_map_elem", BPF_PROG_TYPE_TRACING, prog_fd);
  if (!res.ok()) {
    std::cerr << res.msg() << std::endl;
    return 1;
  }
  std::cerr << "prog_fd: " << prog_fd << std::endl;

  int link_fd = bcc_iter_attach(prog_fd, NULL, 0);
  if (link_fd < 0) {
    std::cerr << "bcc_iter_attach failed: " << link_fd << std::endl;
    switch (link_fd) {
    case -ENOMSG:
      std::cerr << "ENOMSG " << link_fd << std::endl;
      break;
    case -EINVAL:
      std::cerr << "EINVAL " << link_fd << std::endl;
      break;
    case -ENOENT:
      std::cerr << "ENOENT " << link_fd << std::endl;
      break;
    case -ENOSPC:
      std::cerr << "ENOSPC " << link_fd << std::endl;
      break;
    case -EFAULT:
      std::cerr << "EFAULT " << link_fd << std::endl;
      break;
    case -EPERM:
      std::cerr << "EPERM " << link_fd << std::endl;
      break;
    case -EAGAIN:
      std::cerr << "EAGAIN " << link_fd << std::endl;
      break;
    case -EBADF:
      std::cerr << "EBADF " << link_fd << std::endl;
      break;
    default:
      std::cerr << "unknown " << link_fd << std::endl;
    }
    return 1;
  }

  int iter_fd = bcc_iter_create(link_fd);
  if (iter_fd < 0) {
    std::cerr << "bcc_iter_create failed: " << iter_fd << std::endl;
    close(link_fd);
    return 1;
  }

  int len = 0;
  uint32_t dest = 0;
  while ((len = read(iter_fd, (char *)&dest, sizeof(dest)))) {
    if (len < 0) {
      if (len == -EAGAIN)
        continue;
      std::cerr << "read failed: " << len << std::endl;
      break;
    }
  }

  close(iter_fd);
  close(link_fd);
  return 0;
}
