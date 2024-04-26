/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */

#ifndef __RATEMON_H
#define __RATEMON_H

#define MAX_FLOWS 8192

// Map pin paths.
#define FLOW_TO_RWND_PIN_PATH "/sys/fs/bpf/flow_to_rwnd"
#define FLOW_TO_WIN_SCALE_PIN_PATH "/sys/fs/bpf/flow_to_win_scale"

#define BPF_CUBIC "bpf_cubic"

// Key for use in flow-based maps.
struct flow {
  unsigned int local_addr;
  unsigned int remote_addr;
  unsigned short local_port;
  unsigned short remote_port;
};

#endif /* __RATEMON_H */
