/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */

#ifndef __RATEMON_H
#define __RATEMON_H

// Max number of flows that BPF can track.
#define MAX_FLOWS 8192
// Max number of flows that will be active at once.
#define MAX_ACTIVE_FLOWS 20
#define EPOCH_US 10000
// Map pin paths.
#define FLOW_TO_RWND_PIN_PATH "/sys/fs/bpf/flow_to_rwnd"
#define FLOW_TO_WIN_SCALE_PIN_PATH "/sys/fs/bpf/flow_to_win_scale"
// Name of struct_ops CCA that flows must use to be woken up.
#define BPF_CUBIC "bpf_cubic"

// Key for use in flow-based maps.
struct flow {
  unsigned int local_addr;
  unsigned int remote_addr;
  unsigned short local_port;
  unsigned short remote_port;
};

#endif /* __RATEMON_H */
