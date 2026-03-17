#pragma once
// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// Shared helpers for BPF struct_ops congestion control programs.
// Only include this from struct_ops .bpf.c files, since it uses
// bpf_tcp_send_ack() which is only available in struct_ops programs.

#ifndef __RATEMON_STRUCTOPS_BPF_H
#define __RATEMON_STRUCTOPS_BPF_H

#include "ratemon.bpf.h"

// Send a dupACK to wake up the sender with an up-to-date RWND.
// 'bpf_tcp_send_ack' only works in struct_ops programs.
// 'caller' is a string identifying the calling function for log messages.
static __always_inline void rm_send_ack(struct sock *sk, const char *caller) {
  if (sk == NULL) {
    RM_PRINTK("ERROR: '%s' sk=%u", caller, sk);
    return;
  }
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    RM_PRINTK("ERROR: '%s' tp=%u", caller, tp);
    return;
  }

  uint16_t skc_num = 0;
  __be16 skc_dport = 0;
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
  RM_PRINTK("INFO: '%s' sending ACK for flow %u<->%u", caller, skc_num,
            bpf_ntohs(skc_dport));
  uint64_t ret = bpf_tcp_send_ack(tp, tp->rcv_nxt);
  if (ret != 0) {
    RM_PRINTK("ERROR: '%s' failed to send ACK: %u", caller, ret);
  }
}

#endif /* __RATEMON_STRUCTOPS_BPF_H */
