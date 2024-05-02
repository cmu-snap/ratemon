// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon.h"
#include "ratemon_maps.h"
// clang-format on

// int _version SEC("version") = 1;
char LICENSE[] SEC("license") = "Dual BSD/GPL";

// // 'tcp_rcv_established' will be used to track the last time that a flow
// // received data so that we can determine when to classify a flow as idle.
// // TODO: This is not required for the initial version of scheduling that uses
// // fixed timeslices, so ratemon_main.c does not attach this program.
// SEC("kprobe/tcp_rcv_established")
// int BPF_KPROBE(tcp_rcv_established, struct sock *sk, struct sk_buff *skb) {
//   if (sk == NULL || skb == NULL) {
//     bpf_printk("ERROR: 'tcp_rcv_established' sk=%u skb=%u", sk, skb);
//     return 0;
//   }

//   // __u16 skc_num = 0;
//   // __be16 skc_dport = 0;
//   // BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
//   // BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
//   // bpf_printk("INFO: tcp_rcv_established %u->%u", skc_dport, skc_num);

//   // TODO: If this sk_buff includes new (i.e., unacked) data, then record the
//   // current time somewhere.

//   struct tcp_sock *tp = (struct tcp_sock *)(sk);
//   if (tp == NULL) {
//     bpf_printk("ERROR: 'tcp_rcv_established' tp=%u", tp);
//     return 0;
//   }
//   return 0;
// }

// The next several functions are the struct_ops programs for bpf_cubic. The all
// simply delegate to the regular tcp_cubic functions, except for
// 'bpf_cubic_get_into', as described below.

// These are the regular tcp_cubic function that will be called below.
// Defined in:
// https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_cubic.c
extern void cubictcp_init(struct sock *sk) __ksym;
extern __u32 cubictcp_recalc_ssthresh(struct sock *sk) __ksym;
extern void cubictcp_cong_avoid(struct sock *sk, __u32 ack, __u32 acked) __ksym;
extern void cubictcp_state(struct sock *sk, __u8 new_state) __ksym;
extern __u32 tcp_reno_undo_cwnd(struct sock *sk) __ksym;
extern void cubictcp_cwnd_event(struct sock *sk,
                                enum tcp_ca_event event) __ksym;
extern void cubictcp_acked(struct sock *sk,
                           const struct ack_sample *sample) __ksym;

SEC("struct_ops/bpf_cubic_init")
void BPF_PROG(bpf_cubic_init, struct sock *sk) { cubictcp_init(sk); }

SEC("struct_ops/bpf_cubic_recalc_ssthresh")
__u32 BPF_PROG(bpf_cubic_recalc_ssthresh, struct sock *sk) {
  return cubictcp_recalc_ssthresh(sk);
}

SEC("struct_ops/bpf_cubic_cong_avoid")
void BPF_PROG(bpf_cubic_cong_avoid, struct sock *sk, __u32 ack, __u32 acked) {
  cubictcp_cong_avoid(sk, ack, acked);
}

SEC("struct_ops/bpf_cubic_state")
void BPF_PROG(bpf_cubic_state, struct sock *sk, __u8 new_state) {
  cubictcp_state(sk, new_state);
}

SEC("struct_ops/bpf_cubic_undo_cwnd")
__u32 BPF_PROG(bpf_cubic_undo_cwnd, struct sock *sk) {
  return tcp_reno_undo_cwnd(sk);
}

SEC("struct_ops/bpf_cubic_cwnd_event")
void BPF_PROG(bpf_cubic_cwnd_event, struct sock *sk, enum tcp_ca_event event) {
  cubictcp_cwnd_event(sk, event);
}

SEC("struct_ops/bpf_cubic_acked")
void BPF_PROG(bpf_cubic_acked, struct sock *sk,
              const struct ack_sample *sample) {
  cubictcp_acked(sk, sample);
}

// This function is supposed to populate the tcp_cc_info struct, but instead we
// simply send a dupACK. This wakes up the sender by communicating an up-to-date
// RWND.
SEC("struct_ops/bpf_cubic_get_info")
void BPF_PROG(bpf_cubic_get_info, struct sock *sk, u32 ext, int *attr,
              union tcp_cc_info *info) {
  if (sk == NULL) {
    bpf_printk("ERROR: 'bpf_cubic_get_info' sk=%u", sk);
    return;
  }
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    bpf_printk("ERROR: 'bpf_cubic_get_info' tp=%u", tp);
    return;
  }

  // __be32 skc_daddr;
  // __be32 skc_rcv_saddr;
  // __u16 skc_num;
  // __be16 skc_dport;
  // BPF_CORE_READ_INTO(&skc_daddr, sk, __sk_common.skc_daddr);
  // BPF_CORE_READ_INTO(&skc_rcv_saddr, sk, __sk_common.skc_rcv_saddr);
  // BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  // BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
  // bpf_printk("bpf_cubic_get_info addr %u->%u", skc_daddr, skc_rcv_saddr);
  // bpf_printk("bpf_cubic_get_info port %u->%u", skc_dport, skc_num);

  // struct rm_flow flow = {.local_addr = skc_rcv_saddr,
  //                     .remote_addr = skc_daddr,
  //                     .local_port = skc_num,
  //                     .remote_port = skc_dport};
  // unsigned int rwnd = 10000;
  // bpf_map_update_elem(&flow_to_rwnd, &flow, &rwnd, BPF_ANY);

  // 'bpf_tcp_send_ack' only works in struct_ops!
  u64 ret = bpf_tcp_send_ack(tp, tp->rcv_nxt);
  if (ret != 0) {
    bpf_printk("ERROR: 'bpf_cubic_get_info' failed to send ACK: %u", ret);
    return;
  }
}

SEC(".struct_ops")
struct tcp_congestion_ops bpf_cubic = {
    // Cannot directly call cubictcp_*() because they are not struct_ops
    // programs. We need the struct_ops programs above, which then immediately
    // call cubictcp_*().
    .init = (void *)bpf_cubic_init,
    .ssthresh = (void *)bpf_cubic_recalc_ssthresh,
    .cong_avoid = (void *)bpf_cubic_cong_avoid,
    .set_state = (void *)bpf_cubic_state,
    .undo_cwnd = (void *)bpf_cubic_undo_cwnd,
    .cwnd_event = (void *)bpf_cubic_cwnd_event,
    .pkts_acked = (void *)bpf_cubic_acked,
    // This is the only program that actually does something new.
    .get_info = (void *)bpf_cubic_get_info,
    .name = "bpf_cubic",
};