// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon.bpf.h"
#include "ratemon_structops.bpf.h"
// clang-format on

// vmlinux.h does not include #define macros from kernel headers.
// TCP_CONG_NEEDS_ECN is defined in include/net/tcp.h as 0x2.
#ifndef TCP_CONG_NEEDS_ECN
#define TCP_CONG_NEEDS_ECN 0x2
#endif

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// These are the regular DCTCP functions that will be called below.
// Defined in:
// https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_dctcp.c
extern void dctcp_init(struct sock *sk) __ksym;
extern uint32_t dctcp_ssthresh(struct sock *sk) __ksym;
extern void dctcp_update_alpha(struct sock *sk, uint32_t flags) __ksym;
extern void dctcp_state(struct sock *sk, uint8_t new_state) __ksym;
extern void dctcp_cwnd_event(struct sock *sk, enum tcp_ca_event event) __ksym;
extern uint32_t dctcp_cwnd_undo(struct sock *sk) __ksym;
extern void tcp_reno_cong_avoid(struct sock *sk, uint32_t ack,
                                uint32_t acked) __ksym;

// The next several functions simply delegate to the regular DCTCP
// functions, except for 'bpf_dctcp_get_info', as described below.

SEC("struct_ops/bpf_dctcp_init")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_init, struct sock *sk) { dctcp_init(sk); }

SEC("struct_ops/bpf_dctcp_ssthresh")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
uint32_t BPF_PROG(bpf_dctcp_ssthresh, struct sock *sk) {
  return dctcp_ssthresh(sk);
}

SEC("struct_ops/bpf_dctcp_update_alpha")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_update_alpha, struct sock *sk, uint32_t flags) {
  dctcp_update_alpha(sk, flags);
}

SEC("struct_ops/bpf_dctcp_cong_avoid")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_cong_avoid, struct sock *sk, uint32_t ack,
              uint32_t acked) {
  tcp_reno_cong_avoid(sk, ack, acked);
}

SEC("struct_ops/bpf_dctcp_state")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_state, struct sock *sk, uint8_t new_state) {
  dctcp_state(sk, new_state);
}

SEC("struct_ops/bpf_dctcp_undo_cwnd")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
uint32_t BPF_PROG(bpf_dctcp_undo_cwnd, struct sock *sk) {
  return dctcp_cwnd_undo(sk);
}

SEC("struct_ops/bpf_dctcp_cwnd_event")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_cwnd_event, struct sock *sk, enum tcp_ca_event event) {
  dctcp_cwnd_event(sk, event);
}

// This function is supposed to populate the tcp_cc_info struct, but instead we
// simply send a dupACK. This wakes up the sender by communicating an up-to-date
// RWND.
SEC("struct_ops/bpf_dctcp_get_info")
// trunk-ignore(clang-tidy/misc-unused-parameters)
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
void BPF_PROG(bpf_dctcp_get_info, struct sock *sk, uint32_t ext, int *attr,
              // trunk-ignore(clang-tidy/misc-unused-parameters)
              union tcp_cc_info *info) {
  rm_send_ack(sk, "bpf_dctcp_get_info");
}

SEC(".struct_ops")
struct tcp_congestion_ops bpf_dctcp = {
    // Cannot directly call dctcp_*() because they are not struct_ops
    // programs. We need the struct_ops programs above, which then immediately
    // call dctcp_*().
    .init = (void *)bpf_dctcp_init,
    .ssthresh = (void *)bpf_dctcp_ssthresh,
    .in_ack_event = (void *)bpf_dctcp_update_alpha,
    .cong_avoid = (void *)bpf_dctcp_cong_avoid,
    .set_state = (void *)bpf_dctcp_state,
    .undo_cwnd = (void *)bpf_dctcp_undo_cwnd,
    .cwnd_event = (void *)bpf_dctcp_cwnd_event,
    // This is the only program that actually does something new.
    .get_info = (void *)bpf_dctcp_get_info,
    .flags = TCP_CONG_NEEDS_ECN,
    .name = "bpf_dctcp",
};
