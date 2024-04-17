// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// Needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon_test.h"
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// Maps u64 time (ns) to int pid.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, TEST_MAP_MAX);
  __type(key, u64);
  __type(value, int);
} test_map_1 SEC(".maps");

// Array of u64 time (ns). Key is u32 index.
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, TEST_MAP_MAX);
  __type(key, u32);
  __type(value, u64);
} test_map_2 SEC(".maps");

// Single entry, which points to next free entry in test_map_2.
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u32);
} test_map_2_idx SEC(".maps");

// Array of struct sock pointer. Key is u32 index.
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, TEST_MAP_MAX);
  __type(key, u32);
  __type(value, struct tcp_sock *);
} test_map_3 SEC(".maps");

// Array of port numbers. Key is u32 index.
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, TEST_MAP_MAX);
  __type(key, u32);
  __type(value, u16);
} test_map_3a SEC(".maps");

// Single entry, which points to next free entry in test_map_3.
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u32);
} test_map_3_idx SEC(".maps");

int my_pid = 0;

SEC("tp/syscalls/sys_enter_write")
int handle_tp(void *ctx) {
  int pid = bpf_get_current_pid_tgid() >> 32;

  if (pid != my_pid) return 0;

  u64 t = bpf_ktime_get_ns();
  bpf_map_update_elem(&test_map_1, &t, &pid, BPF_ANY);

  u32 zero = 0;
  u32 *idx = bpf_map_lookup_elem(&test_map_2_idx, &zero);
  if (idx == NULL) {
    bpf_printk("Lookup failed!");
    return 0;
  }

  bpf_map_update_elem(&test_map_2, idx, &t, BPF_ANY);
  u32 new_idx = (*idx + 1) % TEST_MAP_MAX;
  bpf_map_update_elem(&test_map_2_idx, &zero, &new_idx, BPF_ANY);

  //   bpf_printk("BPF triggered from PID %d at %lu, idx %u.", pid, t, *idx);
  return 0;
}

SEC("iter/bpf_map_elem")
int test_iter_1(struct bpf_iter__bpf_map_elem *ctx) {
  struct seq_file *seq = ctx->meta->seq;
  u32 seq_num = ctx->meta->seq_num;

  u64 *key = ctx->key;
  int *val = ctx->value;
  if (key == NULL || val == NULL) return 0;

  //   bpf_printk("test_map_1: seq_num: %u, key %lu, value %d", seq_num, *key,
  //              *val);

  BPF_SEQ_PRINTF(seq, "1");

  return 0;
}

SEC("iter/bpf_map_elem")
int test_iter_2(struct bpf_iter__bpf_map_elem *ctx) {
  struct seq_file *seq = ctx->meta->seq;
  u32 seq_num = ctx->meta->seq_num;

  u32 *key = ctx->key;
  u64 *val = ctx->value;
  if (key == NULL || val == NULL) return 0;

  //   bpf_printk("test_map_2: seq_num: %u, key %u, value %lu", seq_num, *key,
  //              *val);

  BPF_SEQ_PRINTF(seq, "2");

  return 0;
}

// SEC("sockops")
// int grab_sock(struct bpf_sock_ops *skops) {
//   switch (skops->op) {
//     case BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB:
//       return set_hdr_cb_flags(skops);
//   }
//   return 1;
// }

// SEC("kprobe/tcp_rcv_established")
// int grab_sock(struct sock *sk) {
//   if (sk == NULL) {
//     bpf_printk("TCP rcv established return 1");
//     return 0;
//   }
//   bpf_printk("TCP rcv established receiver %u -> %u",
//   sk->__sk_common.skc_dport,
//              sk->__sk_common.skc_num);
//   // struct tcp_sock *tp;
//   // tp = (struct tcp_sock *)(sk);
//   // if (tp == NULL) {
//   //   bpf_printk("TCP rcv established return 2");
//   //   return 0;
//   // }
//   return 0;
// }

SEC("kprobe/tcp_rcv_established")
int BPF_KPROBE(tcp_rcv_established, struct sock *sk, struct sk_buff *skb) {
  if (sk == NULL || skb == NULL) {
    bpf_printk("TCP rcv established return 1");
    return 0;
  }

  __u16 skc_num = 0;
  __be16 skc_dport = 0;
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);

  struct tcp_sock *tp;
  tp = (struct tcp_sock *)(sk);
  if (tp == NULL) {
    bpf_printk("TCP rcv established return 2");
    return 0;
  }

  __u16 tcp_header_len = 0;
  BPF_CORE_READ_INTO(&tcp_header_len, tp, tcp_header_len);

  u32 zero = 0;
  u32 *idx = bpf_map_lookup_elem(&test_map_3_idx, &zero);
  if (idx == NULL) {
    bpf_printk("Lookup failed!");
    return 0;
  }

  bpf_map_update_elem(&test_map_3, idx, &tp, BPF_ANY);
  bpf_map_update_elem(&test_map_3a, idx, &tcp_header_len, BPF_ANY);
  u32 new_idx = (*idx + 1) % TEST_MAP_MAX;
  bpf_map_update_elem(&test_map_3_idx, &zero, &new_idx, BPF_ANY);

  // bpf_printk("TCP rcv established receiver %u -> %u %u", skc_dport, skc_num,
  //            tcp_header_len);
  return 0;
}

SEC("iter/bpf_map_elem")
int test_iter_3(struct bpf_iter__bpf_map_elem *ctx) {
  struct seq_file *seq = ctx->meta->seq;
  u32 seq_num = ctx->meta->seq_num;

  u32 *key = ctx->key;
  struct tcp_sock **val = ctx->value;
  if (key == NULL || val == NULL) return 0;

  u32 r = 0;
  BPF_CORE_READ_INTO(&r, *val, rcv_nxt);

  // This only works in struct_ops!
  // u64 ret = bpf_tcp_send_ack(*val, r);
  // if (ret != 0) {
  //   bpf_printk("TCP send ack failed");
  // }

  // bpf_printk("test_map_3: seq_num: %u, key %u, value %u", seq_num, *key, r);

  BPF_SEQ_PRINTF(seq, "2");

  return 0;
}

// #define max(a, b) (((a) > (b)) ? (a) : (b))

SEC("struct_ops/bpf_cubic_init")
void bpf_cubic_init(struct sock *sk) { bpf_printk("bpf_cubic_init"); }

SEC("struct_ops/bpf_cubic_recalc_ssthresh")
__u32 bpf_cubic_recalc_ssthresh(struct sock *sk) {
  bpf_printk("bpf_cubic_recalc_ssthresh");
  return 100;
  // if (sk == NULL) return 10;
  // const struct tcp_sock *tp = (struct tcp_sock *)sk;
  // u32 sc = 0;
  // BPF_CORE_READ_INTO(&sc, tp, snd_cwnd);
  // return sc;
}

SEC("struct_ops/bpf_cubic_cong_avoid")
void bpf_cubic_cong_avoid(struct sock *sk, __u32 ack, __u32 acked) {
  bpf_printk("bpf_cubic_cong_avoid");
}

SEC("struct_ops/bpf_cubic_state")
void bpf_cubic_state(struct sock *sk, __u8 new_state) {
  bpf_printk("bpf_cubic_state");
}

SEC("struct_ops/bpf_cubic_undo_cwnd")
__u32 bpf_cubic_undo_cwnd(struct sock *sk) {
  bpf_printk("bpf_cubic_undo_cwnd");
  return 100;
  // if (sk == NULL) return 10;
  // const struct tcp_sock *tp = (struct tcp_sock *)sk;
  // u32 sc = 0;
  // BPF_CORE_READ_INTO(&sc, tp, snd_cwnd);
  // return sc;
}

SEC("struct_ops/bpf_cubic_cwnd_event")
void bpf_cubic_cwnd_event(struct sock *sk, enum tcp_ca_event event) {
  bpf_printk("bpf_cubic_cwnd_event");
}

SEC("struct_ops/bpf_cubic_acked")
void bpf_cubic_acked(struct sock *sk, const struct ack_sample *sample) {
  bpf_printk("bpf_cubic_acked");
}

SEC(".struct_ops")
struct tcp_congestion_ops bpf_cubic = {
    .init = (void *)bpf_cubic_init,
    .ssthresh = (void *)bpf_cubic_recalc_ssthresh,
    .cong_avoid = (void *)bpf_cubic_cong_avoid,
    .set_state = (void *)bpf_cubic_state,
    .undo_cwnd = (void *)bpf_cubic_undo_cwnd,
    .cwnd_event = (void *)bpf_cubic_cwnd_event,
    .pkts_acked = (void *)bpf_cubic_acked,
    .name = "bpf_cubic",
};