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

#define SOL_SOCKET 1
#define SO_KEEPALIVE 9
#define SOL_TCP 6
#define TCP_KEEPIDLE 4
#define TCP_INFO 11

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

struct flow {
  u32 local_addr;
  u32 remote_addr;
  u16 local_port;
  u16 remote_port;
};

// Need this, because for some reason vmlinux.h does not define bpf_timer.
struct bpf_timer {
  __u64 __opaque[2];
};

struct timer_elem {
  struct bpf_timer timer;
};

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, struct flow);
  __type(value, struct timer_elem);
} timer_map SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, struct flow);
  __type(value, struct sock *);
} flow_to_sock SEC(".maps");

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

SEC("sockops")
int skops_getsockopt(struct bpf_sock_ops *skops) {
  bpf_printk("skops_getsockops");
  switch (skops->op) {
    case BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB:
    case BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB: {
      struct tcp_info info;
      bpf_getsockopt(skops, SOL_TCP, TCP_INFO, &info, sizeof(info));
      bpf_printk("tcp_info snd_cwnd %u", info.tcpi_snd_cwnd);
    }
  }
  return 1;
}

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
void BPF_PROG(bpf_cubic_init, struct sock *sk) {
  bpf_printk("bpf_cubic_init");
  if (sk == NULL) {
    bpf_printk("sk is null");
    return;
  }

  int optval_enable = 1;
  if (bpf_setsockopt(sk, SOL_SOCKET, SO_KEEPALIVE, &optval_enable,
                     sizeof(optval_enable))) {
    bpf_printk("Failed setting SO_KEEPALIVE");
  }
  int optval_time = 1;
  if (bpf_setsockopt(sk, SOL_TCP, TCP_KEEPIDLE, &optval_time,
                     sizeof(optval_time))) {
    bpf_printk("Failed setting TCP_KEEPIDLE");
  }
  bpf_printk("Configured SO_KEEPALIVE and TCP_KEEPIDLE");

  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    bpf_printk("tp is null");
    return;
  }

  u32 keepalive_time = 0;
  BPF_CORE_READ_INTO(&keepalive_time, tp, keepalive_time);
  bpf_printk("tp->keepalive_time %u", keepalive_time);
}

SEC("struct_ops/bpf_cubic_recalc_ssthresh")
__u32 BPF_PROG(bpf_cubic_recalc_ssthresh, struct sock *sk) {
  bpf_printk("bpf_cubic_recalc_ssthresh");
  return 100;
  // if (sk == NULL) return 10;
  // const struct tcp_sock *tp = (struct tcp_sock *)sk;
  // u32 sc = 0;
  // BPF_CORE_READ_INTO(&sc, tp, snd_cwnd);
  // return sc;
}

SEC("struct_ops/bpf_cubic_cong_avoid")
void BPF_PROG(bpf_cubic_cong_avoid, struct sock *sk, __u32 ack, __u32 acked) {
  bpf_printk("bpf_cubic_cong_avoid");
}

SEC("struct_ops/bpf_cubic_state")
void BPF_PROG(bpf_cubic_state, struct sock *sk, __u8 new_state) {
  bpf_printk("bpf_cubic_state");
}

SEC("struct_ops/bpf_cubic_undo_cwnd")
__u32 BPF_PROG(bpf_cubic_undo_cwnd, struct sock *sk) {
  bpf_printk("bpf_cubic_undo_cwnd");
  return 100;
  // if (sk == NULL) return 10;
  // const struct tcp_sock *tp = (struct tcp_sock *)sk;
  // u32 sc = 0;
  // BPF_CORE_READ_INTO(&sc, tp, snd_cwnd);
  // return sc;
}

SEC("struct_ops/bpf_cubic_cwnd_event")
void BPF_PROG(bpf_cubic_cwnd_event, struct sock *sk, enum tcp_ca_event event) {
  bpf_printk("bpf_cubic_cwnd_event");

  if (sk == NULL) {
    return;
  }
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    return;
  }

  u32 rcv_nxt = 0;
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);

  u64 ret = bpf_tcp_send_ack(sk, rcv_nxt);
  if (ret != 0) {
    bpf_printk("TCP send ack failed");
  }
}

static int timer_cb1(void *map, struct flow *key, struct bpf_timer *timer) {
  bpf_printk("timer_cb1");

  if (key == NULL) {
    bpf_printk("timer_cb1 key is null");
    return 0;
  }

  struct sock **skp = bpf_map_lookup_elem(&flow_to_sock, key);
  if (skp == NULL) {
    bpf_printk("timer_cb1 skp is null");
    return 0;
  }
  struct sock *sk = *skp;
  if (sk == NULL) {
    bpf_printk("timer_cb1 sk is null");
    return 0;
  }
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    bpf_printk("timer_cb1 tp is null");
    return 0;
  }

  u32 rcv_nxt = 0;
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);

  if (rcv_nxt == 0) {
    bpf_printk("rcv_nxt is 0");
    return 0;
  }

  //   /*
  //  * bpf_kptr_xchg
  //  *
  //  * 	Exchange kptr at pointer *map_value* with *ptr*, and return the
  //  * 	old value. *ptr* can be NULL, otherwise it must be a referenced
  //  * 	pointer which will be released when this helper is called.
  //  *
  //  * Returns
  //  * 	The old value of kptr (which can be NULL). The returned pointer
  //  * 	if not NULL, is a reference which must be released using its
  //  * 	corresponding release function, or moved into a BPF map before
  //  * 	program exit.
  //  */
  // static void *(*bpf_kptr_xchg)(void *map_value, void *ptr) = (void *) 194;

  // struct sock *sk2;
  // struct sock *sk3 = bpf_kptr_xchg(sk, sk2);
  // if (!sk3) {
  //   return 0;
  // }

  // struct tcp_sock *tp3 = (struct tcp_sock *)sk3;
  // if (tp3 == NULL) {
  //   bpf_printk("tp3 is null");
  //   return 0;
  // }

  // // bpf_sk_lookup_tcp is not available in struct_ops
  // struct bpf_sock_tuple tuple = {};
  // tuple.ipv4.saddr = bpf_htons(key->local_addr);
  // tuple.ipv4.daddr = bpf_htons(key->remote_addr);
  // tuple.ipv4.sport = bpf_htons(key->local_port);
  // tuple.ipv4.dport = bpf_htons(key->remote_port);
  // struct bpf_sock *bpf_skc = bpf_sk_lookup_tcp(sk, &tuple,
  // sizeof(tuple.ipv4), -1, 0); if (bpf_skc == NULL) {
  //   bpf_printk("bpf_skc is null");
  //   return 0;
  // }
  // struct bpf_tcp_sock *bpf_tp = bpf_tcp_sock(bpf_skc);
  // if (bpf_tp == NULL) {
  //   bpf_printk("bpf_tp is null");
  //   return 0;
  // }

  // // TODO: Is there a way to get the sk other than passing
  // // it through a map? One of the BPF helpers?
  // u64 ret = bpf_tcp_send_ack(sk, rcv_nxt);
  // if (ret != 0) {
  //   bpf_printk("TCP send ack failed");
  // }

  // bpf_map_update_elem(&flow_to_sock, key, &sk3, BPF_ANY);

  bpf_printk("timer_cb1 rcv_nxt %u", rcv_nxt);
  return 0;
}

#define CLOCK_MONOTONIC 1

SEC("struct_ops/bpf_cubic_acked")
void BPF_PROG(bpf_cubic_acked, struct sock *sk,
              const struct ack_sample *sample) {
  bpf_printk("bpf_cubic_acked 1");

  if (sk == NULL) {
    return;
  }

  __be32 skc_rcv_saddr = 0;
  __be32 skc_daddr = 0;
  __u16 skc_num = 0;
  __be16 skc_dport = 0;
  BPF_CORE_READ_INTO(&skc_rcv_saddr, sk, __sk_common.skc_rcv_saddr);
  BPF_CORE_READ_INTO(&skc_daddr, sk, __sk_common.skc_daddr);
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);

  struct bpf_timer *t;
  struct flow timer_map_key = {};
  timer_map_key.local_addr = bpf_ntohs(skc_rcv_saddr);
  timer_map_key.remote_addr = bpf_ntohs(skc_daddr);
  timer_map_key.local_port = skc_num;
  timer_map_key.remote_port = bpf_ntohs(skc_dport);

  // Put the sock in the flow_to_sock map so that we can fetch it from the timer
  // callback.
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    return;
  }
  bpf_map_update_elem(&flow_to_sock, &timer_map_key, &sk, BPF_ANY);

  bpf_printk("bpf_cubic_acked 2");

  t = bpf_map_lookup_elem(&timer_map, &timer_map_key);
  if (t == NULL) {
    struct bpf_timer new_t = {};
    bpf_map_update_elem(&timer_map, &timer_map_key, &new_t, BPF_ANY);
    t = bpf_map_lookup_elem(&timer_map, &timer_map_key);
  }
  if (t == NULL) {
    bpf_printk("t is still NULL");
    return;
  }

  bpf_printk("bpf_cubic_acked 3");

  bpf_timer_init(t, &timer_map, CLOCK_MONOTONIC);

  bpf_printk("bpf_cubic_acked 3");

  bpf_timer_set_callback(t, timer_cb1);
  bpf_timer_start(t, 0 /* call timer_cb1 asap */, 0);

  u32 rcv_nxt = 0;
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);

  u64 ret = bpf_tcp_send_ack(sk, rcv_nxt);
  if (ret != 0) {
    bpf_printk("TCP send ack failed");
  }

  bpf_printk("bpf_cubic_acked 4");
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

SEC("iter/bpf_map_elem")
int test_iter_set_await(struct bpf_iter__bpf_map_elem *ctx) {
  struct seq_file *seq = ctx->meta->seq;
  u32 seq_num = ctx->meta->seq_num;

  struct flow *key = ctx->key;
  struct sock **val = ctx->value;
  if (key == NULL || val == NULL) {
    bpf_printk("key or val is null");
    return 0;
  }
  struct sock *sk = *val;
  if (sk == NULL) {
    bpf_printk("sk is null");
    return 0;
  }

  // // Memory access error from verifier
  // u32 await = 1;
  // sk->awaiting_wakeup = await;
  // bpf_printk("Set await");

  // // bpf_getsockopt not available in iter
  // struct tcp_info info;
  // bpf_getsockopt(sk, SOL_TCP, TCP_INFO, &info, sizeof(info));

  BPF_SEQ_PRINTF(seq, "4");
  return 0;
}

#define from_timer(var, callback_timer, timer_fieldname) \
  container_of(callback_timer, typeof(*var), timer_fieldname)

SEC("kprobe/tcp_keepalive_timer")
int BPF_KPROBE(tcp_keepalive_timer, struct timer_list *t) {
  bpf_printk("KROBE tcp_keepalive_timer");
  if (t == NULL) {
    return 0;
  }
  struct sock *sk = from_timer(sk, t, sk_timer);
  if (sk == NULL) {
    return 0;
  }

  u32 await;
  sk->awaiting_wakeup = await;

  bpf_printk("Set await");

  return 0;
}