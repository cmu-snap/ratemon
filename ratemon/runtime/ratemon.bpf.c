// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon.h"
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// BPF SOCK_OPS program return codes.
#define SOCKOPS_OK 1
#define SOCKOPS_ERR 0

#define TC_ACT_OK 0

// TCP header flags.
#define TCPHDR_SYN 0x02
#define TCPHDR_ACK 0x10
#define TCPHDR_SYNACK (TCPHDR_SYN | TCPHDR_ACK)

#define ETH_P_IP 0x0800 /* Internet Protocol packet	*/
#define TCPOPT_WINDOW 3 /* Window scaling */
#define AF_INET 2       /* Internet IP Protocol 	*/

#define min(x, y) ((x) < (y) ? (x) : (y))

// Read RWND limit for flow, as set by userspace. Even though the advertised
// window is only 16 bits in the TCP header, use 32 bits here because we have
// not taken window scaling into account yet.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, MAX_FLOWS);
  __type(key, struct flow);
  __type(value, unsigned int);
  // __type(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_rwnd SEC(".maps");

// Learn window scaling factor for each flow.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, MAX_FLOWS);
  __type(key, struct flow);
  __type(value, unsigned char);
} flow_to_win_scale SEC(".maps");

struct tcp_opt {
  __u8 kind;
  __u8 len;
  __u8 data;
} __attribute__((packed));

SEC("kprobe/tcp_rcv_established")
int BPF_KPROBE(tcp_rcv_established, struct sock *sk, struct sk_buff *skb) {
  if (sk == NULL || skb == NULL) {
    bpf_printk("ERROR tcp_rcv_established sk=%u skb=%u", sk, skb);
    return 0;
  }

  // __u16 skc_num = 0;
  // __be16 skc_dport = 0;
  // BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  // BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
  // bpf_printk("tcp_rcv_established %u->%u", skc_dport, skc_num);

  struct tcp_sock *tp = (struct tcp_sock *)(sk);
  if (tp == NULL) {
    bpf_printk("ERROR tcp_rcv_established tp=%u", tp);
    return 0;
  }
  return 0;
}

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
  bpf_printk("bpf_cubic_acked 1");
  cubictcp_acked(sk, sample);
}

SEC("struct_ops/bpf_cubic_get_info")
void BPF_PROG(bpf_cubic_get_info, struct sock *sk, u32 ext, int *attr,
              union tcp_cc_info *info) {
  if (sk == NULL) {
    bpf_printk("ERROR bpf_cubic_get_info sk=%u", sk);
    return;
  }
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    bpf_printk("ERROR bpf_cubic_get_info tp=%u", tp);
    return;
  }

  __u16 skc_num = 0;
  __be16 skc_dport = 0;
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
  bpf_printk("bpf_cubic_get_info %u->%u", skc_dport, skc_num);

  struct flow flow = {.local_addr = 0,
                      .remote_addr = 0,
                      .local_port = skc_num,
                      .remote_port = skc_dport};
  unsigned int rwnd = 10000;
  bpf_map_update_elem(&flow_to_rwnd, &flow, &rwnd, BPF_ANY);

  // This only works in struct_ops!
  u64 ret = bpf_tcp_send_ack(tp, tp->rcv_nxt);
  if (ret != 0) {
    bpf_printk("ERROR bpf_cubic_get_info bpf_tcp_send_ack()=%u", ret);
  } else {
    bpf_printk("bpf_cubic_get_info SENT ACK!!!");
  }
}

SEC(".struct_ops")
struct tcp_congestion_ops bpf_cubic = {
    // Cannot directly use cubictcp_*() because they are not struct_ops
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

// Inspired by:
// https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
SEC("tc/egress")
int do_rwnd_at_egress(struct __sk_buff *skb) {
  if (skb == NULL) {
    return TC_ACT_OK;
  }

  void *data = (void *)(long)skb->data;
  void *data_end = (void *)(long)skb->data_end;
  struct ethhdr *eth = data;
  struct iphdr *ip;
  struct tcphdr *tcp;

  // Do a sanity check to make sure that the IP header does not extend past
  // the end of the packet.
  if ((void *)eth + sizeof(*eth) > data_end) {
    return TC_ACT_OK;
  }

  // Check that this is IP. Calculate the start of the IP header, and do a
  // sanity check to make sure that the IP header does not extend past the
  // end of the packet.
  if (eth->h_proto != bpf_htons(ETH_P_IP)) {
    return TC_ACT_OK;
  }
  ip = data + sizeof(*eth);
  if ((void *)ip + sizeof(*ip) > data_end) {
    return TC_ACT_OK;
  }

  // Similar for TCP header.
  if (ip->protocol != IPPROTO_TCP) {
    return TC_ACT_OK;
  }
  tcp = (void *)ip + sizeof(*ip);
  if ((void *)tcp + sizeof(*tcp) > data_end) {
    return TC_ACT_OK;
  }

  // Prepare the lookup key.
  struct flow flow;
  flow.local_addr = ip->saddr;
  flow.remote_addr = ip->daddr;
  flow.local_port = bpf_ntohs(tcp->source);
  flow.remote_port = bpf_ntohs(tcp->dest);

  // Look up the RWND value for this flow.
  u32 *rwnd = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  if (rwnd == NULL) {
    // We do not know the RWND value to use for this flow.
    return TC_ACT_OK;
  }
  if (*rwnd == 0) {
    // The RWND is configured to be 0. That does not make sense.
    bpf_printk("Error: Flow with local port %u, remote port %u, RWND=0D\n",
               flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  u8 *win_scale = bpf_map_lookup_elem(&flow_to_win_scale, &flow);
  if (win_scale == NULL) {
    // We do not know the window scale to use for this flow.
    bpf_printk("Error: Flow with local port %u, remote port %u, no win scale\n",
               flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Apply the window scale to the configured RWND value.
  u16 to_set = (u16)(*rwnd >> *win_scale);
  // bpf_printk("Setting RWND for flow with local port %u to %u (win
  // scale: %u)\n", flow.local_port, to_set, *win_scale);
  // bpf_printk("Setting RWND to %u (win scale: %u, RWND with win scale:
  // %u)\n", *rwnd, *win_scale, to_set);

  // Set the RWND value in the TCP header. If the existing advertised window
  // set by flow control is smaller, then use that instead so that we
  // preserve flow control..
  tcp->window = min(tcp->window, bpf_htons(to_set));
  // tcp->window = bpf_htons((u16)(*rwnd >> *win_scale));

  return TC_ACT_OK;
}

int set_hdr_cb_flags(struct bpf_sock_ops *skops) {
  // Set the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (bpf_sock_ops_cb_flags_set(skops, skops->bpf_sock_ops_cb_flags |
                                           BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

int clear_hdr_cb_flags(struct bpf_sock_ops *skops) {
  // Clear the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (bpf_sock_ops_cb_flags_set(skops, skops->bpf_sock_ops_cb_flags &
                                           ~BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

int handle_hdr_opt_len(struct bpf_sock_ops *skops) {
  // If this is a SYN or SYNACK, then trigger the BPF_SOCK_OPS_WRITE_HDR_OPT_CB
  // callback by reserving three bytes (the minimim) for a TCP header option.
  // These three bytes will never actually be used, but reserving space is the
  // only way for that callback to be triggered.
  if (((skops->skb_tcp_flags & TCPHDR_SYN) == TCPHDR_SYN) &&
      bpf_reserve_hdr_opt(skops, 3, 0))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

int handle_write_hdr_opt(struct bpf_sock_ops *skops) {
  if (skops->family != AF_INET) {
    // This is not an IPv4 packet. We only support IPv4 packets because the
    // struct we use as a map key stores IP addresses as 32 bits. This is purely
    // an implementation detail.
    bpf_printk("Warning: Not using IPv4 for flow on local port %u: family=%u\n",
               skops->local_port, skops->family);
    return SOCKOPS_OK;
  }

  // Keep in mind that the window scale is set by the local host on
  // _outgoing_ SYN and SYNACK packets. The handle_write_hdr_opt() sockops
  // callback is only triggered for outgoing packets, so all we need to do
  // is filter for SYN and SYNACK.
  if ((skops->skb_tcp_flags & TCPHDR_SYN) != TCPHDR_SYN) {
    // This is not a SYN or SYNACK packet.
    return SOCKOPS_OK;
  }

  // This is an outgoing SYN or SYNACK packet. It should contain the window
  // scale. Let's try to look it up.

  struct tcp_opt win_scale_opt = {.kind = TCPOPT_WINDOW, .len = 0, .data = 0};
  int ret = bpf_load_hdr_opt(skops, &win_scale_opt, sizeof(win_scale_opt), 0);
  if (ret != 3 || win_scale_opt.len != 3 ||
      win_scale_opt.kind != TCPOPT_WINDOW) {
    return SOCKOPS_ERR;
  }

  bpf_printk("TCP window scale for flow %u -> %u = %u\n",
             bpf_ntohl(skops->remote_port), skops->local_port,
             win_scale_opt.data);

  // Record this window scale for use when setting the RWND in the egress path.
  struct flow flow = {.local_addr = skops->local_ip4,
                      .remote_addr = skops->remote_ip4,
                      .local_port = (u16)skops->local_port,
                      .remote_port = (u16)bpf_ntohl(skops->remote_port)};
  // Use update() instead of insert() in case this port is being reused.
  // TODO: Change to insert() once the flow cleanup code is implemented.
  bpf_map_update_elem(&flow_to_win_scale, &flow, &win_scale_opt.data, BPF_ANY);

  // Clear the flag that enables the header option write callback.
  return clear_hdr_cb_flags(skops);
}

SEC("sockops")
int read_win_scale(struct bpf_sock_ops *skops) {
  switch (skops->op) {
    case BPF_SOCK_OPS_TCP_LISTEN_CB:
      return set_hdr_cb_flags(skops);
    case BPF_SOCK_OPS_HDR_OPT_LEN_CB:
      return handle_hdr_opt_len(skops);
    case BPF_SOCK_OPS_WRITE_HDR_OPT_CB:
      return handle_write_hdr_opt(skops);
  }
  return SOCKOPS_OK;
}
