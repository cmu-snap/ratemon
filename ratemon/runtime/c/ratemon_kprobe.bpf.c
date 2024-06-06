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

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// 'tcp_rcv_established' will be used to track the last time that a flow
// received data so that we can determine when to classify a flow as idle.
SEC("kprobe/tcp_rcv_established")
int BPF_KPROBE(tcp_rcv_established, struct sock *sk, struct sk_buff *skb) {
  if (sk == NULL || skb == NULL) {
    bpf_printk("ERROR: 'tcp_rcv_established' sk=%u skb=%u", sk, skb);
    return 0;
  }
  // Since this is tcp_rcv_established, we know that the packet is TCP.
  // Extract the TCP header.
  // All accesses to struct members must be done through BPF_CORE_READ_INTO.
  unsigned char *data;
  BPF_CORE_READ_INTO(&data, skb, data);
  const struct tcphdr *th = (const struct tcphdr *)data;
  if (th == NULL) {
    bpf_printk("ERROR: 'tcp_rcv_established' th=%u", th);
    return 0;
  }
  // We know this is a TCP socket.
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    bpf_printk("ERROR: 'tcp_rcv_established' tp=%u", tp);
    return 0;
  }
  // If this packet does not contain new data (e.g., it is a pure ACK or a
  // retransmission), then we are not interested in it.
  __be32 seq;
  u32 rcv_nxt;
  BPF_CORE_READ_INTO(&seq, th, seq);
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);
  if (bpf_ntohl(seq) < rcv_nxt) {
    bpf_printk("ERROR: 'tcp_rcv_established' packet does not contain new data",
               tp);
    return 0;
  }
  // Build the flow struct.
  __be32 skc_daddr;
  __be32 skc_rcv_saddr;
  __u16 skc_num;
  __be16 skc_dport;
  BPF_CORE_READ_INTO(&skc_daddr, sk, __sk_common.skc_daddr);
  BPF_CORE_READ_INTO(&skc_rcv_saddr, sk, __sk_common.skc_rcv_saddr);
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);
  struct rm_flow flow = {.local_addr = bpf_ntohl(skc_rcv_saddr),
                         .remote_addr = bpf_ntohl(skc_daddr),
                         .local_port = skc_num,
                         .remote_port = bpf_ntohs(skc_dport)};
  // Check if we should record the last data time for this flow.
  if (bpf_map_lookup_elem(&flow_to_last_data_time_ns, &flow) == NULL) {
    // This flow is not in the map, so we are not supposed to track its last
    // data time.
    return 0;
  }
  // Get the current time and store it for this flow.
  unsigned long now_ns = bpf_ktime_get_ns();
  if (bpf_map_update_elem(&flow_to_last_data_time_ns, &flow, &now_ns,
                          BPF_ANY)) {
    bpf_printk("ERROR: 'tcp_rcv_established' error updating last data time");
  }
  return 0;
}
