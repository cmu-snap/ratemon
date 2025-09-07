// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// trunk-ignore-begin(include-what-you-use)
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon.h"
#include "ratemon.bpf.h"
#include "ratemon_maps.h"
// trunk-ignore-end(include-what-you-use)
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

inline int max(int val1, int val2) { return val1 > val2 ? val1 : val2; }

// 'tcp_rcv_established' will be used to track the last time that a flow
// received data so that we can determine when to classify a flow as idle.
SEC("kprobe/tcp_rcv_established")
// trunk-ignore(clang-tidy/performance-no-int-to-ptr)
// trunk-ignore(clang-tidy/misc-unused-parameters)
int BPF_KPROBE(tcp_rcv_established, struct sock *sk, struct sk_buff *skb) {
  if (sk == NULL || skb == NULL) {
    RM_PRINTK("ERROR: 'tcp_rcv_established' sk=%u skb=%u", sk, skb);
    return 0;
  }
  // Since this is tcp_rcv_established, we know that the packet is TCP.
  // Extract the TCP header.
  // All accesses to struct members must be done through BPF_CORE_READ_INTO.
  void *data = NULL;
  BPF_CORE_READ_INTO(&data, skb, data);
  const struct tcphdr *th = (const struct tcphdr *)data;
  if (th == NULL) {
    RM_PRINTK("ERROR: 'tcp_rcv_established' th=%u", th);
    return 0;
  }
  // We know this is a TCP socket.
  struct tcp_sock *tp = (struct tcp_sock *)sk;
  if (tp == NULL) {
    RM_PRINTK("ERROR: 'tcp_rcv_established' tp=%u", tp);
    return 0;
  }

  // Safely extract members from tcp_sock, tcphdr, and sk_buff.
  // tcp_sock:
  uint32_t rcv_nxt = 0;
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);
  // tcphdr
  __be32 seq_ = 0;
  BPF_CORE_READ_INTO(&seq_, th, seq);
  uint32_t seq = bpf_ntohl(seq_);
  uint64_t doff = BPF_CORE_READ_BITFIELD_PROBED(th, doff);
  uint64_t syn = BPF_CORE_READ_BITFIELD_PROBED(th, syn);
  uint64_t fin = BPF_CORE_READ_BITFIELD_PROBED(th, fin);
  uint64_t rst = BPF_CORE_READ_BITFIELD_PROBED(th, rst);
  // sk_buff:
  uint32_t len = 0;
  __be32 skc_daddr = 0;
  __be32 skc_rcv_saddr = 0;
  uint16_t skc_num = 0;
  __be16 skc_dport = 0;
  BPF_CORE_READ_INTO(&len, skb, len);
  BPF_CORE_READ_INTO(&skc_daddr, sk, __sk_common.skc_daddr);
  BPF_CORE_READ_INTO(&skc_rcv_saddr, sk, __sk_common.skc_rcv_saddr);
  BPF_CORE_READ_INTO(&skc_num, sk, __sk_common.skc_num);
  BPF_CORE_READ_INTO(&skc_dport, sk, __sk_common.skc_dport);

  // Build the flow struct.
  struct rm_flow flow = {.local_addr = bpf_ntohl(skc_rcv_saddr),
                         .remote_addr = bpf_ntohl(skc_daddr),
                         .local_port = skc_num,
                         .remote_port = bpf_ntohs(skc_dport)};

  // Check for TCP keepalive. From Wireshark
  // (https://www.wireshark.org/docs/wsug_html_chunked/ChAdvTCPAnalysis.html):
  // A packet is a keepalive "...when the segment size is zero or one, the
  // current sequence number is one byte less than the next expected sequence
  // number, and none of SYN, FIN, or RST are set".
  uint32_t payload_bytes = len - (doff * 4);
  if ((payload_bytes <= 1) && (seq == rcv_nxt - 1) && !syn && !fin && !rst) {
    // This is a keepalive packet. Set a 1 to indicate that this flow received a
    // keepalive. This 1 will be removed when the flow goes idle.
    RM_PRINTK("INFO: 'tcp_rcv_established' found keepalive for flow %u<->%u",
              flow.local_port, flow.remote_port);
    int one = 1;
    if (bpf_map_update_elem(&flow_to_keepalive, &flow, &one, BPF_ANY)) {
      RM_PRINTK("ERROR: 'tcp_rcv_established' error updating "
                "flow_to_keepalive for flow %u<->%u",
                flow.local_port, flow.remote_port);
    }
  }

  return 0;

  // clang-format off
  // Disable clang-format so that it does not wreck commented-out code.

  // The rest is disabled because because grant checks will now happen based on
  // ACK seq in tc egress.

  // // // If this packet does not contain new data (e.g., it is a pure ACK or a
  // // // retransmission), then we are not interested in it.
  // // if (seq < rcv_nxt) {
  // //   RM_PRINTK("INFO: 'tcp_rcv_established' packet does not contain new data
  // //   "
  // //              "for flow %u<->%u",
  // //              flow.local_port, flow.remote_port);
  // //   return 0;
  // // }

  // uint32_t grant_used = max(0, (seq + payload_bytes) - rcv_nxt);

  // // Check if we should record the last data time for this flow.
  // if (bpf_map_lookup_elem(&flow_to_last_data_time_ns, &flow) != NULL) {
  //   // This flow is in the map, so we are supposed to track its last data time.
  //   // Get the current time and store it for this flow.
  //   uint64_t now_ns = bpf_ktime_get_ns();
  //   if (bpf_map_update_elem(&flow_to_last_data_time_ns, &flow, &now_ns,
  //                           BPF_ANY)) {
  //     RM_PRINTK("ERROR: 'tcp_rcv_established' error updating "
  //                "flow_to_last_data_time_ns for flow %u<->%u",
  //                flow.local_port, flow.remote_port);
  //   }
  // }

  // // Grant update. Do this regardless of whether we are using scheduling mode
  // // "byte" or "time". If we are actually doing time-based scheduling, then
  // // this will have no effect.
  // //
  // // Look up the flow in the flow_to_rwnd map.
  // struct rm_grant_info *grant_info = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  // if (grant_info == NULL) {
  //   return 0;
  // }
  // // Wait until here to print this log to we only log flows we care about.
  // RM_PRINTK("INFO: 'tcp_rcv_established' flow %u<->%u received %u bytes",
  //            flow.local_port, flow.remote_port, payload_bytes);
  // RM_PRINTK("INFO: 'tcp_rcv_established' flow %u<->%u used %u bytes of grant",
  //            flow.local_port, flow.remote_port, grant_used);
  // RM_PRINTK(
  //     "INFO: 'tcp_rcv_established' flow %u<->%u is active and in flow_to_rwnd",
  //     flow.local_port, flow.remote_port);
  // if (grant_info->override_rwnd_bytes == 0) {
  //   RM_PRINTK("INFO: 'tcp_rcv_established' flow %u<->%u has already exhausted "
  //              "its grant, so we do nothing",
  //              flow.local_port, flow.remote_port);
  //   return 0;
  // }

  // // Decrement the rwnd by the payload size to account for grant that has been
  // // used.
  // if (grant_info->override_rwnd_bytes >= grant_used) {
  //   grant_info->override_rwnd_bytes -= grant_used;
  // } else {
  //   grant_info->override_rwnd_bytes = 0;
  // }
  // // Update flow_to_rwnd with the new value.
  // RM_PRINTK("INFO: 'tcp_rcv_established' updating flow_to_rwnd for flow "
  //            "%u<->%u to %u",
  //            flow.local_port, flow.remote_port, grant_info->override_rwnd_bytes);
  // if (bpf_map_update_elem(&flow_to_rwnd, &flow, grant, BPF_ANY)) {
  //   RM_PRINTK("ERROR: 'tcp_rcv_established' error updating flow_to_rwnd for "
  //              "flow %u<->%u to %u",
  //              flow.local_port, flow.remote_port, grant_info->override_rwnd_bytes);
  //   return 0;
  // }
  // // If the flow has grant remaining, then we are done.
  // if (grant_info->override_rwnd_bytes > 0) {
  //   RM_PRINTK("INFO: 'tcp_rcv_established' flow %u<->%u has remaining grant "
  //              "%u, so it is not done",
  //              flow.local_port, flow.remote_port, grant_info->override_rwnd_bytes);
  //   return 0;
  // }

  // // The flow has exhausted its grant with this segment, so add it to the
  // // done_flows ringbuf.
  // RM_PRINTK("INFO: 'tcp_rcv_established' flow %u<->%u has exhausted its "
  //            "grant, adding to done_flows",
  //            flow.local_port, flow.remote_port);
  // if (bpf_ringbuf_output(&done_flows, &flow, sizeof(flow), 0)) {
  //   RM_PRINTK("ERROR: 'tcp_rcv_established' error submitting done flow "
  //              "%u<->%u to ringbuf",
  //              flow.local_port, flow.remote_port);
  //   return 0;
  // }

  // return 0;

  // clang-format on
}
