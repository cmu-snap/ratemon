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

// TODO: Update this to also track the last sequence number for a flow. Use
// sequence number to decrement a flow's RWND over time based on how much data
// has been received. But do we even need this? If TCP does not withdraw
// advertised window, then it should be fine to advertise once, then advertise 0
// immediately after and the sender will have tracked the amount of data it can
// send. Maybe. But to be truly TCP compatible, we should gentle decrement RWND
// as data arrives. However, we do still need to know when it is safe to give an
// allocation to a new flow. So we need to track when a flow that was just
// granted to has sent its entire grant. So we do need to track the last
// sequence number for a flow. How will this work. We track last seq here. Need
// to know the seq that we started the grant with as well. Then in interop we
// periodically check our stored last seq to see if we've gone far enough? No, I
// think a better way is to track the remaining grant. Whenever data comes in,
// we decrement the remaining grant by that amount. Perhaps can track this in
// the RWND map. Is this too much work to do per-packet? No, because this is
// per-skb which will be a large chunk of data (64KB) if we're doing a good job.
// Probably best.
// Then this kprobe can handle the grant decrement without involving interop.
// Interop just needs to know when to give a grant to a different flow.
// Could create a map that is like "recently_done_with_grant" and kprobe can put
// a flow in that when it has fulfilled its grant. Of course there will be some
// inaccuracy because it will take a while for an ACK to go out with RWND=0 and
// get to the sender, so more data will be in flight. But actually no, RWND is a
// window, so the sender won't ever transmit more than we grant.
// So then the design of interop is to just wait for the done queue to be
// nonempty, update its internal data structures, select a new flow to activate,
// and set its RWND. I think the only thing that needs to change in interop is
// the selection of when a flow is done. Instead of time-based, if a flow shows
// up in the done queue, then it is done. We can keep the existing logic for
// idle timeout to remove a flow earlier. Do we even need a new map for this?
// Interop knows the flows that it has granted to. In the rwnd map, they are
// only not done with their grants if the rwnd limit is greater than 0. So
// instead of a done map, interop could just check each of the active flows in
// the rwnd map to see if they are down to 0. The con of this is that it
// requires checking all active flows in the rwnd map, whereas a done queue can
// be checked only once for every flow that finishes. Lets start with the design
// that reuses the rwnd map. Use a BPF ringbuf:
// https://docs.ebpf.io/linux/map-type/BPF_MAP_TYPE_RINGBUF/ Then interop can
// epoll to wait for flows to finish. This will save CPU and complexity of
// repeatedly waking up to check. This may also remove the need for interop's
// timer-based design, since epoll will handle the wait. What about the leftover
// grants at the end of a burst? If the response size is not a perfect multiple
// of the grant size, then there will be leftover grants. So we may still need a
// way to retract a grant. Basically, all flows will end up with a little bit of
// grant left, so at the next burst they will all send at once. The scale of
// this impact depends on the number of flows, response size skew, and burst
// duration. 3 approaches: 1) assume perfect knowledge of response sizes to set
// grants perfectly; 2) use first responses to estimate typical response size,
// switch from large grants to small
//    grants ("minimally active set") as a flow approaches the estimated
//    response size, this limits the impact of leftover grants;
// 3) modify TCP to support retracting grants.

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
  void *data;
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

  // Safely extract members from tcp_sock, tcphdr, and sk_buff.
  // tcp_sock:
  u32 rcv_nxt;
  BPF_CORE_READ_INTO(&rcv_nxt, tp, rcv_nxt);
  // tcphdr
  __be32 seq_;
  BPF_CORE_READ_INTO(&seq_, th, seq);
  u32 seq = bpf_ntohl(seq_);
  u64 doff = BPF_CORE_READ_BITFIELD_PROBED(th, doff);
  u64 syn = BPF_CORE_READ_BITFIELD_PROBED(th, syn);
  u64 fin = BPF_CORE_READ_BITFIELD_PROBED(th, fin);
  u64 rst = BPF_CORE_READ_BITFIELD_PROBED(th, rst);
  // sk_buff:
  u32 len;
  __be32 skc_daddr;
  __be32 skc_rcv_saddr;
  __u16 skc_num;
  __be16 skc_dport;
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
  u32 payload_bytes = len - (doff * 4);
  if ((payload_bytes <= 1) && (seq == rcv_nxt - 1) && !syn && !fin && !rst) {
    // This is a keepalive packet.
    bpf_printk("INFO: 'tcp_rcv_established' found keepalive");
    int one = 1;
    if (bpf_map_update_elem(&flow_to_keepalive, &flow, &one, BPF_ANY)) {
      bpf_printk(
          "ERROR: 'tcp_rcv_established' error updating flow_to_keepalive");
    }
  }

  // If this packet does not contain new data (e.g., it is a pure ACK or a
  // retransmission), then we are not interested in it.
  if (seq < rcv_nxt) {
    bpf_printk("INFO: 'tcp_rcv_established' packet does not contain new data",
               tp);
    return 0;
  }

  // Check if we should record the last data time for this flow.
  if (bpf_map_lookup_elem(&flow_to_last_data_time_ns, &flow) != NULL) {
    // This flow is in the map, so we are supposed to track its last data time.
    // Get the current time and store it for this flow.
    u64 now_ns = bpf_ktime_get_ns();
    if (bpf_map_update_elem(&flow_to_last_data_time_ns, &flow, &now_ns,
                            BPF_ANY)) {
      bpf_printk("ERROR: 'tcp_rcv_established' error updating "
                 "flow_to_last_data_time_ns");
    }
  }

  // Grant update.
  //
  // Look up the flow in the flow_to_rwnd.
  u32 *rwnd_ptr = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  if (rwnd_ptr == NULL) {
    bpf_printk("ERROR: 'tcp_rcv_established' error flow is active but not in "
               "flow_to_rwnd");
    return 0;
  }
  // Decrement the rwnd by the payload size.
  if (*rwnd_ptr >= payload_bytes) {
    *rwnd_ptr -= payload_bytes;
  } else {
    *rwnd_ptr = 0;
  }
  // Update the flow_to_rwnd with the new value.
  if (bpf_map_update_elem(&flow_to_rwnd, &flow, rwnd_ptr, BPF_ANY)) {
    bpf_printk("ERROR: 'tcp_rcv_established' error updating flow_to_rwnd");
    return 0;
  }
  // The flow still has grant remaining.
  if (*rwnd_ptr > 0) {
    return 0;
  }
  // The flow has exhausted its grant. Add it to the done_flows ringbuf.
  if (bpf_ringbuf_output(&done_flows, &flow, sizeof(flow), 0)) {
    bpf_printk("ERROR: 'tcp_rcv_established' error submitting done flow");
    return 0;
  }

  return 0;
}
