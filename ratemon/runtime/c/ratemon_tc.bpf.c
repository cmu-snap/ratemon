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

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

enum {
  ETH_P_IP = 0x0800 /* Internet Protocol packet	*/
};

enum { TC_ACT_OK = 0 };

// Perform RWND tuning at TC egress. If a flow has an entry in flow_to_rwnd,
// then install that value in the advertised window field. Inspired by:
// https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
SEC("tc/egress")
int do_rwnd_at_egress(struct __sk_buff *skb) {
  // bpf_printk("INFO: do_rwnd_at_egress");
  if (skb == NULL) {
    return TC_ACT_OK;
  }

  // trunk-ignore(clang-tidy/performance-no-int-to-ptr)
  void *data = (void *)(long)skb->data;
  // trunk-ignore(clang-tidy/performance-no-int-to-ptr)
  void *data_end = (void *)(long)skb->data_end;
  // We get the packet starting with the Ethernet header and need to parse the
  // network and transport headers.
  struct ethhdr *eth = data;
  struct iphdr *ip = NULL;
  struct tcphdr *tcp = NULL;

  // Check that the Ethernet header does not extend past the end of the packet.
  if ((void *)eth + sizeof(*eth) > data_end) {
    return TC_ACT_OK;
  }
  // Check that the network header is IP, find its start, and check that it does
  // not extend past the end of the packet.
  if (eth->h_proto != bpf_htons(ETH_P_IP)) {
    return TC_ACT_OK;
  }
  ip = data + sizeof(*eth);
  if ((void *)ip + sizeof(*ip) > data_end) {
    return TC_ACT_OK;
  }
  // Check that the transport header is TCP, find its start, and check that it
  // does not extend past the end of the packet.
  if (ip->protocol != IPPROTO_TCP) {
    return TC_ACT_OK;
  }
  tcp = (void *)ip + sizeof(*ip);
  if ((void *)tcp + sizeof(*tcp) > data_end) {
    return TC_ACT_OK;
  }

  // Look up the RWND value for this flow.
  // Note the difference between bpf_ntohl and bpf_ntohs.
  struct rm_flow flow = {.local_addr = bpf_ntohl(ip->saddr),
                         .remote_addr = bpf_ntohl(ip->daddr),
                         .local_port = bpf_ntohs(tcp->source),
                         .remote_port = bpf_ntohs(tcp->dest)};
  struct rm_grant_info *grant_info = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  if (grant_info == NULL) {
    return TC_ACT_OK;
  }

  u32 ack_seq = bpf_ntohl(tcp->ack_seq);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: ack_seq: %u",
             flow.local_port, flow.remote_port, ack_seq);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: ungranted_bytes: %u",
             flow.local_port, flow.remote_port, grant_info->ungranted_bytes);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
             "grant_info->override_rwnd_bytes: %u",
             flow.local_port, flow.remote_port,
             grant_info->override_rwnd_bytes);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
             "grant_info->new_grant_bytes: %u",
             flow.local_port, flow.remote_port, grant_info->new_grant_bytes);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
             "grant_info->grant_seq_num_end_bytes: %u",
             flow.local_port, flow.remote_port,
             grant_info->grant_seq_num_end_bytes);
  bpf_printk(
      "INFO: 'do_rwnd_at_egress' flow %u<->%u: grant_info->grant_done: %u",
      flow.local_port, flow.remote_port, grant_info->grant_done);
  bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
             "grant_info->excessive_grant_bytes: %u",
             flow.local_port, flow.remote_port,
             grant_info->excessive_grant_bytes);

  // Determine new RWND value, either based on the override or the grant info.
  u32 rwnd = 0;
  if (grant_info->override_rwnd_bytes == 0xFFFFFFFF) {
    // Override is 2^32-1, so use grant info.
    // TODO(unknown): Handle sequence number wraparound.
    if (grant_info->new_grant_bytes > 0) {
      u32 grant_to_use =
          min(grant_info->ungranted_bytes, grant_info->new_grant_bytes);
      grant_info->ungranted_bytes -= grant_to_use;
      // This flow received a new grant. Update the grant end seq num and clear
      // the new grant field. Support stacking grants by basing the new end seq
      // num off of the old end seq num if the latter is already ahead of the
      // ACK seq num, e.g., if there is currently a grant active. Take into
      // account pre-granted bytes.
      grant_info->grant_seq_num_end_bytes =
          max(ack_seq, grant_info->grant_seq_num_end_bytes) + grant_to_use +
          grant_info->excessive_grant_bytes;
      rwnd = grant_to_use;
      grant_info->new_grant_bytes = 0;
      // We have applied the pre-granted bytes to the grant end seq num, so
      // reset this.
      grant_info->excessive_grant_bytes = 0;
      bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u received new grant "
                 "of %u bytes",
                 flow.local_port, flow.remote_port, rwnd);
    } else if (ack_seq >= grant_info->grant_seq_num_end_bytes) {
      // The flow has no grant and should be paused.
      rwnd = 0;
      // Check grant_done so that we only submit this flow to done_flows once.
      if (!grant_info->grant_done) {
        grant_info->grant_done = true;
        // The flow has exhausted its grant with this segment, so add it to the
        // done_flows ringbuf.
        bpf_printk(
            "INFO: 'do_rwnd_at_egress' flow %u<->%u has spent its grant, "
            "adding to done_flows",
            flow.local_port, flow.remote_port);
        if (bpf_ringbuf_output(&done_flows, &flow, sizeof(flow), 0)) {
          bpf_printk("ERROR: 'do_rwnd_at_egress' error submitting done flow "
                     "%u<->%u to ringbuf",
                     flow.local_port, flow.remote_port);
          return 0;
        }
      }
      bpf_printk(
          "INFO: 'do_rwnd_at_egress' flow %u<->%u has exhausted its grant, "
          "pausing with RWND 0 bytes",
          flow.local_port, flow.remote_port);
    } else {
      // The flow is on an existing grant.
      rwnd = grant_info->grant_seq_num_end_bytes - ack_seq;
      bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u has active grant "
                 "with %u bytes remaining",
                 flow.local_port, flow.remote_port, rwnd);
    }
  } else {
    // Override is not 2^32-1, so ignore grant info and use the override value.
    rwnd = grant_info->override_rwnd_bytes;
    bpf_printk("INFO: 'do_rwnd_at_egress' flow %u<->%u has override RWND %u",
               flow.local_port, flow.remote_port, rwnd);
  }

  // Apply the new RWND value.

  if (rwnd == 0) {
    // If the configured RWND value is 0 B, then we can take a shortcut and not
    // bother looking up the window scale.
    tcp->window = 0;
    bpf_printk("INFO: 'do_rwnd_at_egress' set RWND for flow with local port %u "
               "and remote port %u to 0B",
               flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Only need to look up the window scale value if the RWND is not 0.
  u8 *win_scale = bpf_map_lookup_elem(&flow_to_win_scale, &flow);
  if (win_scale == NULL) {
    // We do not know the window scale to use for this flow.
    bpf_printk("ERROR: 'do_rwnd_at_egress' flow with local port %u, remote "
               "port %u, no win scale",
               flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Apply the window scale to the configured RWND value.
  u16 rwnd_with_win_scale = (u16)(rwnd >> *win_scale);
  // The smallest RWND we can set with accuracy is 1 << win scale. If applying
  // the win scale makes the RWND 0 when it was not 0 before, then set it to 1
  // so we do not accidentally pause the flow. We know the RWND is not supposed
  // to be 0 at this point in the code because there is a check for that case
  // above. Even though we are technically granting more than we should, the
  // official grant end seq num is still precise, so we can still correctly
  // determine when a flow is done.
  u16 rwnd_to_set = 0;
  if (rwnd_with_win_scale == 0) {
    rwnd_to_set = 1;
    u32 extra_grant = (((u32)rwnd_to_set) << *win_scale) - rwnd;
    // Since we are pre-granting this amount, subtract it from the ungranted
    // bytes. But do not change the grant end seq num, since that will mess up
    // our grant end detection.
    grant_info->ungranted_bytes -=
        min(grant_info->ungranted_bytes, extra_grant);
    grant_info->excessive_grant_bytes += extra_grant;
    bpf_printk("INFO: 'do_rwnd_at_egress' flow with local port %u, remote port "
               "%u carrying forward excessive grant of %d bytes",
               flow.local_port, flow.remote_port, extra_grant);
  } else {
    rwnd_to_set = rwnd_with_win_scale;
  }
  // Set the RWND value in the TCP header. If the existing advertised window
  // set by flow control is smaller, then use that instead so that we
  // preserve flow control.
  tcp->window = bpf_htons(min(bpf_ntohs(tcp->window), rwnd_to_set));
  bpf_printk("INFO: 'do_rwnd_at_egress' set RWND for flow with remote port %u "
             "to %u (win scale: %u)",
             flow.remote_port, rwnd_to_set, *win_scale);
  return TC_ACT_OK;
}