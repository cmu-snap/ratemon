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

#define ETH_P_IP 0x0800 /* Internet Protocol packet	*/

#define TC_ACT_OK 0

#define min(x, y) ((x) < (y) ? (x) : (y))

// Perform RWND tuning at TC egress. If a flow has an entry in flow_to_rwnd,
// then install that value in the advertised window field. Inspired by:
// https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
SEC("tc/egress")
int do_rwnd_at_egress(struct __sk_buff *skb) {
  // bpf_printk("INFO: do_rwnd_at_egress");
  if (skb == NULL) {
    return TC_ACT_OK;
  }

  void *data = (void *)(long)skb->data;
  void *data_end = (void *)(long)skb->data_end;
  // We get the packet starting with the Ethernet header and need to parse the
  // network and transport headers.
  struct ethhdr *eth = data;
  struct iphdr *ip;
  struct tcphdr *tcp;

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
  u32 *rwnd = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  if (rwnd == NULL) {
    // This flow does not have a custom RWND value.
    // bpf_printk(
    //     "WARNING: flow with local port %u and remote port %u has no RWND
    //     value", flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }
  // For scheduled RWND tuning, it is fine for the RWND to be 0.
  // if (*rwnd == 0) {
  //   // The RWND is configured to be 0. That does not make sense.
  //   bpf_printk("ERROR: Flow with local port %u, remote port %u, RWND=0B",
  //              flow.local_port, flow.remote_port);
  //   return TC_ACT_OK;
  // }

  if (*rwnd == 0) {
    // If the configured RWND value is 0 B, then we can take a shortcut and not
    // bother looking up the window scale.
    tcp->window = 0;
    // bpf_printk(
    //     "INFO: set RWND for flow with local port %u and remote port %u to 0
    //     B", flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Only need to look up the window scale value if the RWND is not 0.
  u8 *win_scale = bpf_map_lookup_elem(&flow_to_win_scale, &flow);
  if (win_scale == NULL) {
    // We do not know the window scale to use for this flow.
    bpf_printk("ERROR: flow with local port %u, remote port %u, no win scale",
               flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Apply the window scale to the configured RWND value.
  u16 rwnd_with_win_scale = (u16)(*rwnd >> *win_scale);
  // Set the RWND value in the TCP header. If the existing advertised window
  // set by flow control is smaller, then use that instead so that we
  // preserve flow control.
  tcp->window = min(tcp->window, rwnd_with_win_scale);
  // bpf_printk(
  //     "INFO: set RWND for flow with remote port %u to %u (win scale: %u)",
  //     flow.remote_port, rwnd_with_win_scale, *win_scale);
  return TC_ACT_OK;
}