
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

// BPF SOCK_OPS program return codes.
#define SOCKOPS_OK 1
#define SOCKOPS_ERR 0

#define AF_INET 2

#define EINVAL 22

#define TCPOPT_WINDOW 3 /* Window scaling */

// TCP header flags.
#define TCPHDR_SYN 0x02
#define TCPHDR_ACK 0x10
#define TCPHDR_SYNACK (TCPHDR_SYN | TCPHDR_ACK)

struct tcp_opt {
  __u8 kind;
  __u8 len;
  __u8 data;
} __attribute__((packed));

// The next several functions are helpers for the sockops program that records
// the receiver's TCP window scale value.

__always_inline int set_hdr_cb_flags(struct bpf_sock_ops *skops, int flags) {
  long ret = bpf_sock_ops_cb_flags_set(skops, flags);
  if (ret == -EINVAL) {
    // This is not a fullsock.
    // Note: bpf_sk_fullsock() is not available in sockops, so if this is not a
    // fullsock there is nothing we can do.
    bpf_printk(
        "ERROR: failed to set sockops flags because socket is not full socket");
    return SOCKOPS_ERR;
  } else if (ret) {
    bpf_printk("ERROR: failed to set specific sockops flag: %ld", ret);
    return SOCKOPS_ERR;
  }
  return SOCKOPS_OK;
}

__always_inline int enable_hdr_cbs(struct bpf_sock_ops *skops) {
  // Set the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (set_hdr_cb_flags(skops, skops->bpf_sock_ops_cb_flags |
                                  BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG) ==
      SOCKOPS_ERR) {
    bpf_printk("ERROR: could not enable sockops header callbacks");
    return SOCKOPS_ERR;
  }
  return SOCKOPS_OK;
}

__always_inline int disable_hdr_cbs(struct bpf_sock_ops *skops) {
  // Clear the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (set_hdr_cb_flags(skops, skops->bpf_sock_ops_cb_flags &
                                  ~BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG) ==
      SOCKOPS_ERR) {
    bpf_printk("ERROR: could not disable sockops header callbacks");
    return SOCKOPS_ERR;
  }
  return SOCKOPS_OK;
}

__always_inline int handle_hdr_opt_len(struct bpf_sock_ops *skops) {
  // Keep in mind that the window scale is set by the local host on
  // _outgoing_ SYN and SYNACK packets. The handle_write_hdr_opt() sockops
  // callback is only triggered for outgoing packets, so all we need to do
  // is filter for SYN and SYNACK.
  if ((skops->skb_tcp_flags & TCPHDR_SYN) != TCPHDR_SYN) {
    // This is not a SYN or SYNACK packet.
    return SOCKOPS_OK;
  }

  // If this is a SYN or SYNACK, then trigger the BPF_SOCK_OPS_WRITE_HDR_OPT_CB
  // callback by reserving three bytes (the minimim) for a TCP header option.
  // These three bytes will never actually be used, but reserving space is the
  // only way for that callback to be triggered.
  if (bpf_reserve_hdr_opt(skops, 3, 0)) {
    bpf_printk("ERROR: failed to reserve space for a header option");
    return SOCKOPS_ERR;
  }
  return SOCKOPS_OK;
}

__always_inline int handle_write_hdr_opt(struct bpf_sock_ops *skops) {
  // Keep in mind that the window scale is set by the local host on
  // _outgoing_ SYN and SYNACK packets. The handle_write_hdr_opt() sockops
  // callback is only triggered for outgoing packets, so all we need to do
  // is filter for SYN and SYNACK.
  if ((skops->skb_tcp_flags & TCPHDR_SYN) != TCPHDR_SYN) {
    // This is not a SYN or SYNACK packet.
    return SOCKOPS_OK;
  }

  // Look up the TCP window scale.
  struct tcp_opt win_scale_opt = {.kind = TCPOPT_WINDOW, .len = 0, .data = 0};
  if (bpf_load_hdr_opt(skops, &win_scale_opt, sizeof(win_scale_opt), 0) != 3 ||
      win_scale_opt.len != 3 || win_scale_opt.kind != TCPOPT_WINDOW) {
    bpf_printk("ERROR: failed to retrieve window scale option");
    return SOCKOPS_ERR;
  }

  if (skops->family != AF_INET) {
    // This is not an IPv4 packet. We only support IPv4 packets because the
    // struct we use as a map key stores IP addresses as 32 bits. This is purely
    // an implementation detail.
    bpf_printk("WARNING: not using IPv4 for flow on local port %u: family=%u",
               skops->local_port, skops->family);
    disable_hdr_cbs(skops);
    return SOCKOPS_OK;
  }

  struct rm_flow flow = {.local_addr = skops->local_ip4,
                         .remote_addr = skops->remote_ip4,
                         .local_port = (u16)skops->local_port,
                         // Use bpf_ntohl instead of bpf_ntohs because the port
                         // is actually stored as a u32.
                         .remote_port = (u16)bpf_ntohl(skops->remote_port)};
  bpf_printk(
      "INFO: TCP window scale for flow with remote port %u and local port %u "
      "is %u",
      flow.remote_port, flow.local_port, win_scale_opt.data);

  // Record this window scale for use when setting the RWND in the egress path.
  // Use update() instead of insert() in case this port is being reused.
  // TODO: Change to insert() once the flow cleanup code is implemented.
  bpf_map_update_elem(&flow_to_win_scale, &flow, &win_scale_opt.data, BPF_ANY);

  // Clear the flag that enables the header option write callback.
  disable_hdr_cbs(skops);
  return SOCKOPS_OK;
}

// This sockops program records a flow's TCP window scale, which is set in
// receiver's outgoing SYNACK packet.
SEC("sockops")
int read_win_scale(struct bpf_sock_ops *skops) {
  switch (skops->op) {
    case BPF_SOCK_OPS_TCP_LISTEN_CB:
      return enable_hdr_cbs(skops);
    case BPF_SOCK_OPS_HDR_OPT_LEN_CB:
      return handle_hdr_opt_len(skops);
    case BPF_SOCK_OPS_WRITE_HDR_OPT_CB:
      return handle_write_hdr_opt(skops);
  }
  return SOCKOPS_OK;
}