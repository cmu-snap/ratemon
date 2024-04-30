// Intercepts incoming IPv4/TCP packets, extracts useful header fields and
// metrics, and passes them to userspace.
//
// Based on tcprtt.py and tcpdrop.py:
//     https://github.com/iovisor/bcc/blob/master/tools/tcprtt.py
//     https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py

#include <bcc/proto.h>
#include <linux/ip.h>
#include <linux/ktime.h>
#include <linux/skbuff.h>
#include <linux/tcp.h>
#include <linux/time.h>
#include <net/ip.h>
#include <net/sock.h>
#include <net/tcp.h>

// BPF SOCK_OPS program return codes.
#define SOCKOPS_OK 1
#define SOCKOPS_ERR 0

// TCP header flags.
#define TCPHDR_SYN 0x02
#define TCPHDR_ACK 0x10
#define TCPHDR_SYNACK (TCPHDR_SYN | TCPHDR_ACK)

#define FLOW_MAX_PACKETS 8192
// #define FLOW_MAX_PACKETS 65536
// #define FLOW_MAX_PACKETS 262144
// #define FLOW_MAX_PACKETS 524288
// #define FLOW_MAX_PACKETS 1024

// Key for use in flow-based maps.
struct flow_t {
  u32 local_addr;
  u32 remote_addr;
  u16 local_port;
  u16 remote_port;
};

struct tcp_opt {
  __u8 kind;
  __u8 len;
  __u8 data;
} __attribute__((packed));

// Read RWND limit for flow, as set by userspace. Even though the advertised
// window is only 16 bits in the TCP header, use 32 bits here because we have
// not taken window scaling into account yet. BPF_HASH(flow_to_rwnd, struct
// flow_t, u32);
BPF_TABLE_PINNED("hash", struct flow_t, u32, flow_to_rwnd, 1024,
                 "/sys/fs/bpf/flow_to_rwnd");
// Read RWND limit for flow, as set by userspace.
// BPF_HASH(flow_to_win_scale, struct flow_t, u8);
BPF_TABLE_PINNED("hash", struct flow_t, u8, flow_to_win_scale, 1024,
                 "/sys/fs/bpf/flow_to_win_scale");
// Test map
// BPF_TABLE_PINNED("array", u32, test_map, 1024, "/sys/fs/bpf/test_map");
// // Array of paused sockets, for use by BPF_ITER(bpf_map).
// BPF_TABLE_PINNED("array", struct sock, paused_socks, 1024,
//                  "/sys/fs/bpf/paused_socks");

// BPF_TABLE_PINNED(_table_type, _key_type, _leaf_type, _name, _max_entries,
// "/sys/fs/bpf/xyz");

// Inspired by:
// https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
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
  struct flow_t flow;
  flow.local_addr = ip->saddr;
  flow.remote_addr = ip->daddr;
  flow.local_port = bpf_ntohs(tcp->source);
  flow.remote_port = bpf_ntohs(tcp->dest);

  // Look up the RWND value for this flow.
  u32 *rwnd = flow_to_rwnd.lookup(&flow);
  if (rwnd == NULL) {
    // We do not know the RWND value to use for this flow.
    return TC_ACT_OK;
  }
  if (*rwnd == 0) {
    // The RWND is configured to be 0. That does not make sense.
    bpf_trace_printk(
        "Error: Flow with local port %u, remote port %u, RWND=0D\n",
        flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  u8 *win_scale = flow_to_win_scale.lookup(&flow);
  if (win_scale == NULL) {
    // We do not know the window scale to use for this flow.
    bpf_trace_printk(
        "Error: Flow with local port %u, remote port %u, no win scale\n",
        flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  // Apply the window scale to the configured RWND value.
  u16 to_set = (u16)(*rwnd >> *win_scale);
  // bpf_trace_printk("Setting RWND for flow with local port %u to %u (win
  // scale: %u)\n", flow.local_port, to_set, *win_scale);
  // bpf_trace_printk("Setting RWND to %u (win scale: %u, RWND with win scale:
  // %u)\n", *rwnd, *win_scale, to_set);

  // Set the RWND value in the TCP header. If the existing advertised window
  // set by flow control is smaller, then use that instead so that we
  // preserve flow control..
  tcp->window = min(tcp->window, bpf_htons(to_set));
  // tcp->window = bpf_htons((u16)(*rwnd >> *win_scale));

  return TC_ACT_OK;
}

static inline int set_hdr_cb_flags(struct bpf_sock_ops *skops) {
  // Set the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (bpf_sock_ops_cb_flags_set(skops, skops->bpf_sock_ops_cb_flags |
                                           BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

static inline int clear_hdr_cb_flags(struct bpf_sock_ops *skops) {
  // Clear the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
  // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
  if (bpf_sock_ops_cb_flags_set(skops, skops->bpf_sock_ops_cb_flags &
                                           ~BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

static inline int handle_hdr_opt_len(struct bpf_sock_ops *skops) {
  // If this is a SYN or SYNACK, then trigger the BPF_SOCK_OPS_WRITE_HDR_OPT_CB
  // callback by reserving three bytes (the minimim) for a TCP header option.
  // These three bytes will never actually be used, but reserving space is the
  // only way for that callback to be triggered.
  if (((skops->skb_tcp_flags & TCPHDR_SYN) == TCPHDR_SYN) &&
      bpf_reserve_hdr_opt(skops, 3, 0))
    return SOCKOPS_ERR;
  return SOCKOPS_OK;
}

static inline int handle_write_hdr_opt(struct bpf_sock_ops *skops) {
  if (skops->family != AF_INET) {
    // This is not an IPv4 packet. We only support IPv4 packets because the
    // struct we use as a map key stores IP addresses as 32 bits. This is purely
    // an implementation detail.
    bpf_trace_printk(
        "Warning: Not using IPv4 for flow on local port %u: family=%u\n",
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
    switch (ret) {
      case -ENOMSG:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -ENOMSG\n",
            skops->local_port);
        break;
      case -EINVAL:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -EINVAL\n",
            skops->local_port);
        break;
      case -ENOENT:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -ENOENT\n",
            skops->local_port);
        break;
      case -ENOSPC:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -ENOSPC\n",
            skops->local_port);
        break;
      case -EFAULT:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -EFAULT\n",
            skops->local_port);
        break;
      case -EPERM:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: -EPERM\n",
            skops->local_port);
        break;
      default:
        bpf_trace_printk(
            "Error: Failure loading window scale option for flow on local port "
            "%u: failure code = %d\n",
            skops->local_port, ret);
    }
    return SOCKOPS_ERR;
  }

  // Use bpf_ntohl instead of bpf_ntohs because the port is actually stored as a
  // u32.
  bpf_trace_printk("TCP window scale for flow %u -> %u = %u\n",
                   bpf_ntohl(skops->remote_port), skops->local_port,
                   win_scale_opt.data);

  // Record this window scale for use when setting the RWND in the egress path.
  struct flow_t flow = {.local_addr = skops->local_ip4,
                        .remote_addr = skops->remote_ip4,
                        .local_port = (u16)skops->local_port,
                        // Use bpf_ntohl instead of bpf_ntohs because the port
                        // is actually stored as a u32.
                        .remote_port = (u16)bpf_ntohl(skops->remote_port)};
  // Use update() instead of insert() in case this port is being reused.
  // TODO: Change to insert() once the flow cleanup code is implemented.
  flow_to_win_scale.update(&flow, &win_scale_opt.data);

  // Clear the flag that enables the header option write callback.
  return clear_hdr_cb_flags(skops);
}

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

// struct bpf_iter__tcp {
//   __bpf_md_ptr(struct bpf_iter_meta *, meta);
//   __bpf_md_ptr(struct sock_common *, sk_common);
//   uid_t uid __aligned(8);
// };

// BPF_ITER(tcp) {
//   struct sock_common *sk_common = ctx->sk_common;
//   bpf_trace_printk("Iterating\n");
//   return 0;
// }

// BPF_ITER(task)
// {
//   struct seq_file *seq = ctx->meta->seq;
//   struct task_struct *task = ctx->task;

//   if (task == (void *)0)
//     return 0;

//   //   ... task->pid, task->tgid, task->comm, ...
//   return 0;
// }

// BPF_ITER(bpf_map) {
//   struct bpf_map *map = ctx->map;
//   u32 *key = ctx->key;
//   u32 *val = ctx->value;
//   if (key == (void *)0 || val == (void *)0) return 0;
//   bpf_trace_printk("Iterating %d %d\n", key, val);
//   return 0;
// }
