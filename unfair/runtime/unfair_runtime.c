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

#define FLOW_MAX_PACKETS 65536
// #define FLOW_MAX_PACKETS 1024

// Key for use in flow-based maps.
struct flow_t
{
    u32 local_addr;
    u32 remote_addr;
    u16 local_port;
    u16 remote_port;
};

// Packet features to pass to userspace.
struct pkt_t
{
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u32 seq;
    u32 srtt_us;
    u32 tsval;
    u32 tsecr;
    u32 total_bytes;
    u32 ihl_bytes;
    u32 thl_bytes;
    u32 payload_bytes;
    u32 epoch;
    u64 time_us;
    u32 valid;
};

struct tcp_opt
{
    __u8 kind;
    __u8 len;
    __u8 data;
} __attribute__((packed));

// struct flow_buff_t
// {
//     struct pkt_t pkts[FLOW_MAX_PACKETS];
//     u32 next_free;
// };

// Pass packet features to userspace.
// BPF_PERF_OUTPUT(pkts);
// Assemble mapping of flow to packets.
// BPF_HASH(flow_to_pkts, struct flow_t, struct flow_buff_t);
BPF_ARRAY(ringbuffer, struct pkt_t, FLOW_MAX_PACKETS);
BPF_ARRAY(next_free, int, 1);
BPF_ARRAY(epoch, u32, 1);
// Read RWND limit for flow, as set by userspace.
BPF_HASH(flow_to_rwnd, struct flow_t, u16);
// Read RWND limit for flow, as set by userspace.
BPF_HASH(flow_to_win_scale, struct flow_t, u8);

// Need to redefine this because the BCC rewriter does not support rewriting
// ip_hdr()'s internal dereferences of skb members.
// Based on: https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py
static inline struct iphdr *skb_to_iphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in ip_hdr() -> skb_network_header().
    // https://elixir.bootlin.com/linux/v4.15/source/include/linux/skbuff.h#L2286
    return (struct iphdr *)(skb->head + skb->network_header);
}

// Need to redefine this because the BCC rewriter does not support rewriting
// tcp_hdr()'s internal dereferences of skb members.
// Based on: https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py
static inline struct tcphdr *skb_to_tcphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in tcp_hdr() -> skb_transport_header().
    // https://elixir.bootlin.com/linux/v4.15/source/include/linux/skbuff.h#L2269
    return (struct tcphdr *)(skb->head + skb->transport_header);
}

// Get the total length in bytes of an IP header.
static inline void ip_hdr_len(struct iphdr *ip, u32 *ihl_int)
{
    // The header length is before the ToS field in the TCP header. However, the header
    // length is in a bitfield, but BPF cannot read bitfield elements. So we need to
    // read a whole byte and extract the data offset from that. We only read a single
    // byte, so we do not need to use ntohs().
    u8 ihl = *(u8 *)(&ip->tos - 1);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    ihl = (ihl & 0xf0) >> 4;
#elif __BYTE_ORDER == __BIG_ENDIAN
    ihl = ihl & 0x0f;
#endif
    *ihl_int = (u32)ihl * 4;
}

// Get the total length in bytes of a TCP header.
static inline void tcp_hdr_len(struct tcphdr *tcp, u32 *thl_int)
{
    // The TCP data offset is located after the ACK sequence number in the TCP header.
    // However, the data offset is in a bitfield, but BPF cannot read bitfield
    // elements. So we need to read a whole byte and extract the data offset from that.
    // We only read a single byte, so we do not need to use ntohs().
    u8 thl = *(u8 *)(&tcp->ack_seq + 4);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    thl = (thl & 0x0f) >> 4;
#elif __BYTE_ORDER == __BIG_ENDIAN
    thl = (thl & 0xf0) >> 4;
#endif
    *thl_int = (u32)thl * 4;
}

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    if (skb == NULL)
    {
        return 0;
    }
    // Check this is IPv4.
    if (skb->protocol != bpf_htons(ETH_P_IP))
    {
        return 0;
    }

    struct iphdr *ip = skb_to_iphdr(skb);
    // Check this is TCP.
    if (ip->protocol != IPPROTO_TCP)
    {
        return 0;
    }

    // Look up the current epoch.
    int zero = 0;
    u32 *epoch_ptr = epoch.lookup(&zero);
    if (epoch_ptr == NULL)
    {
        return 0;
    }
    u32 epoch_int = *epoch_ptr;

    // Look up the index of the next location in the ringbuffer.
    int *next_free_ptr = next_free.lookup(&zero);
    if (next_free_ptr == NULL)
    {
        return 0;
    }
    int next_free_int = *next_free_ptr;
    // bpf_trace_printk("next_free_int: %d, epoch_int: %u\n", next_free_int, epoch_int);

    // Get the next location in the ringbuffer.
    struct pkt_t *pkt = ringbuffer.lookup(&next_free_int);
    if (pkt == NULL)
    {
        return 0;
    }
    // Fill in packet metadata needed for managing the ringbuffer.
    pkt->valid = 1;
    pkt->epoch = epoch_int;

    pkt->saddr = ip->saddr;
    pkt->daddr = ip->daddr;

    struct tcphdr *tcp = skb_to_tcphdr(skb);
    pkt->sport = bpf_ntohs(tcp->source);
    pkt->dport = bpf_ntohs(tcp->dest);
    pkt->seq = bpf_ntohl(tcp->seq);

    struct tcp_sock *ts = tcp_sk(sk);
    pkt->srtt_us = ts->srtt_us >> 3;
    // TODO: For the timestamp option, we also need to parse the sent packets.
    // We use the timestamp option to determine the RTT. But what if we just
    // use srtt instead? Let's start with that.
    pkt->tsval = ts->rx_opt.rcv_tsval;
    pkt->tsecr = ts->rx_opt.rcv_tsecr;

    // Determine the IP/TCP header lengths.
    ip_hdr_len(ip, &pkt->ihl_bytes);
    tcp_hdr_len(tcp, &pkt->thl_bytes);

    // Determine total IP packet size and the TCP payload size.
    pkt->total_bytes = bpf_ntohs(ip->tot_len);
    // The TCP payload is the total IP packet length minus IP header minus TCP
    // header.
    pkt->payload_bytes = pkt->total_bytes - pkt->ihl_bytes - pkt->thl_bytes;

    // BPF has trouble extracting the time the proper way
    // (skb_get_timestamp()), so we do this manually. The skb's raw timestamp
    // is just a u64 in nanoseconds.
    ktime_t tstamp = skb->tstamp;
    pkt->time_us = (u64)tstamp / 1000;

    // Advance the ringbuffer to the next position.
    (*next_free_ptr)++;
    if ((*next_free_ptr) >= FLOW_MAX_PACKETS)
    {
        (*next_free_ptr) = 0;
        (*epoch_ptr)++;
    }

    // struct timeval tstamp;
    // skb_get_timestamp(skb, &tstamp);
    // pkt.time_us = tstamp.tv_sec * 1000000 + tstamp.tv_usec;

    return 0;
}

// Inspired by: https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
int handle_egress(struct __sk_buff *skb)
{
    if (skb == NULL)
    {
        return TC_ACT_OK;
    }

    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    struct tcphdr *tcp;

    // Do a sanity check to make sure that the IP header does not extend past
    // the end of the packet.
    if ((void *)eth + sizeof(*eth) > data_end)
    {
        return TC_ACT_OK;
    }

    // Check that this is IP. Calculate the start of the IP header, and do a
    // sanity check to make sure that the IP header does not extend past the
    // end of the packet.
    if (eth->h_proto != bpf_htons(ETH_P_IP))
    {
        return TC_ACT_OK;
    }
    ip = data + sizeof(*eth);
    if ((void *)ip + sizeof(*ip) > data_end)
    {
        return TC_ACT_OK;
    }

    // Similar for TCP header.
    if (ip->protocol != IPPROTO_TCP)
    {
        return TC_ACT_OK;
    }
    tcp = (void *)ip + sizeof(*ip);
    if ((void *)tcp + sizeof(*tcp) > data_end)
    {
        return TC_ACT_OK;
    }

    // Prepare the lookup key.
    struct flow_t flow;
    flow.local_addr = ip->saddr;
    flow.remote_addr = ip->daddr;
    flow.local_port = bpf_ntohs(tcp->source);
    flow.remote_port = bpf_ntohs(tcp->dest);

    // For debugging purposes, only modify flows from local port 9998-10000.
    if (flow.local_port < 9998 || flow.local_port > 10000)
    {
        return TC_ACT_OK;
    }

    // Look up the RWND value for this flow.
    u16 *rwnd = flow_to_rwnd.lookup(&flow);
    if (rwnd == NULL)
    {
        // We do not know the RWND value to use for this flow.
        return TC_ACT_OK;
    }
    if (*rwnd == 0)
    {
        // The RWND is configured to be 0. That does not make sense.
        return TC_ACT_OK;
    }

    u8 *win_scale = flow_to_win_scale.lookup(&flow);
    if (win_scale == NULL)
    {
        // We do not know the window scale to use for this flow.
        return TC_ACT_OK;
    }

    // bpf_trace_printk("Setting RWND for flow with local port %u to %u (win scale: %u)\n", flow.local_port, *rwnd, *win_scale);

    // Apply the window scale to the configured RWND value and set it in the packet.
    tcp->window = bpf_htons(*rwnd >> *win_scale);

    return TC_ACT_OK;
}

static inline int set_hdr_cb_flags(struct bpf_sock_ops *skops)
{
    // Set the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
    // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
    if (bpf_sock_ops_cb_flags_set(skops,
                                  skops->bpf_sock_ops_cb_flags |
                                      BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
        return SOCKOPS_ERR;
    return SOCKOPS_OK;
}

static inline int clear_hdr_cb_flags(struct bpf_sock_ops *skops)
{
    // Clear the flag enabling the BPF_SOCK_OPS_HDR_OPT_LEN_CB and
    // BPF_SOCK_OPS_WRITE_HDR_OPT_CB callbacks.
    if (bpf_sock_ops_cb_flags_set(skops,
                                  skops->bpf_sock_ops_cb_flags &
                                      ~BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG))
        return SOCKOPS_ERR;
    return SOCKOPS_OK;
}

static inline int handle_hdr_opt_len(struct bpf_sock_ops *skops)
{
    // If this is a SYNACK, then trigger the BPF_SOCK_OPS_WRITE_HDR_OPT_CB callback by
    // reserving three bytes (the minimim) for a TCP header option. These three bytes
    // will never actually be used, but reserving space is the only way for that
    // callback to be triggered.
    if (((skops->skb_tcp_flags & TCPHDR_SYNACK) == TCPHDR_SYNACK) &&
        bpf_reserve_hdr_opt(skops, 3, 0))
        return SOCKOPS_ERR;
    return SOCKOPS_OK;
}

static inline int handle_write_hdr_opt(struct bpf_sock_ops *skops)
{
    if (skops->family != AF_INET)
    {
        // This is not an IPv4 packet. We only support IPv4 packets because the struct
        // we use as a map key stores IP addresses as 32 bits. This is purely an
        // implementation detail.
        bpf_trace_printk("Warning: Not using IPv4 for flow on local port %u\n", skops->local_port);
        return SOCKOPS_OK;
    }
    if ((skops->skb_tcp_flags & TCPHDR_SYNACK) != TCPHDR_SYNACK)
    {
        // This is not a SYNACK packet.
        return SOCKOPS_OK;
    }

    // This is a SYNACK packet.

    struct tcp_opt win_scale_opt = {
        .kind = TCPOPT_WINDOW,
        .len = 0,
        .data = 0};
    int ret = bpf_load_hdr_opt(skops, &win_scale_opt, sizeof(win_scale_opt), 0);
    if (ret != 3 || win_scale_opt.len != 3 ||
        win_scale_opt.kind != TCPOPT_WINDOW)
    {
        switch (ret)
        {
        case -ENOMSG:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -ENOMSG\n", skops->local_port);
            break;
        case -EINVAL:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -EINVAL\n", skops->local_port);
            break;
        case -ENOENT:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -ENOENT\n", skops->local_port);
            break;
        case -ENOSPC:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -ENOSPC\n", skops->local_port);
            break;
        case -EFAULT:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -EFAULT\n", skops->local_port);
            break;
        case -EPERM:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: -EPERM\n", skops->local_port);
            break;
        default:
            bpf_trace_printk("Error: Failure loading window scale option for flow on local port %u: failure code = %d\n", skops->local_port, ret);
        }
        return SOCKOPS_ERR;
    }

    bpf_trace_printk(
        "TCP window scale for flow %u -> %u = %u\n",
        bpf_ntohl(skops->remote_port), skops->local_port, win_scale_opt.data);

    // Record this window scale for use when setting the RWND in the egress path.
    struct flow_t flow = {
        .local_addr = skops->local_ip4,
        .remote_addr = skops->remote_ip4,
        .local_port = (u16)skops->local_port,
        .remote_port = (u16)bpf_ntohl(skops->remote_port)};
    // Use update() instead of insert() in case this port is being reused.
    // TODO: Change to insert() once the flow cleanup code is implemented.
    flow_to_win_scale.update(&flow, &win_scale_opt.data);

    // Clear the flag that enables the header option write callback.
    return clear_hdr_cb_flags(skops);
}

int read_win_scale(struct bpf_sock_ops *skops)
{
    switch (skops->op)
    {
    case BPF_SOCK_OPS_TCP_LISTEN_CB:
        return set_hdr_cb_flags(skops);
    case BPF_SOCK_OPS_HDR_OPT_LEN_CB:
        return handle_hdr_opt_len(skops);
    case BPF_SOCK_OPS_WRITE_HDR_OPT_CB:
        return handle_write_hdr_opt(skops);
    }
    return SOCKOPS_OK;
}
