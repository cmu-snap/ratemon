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
// #include <bpf_endian.h>


// The fixed TCP window scale to use.
#define FIXED_WIN_SCALE 0

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
    // Required for time_us to be 64 bits.
    u32 padding;
    u64 time_us;
};

struct tcp_opt {
	__u8 kind;
	__u8 len;
	__u8 data;
} __attribute__((packed));

// Pass packet features to userspace.
BPF_PERF_OUTPUT(pkts);
// Read RWND limit for flow, as set by userspace.
BPF_HASH(flow_to_rwnd, struct flow_t, u16);

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
static struct tcphdr *skb_to_tcphdr(const struct sk_buff *skb)
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

static inline bool is_syn(struct tcphdr* tcp)
{
    u8 flags = *(u8 *)(&tcp->ack_seq + 5);
    bpf_trace_printk("flags: %u\n", flags);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    bpf_trace_printk("little endian\n");
    return flags & 0b01000000;
#elif __BYTE_ORDER == __BIG_ENDIAN
    bpf_trace_printk("big endian\n");
    return flags & 0b00000010;
#else
    return 0;
#endif
}

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    if (skb == NULL)
    {
        return 0;
    }
    // Check this is IPv4.
    if (skb->protocol != htons(ETH_P_IP))
    {
        return 0;
    }

    struct iphdr *ip = skb_to_iphdr(skb);
    // Check this is TCP.
    if (ip->protocol != IPPROTO_TCP)
    {
        return 0;
    }
    struct pkt_t pkt = {};
    pkt.saddr = ip->saddr;
    pkt.daddr = ip->daddr;

    struct tcphdr *tcp = skb_to_tcphdr(skb);
    u16 sport = tcp->source;
    u16 dport = tcp->dest;
    pkt.sport = ntohs(sport);
    pkt.dport = ntohs(dport);
    u32 seq = tcp->seq;
    pkt.seq = ntohl(seq);

    // if (is_syn(tcp)) {
    //     bpf_trace_printk("is syn!\n");
    //     // set_window_scaling(tcp);
    // } else {
    //     bpf_trace_printk("not syn\n");
    // }
    // bpf_trace_printk("sport: %u, dport: %u, seq: %u\n", pkt.sport, pkt.dport, pkt.seq);
    // bpf_trace_printk("saddr: %u, daddr: %u\n", pkt.saddr, pkt.daddr, pkt.seq);
    // bpf_trace_printk("daddr: %d, dport: %d, seq: %d\n", pkt.daddr, pkt.dport, pkt.seq);

    struct tcp_sock *ts = tcp_sk(sk);
    pkt.srtt_us = ts->srtt_us >> 3;
    // TODO: For the timestamp option, we also need to parse the sent packets.
    // We use the timestamp option to determine the RTT. But what if we just
    // use srtt instead? Let's start with that.
    pkt.tsval = ts->rx_opt.rcv_tsval;
    pkt.tsecr = ts->rx_opt.rcv_tsecr;

    // Determine the IP/TCP header lengths.
    ip_hdr_len(ip, &pkt.ihl_bytes);
    tcp_hdr_len(tcp, &pkt.thl_bytes);

    // Determine total IP packet size and the TCP payload size.
    u16 total_bytes = ip->tot_len;
    pkt.total_bytes = ntohs(total_bytes);
    // The TCP payload is the total IP packet length minus IP header minus TCP
    // header.
    pkt.payload_bytes = pkt.total_bytes - pkt.ihl_bytes - pkt.thl_bytes;

    // BPF has trouble extracting the time the proper way
    // (skb_get_timestamp()), so we do this manually. The skb's raw timestamp
    // is just a u64 in nanoseconds.
    ktime_t tstamp = skb->tstamp;
    pkt.time_us = (u64)tstamp / 1000;

    // struct timeval tstamp;
    // skb_get_timestamp(skb, &tstamp);
    // pkt.time_us = tstamp.tv_sec * 1000000 + tstamp.tv_usec;

    pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    return 0;
}

// // Sets the TCP window scale option (if present) to 5.
// static void set_window_scaling(struct tcphdr *tcp)
// {
//     u32 hdr_len;
//     tcp_hdr_len(tcp, &hdr_len);
//     u32 options_len = hdr_len - 20;

//     if (options_len == 0)
//     {
//         return;
//     }

//     u8 *options = (u8*) (&tcp->urg_ptr + 2);
//     // u32 options_offset = 0;

//     // #pragma clang loop unroll(full)
//     for (int i = 0; i < 10;) {
//         int j = i / 2;
//         i += 2;
//     }

//     // There will be at most 20 bytes of options, so we can upper-bound this loop.
//     for (u32 options_offset = 0; options_offset < 20;)
//     // while (options_offset < 20)
//     {
//         // If there are less than 20 bytes of options and we have read them all, then return.
//         if (options_offset > options_len)
//         {
//             return;
//         }

//         // Parse this option:
//         //     1-byte kind,
//         //     1-byte length (includes kind and length as well),
//         //     variable-length option value.
//         u8 kind = *(options + options_offset);
//         u8 len = *(options + 1);
//         // If this is the window scale option, then set the window scale to 1.
//         if (kind == TCPOPT_WINDOW)
//         {
//             u8 *win_scale = options + options_offset + 2;
//             *win_scale = FIXED_WIN_SCALE;
//             return;
//         }

//         options_offset += len;
//     }

//     // // There will be at most 20 bytes of options, so we can upper-bound this loop.
//     // while (options_offset < 20)
//     // {
//     //     // If there are less than 20 bytes of options and we have read them all, then return.
//     //     if (options_offset > options_len)
//     //     {
//     //         return;
//     //     }

//     //     // Parse this option:
//     //     //     1-byte kind,
//     //     //     1-byte length (includes kind and length as well),
//     //     //     variable-length option value.
//     //     u8 kind = *(options + options_offset);
//     //     u8 len = *(options + 1);
//     //     // If this is the window scale option, then set the window scale to 1.
//     //     if (kind == TCPOPT_WINDOW)
//     //     {
//     //         u8 *win_scale = options + options_offset + 2;
//     //         *win_scale = FIXED_WIN_SCALE;
//     //         return;
//     //     }

//     //     options_offset += len;
//     // }
//     return;
// }

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
    if (eth->h_proto != htons(ETH_P_IP))
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

    // If this is a SYN packet, then overwrite the window scaling option.
    // if (*(&tcp->ack_seq + 5) & TCP_FLAG_SYN)
    // {
    //     set_window_scaling(tcp);
    // }

    // Prepare the lookup key.
    struct flow_t flow;
    flow.local_addr = ip->saddr;
    flow.remote_addr = ip->daddr;
    u16 local_port = tcp->source;
    u16 remote_port = tcp->dest;
    flow.local_port = ntohs(local_port);
    flow.remote_port = ntohs(remote_port);

    // For debugging purposes, only modify flows from local port 9998-10000.
    if (flow.local_port < 9998 || flow.local_port > 10000)
    {
        return TC_ACT_OK;
    }

    // bpf_trace_printk("local_addr: %u\n", flow.local_addr);
    // bpf_trace_printk("remote_addr: %u\n", flow.remote_addr);
    // bpf_trace_printk("local_port: %u\n", flow.local_port);
    // bpf_trace_printk("remote_port: %u\n", flow.remote_port);

    // bpf_trace_printk("Looking up RWND for flow (local): %u:%u\n", flow.local_addr, flow.local_port);

    // Look up the RWND value for this flow.
    u16 *rwnd = flow_to_rwnd.lookup(&flow);
    if (rwnd == NULL)
    {
        // bpf_trace_printk("Warning: Could not find RWND for flow (local): %u:%u\n", flow.local_addr, flow.local_port);
        // bpf_trace_printk("Warning: Could not find RWND for flow (remote): %u:%u\n", flow.remote_addr, flow.remote_port);
        return TC_ACT_OK;
    }
    if (*rwnd == 0)
    {
        // bpf_trace_printk("Warning: Zero RWND for flow (local): %u:%u\n", *rwnd, flow.local_addr, flow.local_port);
        // bpf_trace_printk("Warning: Zero RWND for flow (remote): %u:%u\n", *rwnd, flow.remote_addr, flow.remote_port);
        return TC_ACT_OK;
    }

    // bpf_trace_printk("Setting RWND for flow (local): %u:%u = %u\n", flow.local_addr, flow.local_port, *rwnd);
    // bpf_trace_printk("Setting RWND for flow (remote): %u:%u = %u\n", flow.remote_addr, flow.remote_port, *rwnd);

    // TODO: Need to take window scaling into account? The scaling option is part of
    //       the handshake. Detect if an outgoing packet is part of a handshake and
    //       extract the window scale option and store it in a map.

    // // Write the new RWND value into the packet.
    // bpf_skb_store_bytes(
    //     skb,
    //     ((void *)tcp + 14 - (void *)skb),
    //     &rwnd,
    //     sizeof(u16),
    //     BPF_F_RECOMPUTE_CSUM);
    tcp->window = htons(*rwnd >> FIXED_WIN_SCALE);

    // u16 rwnd_check = tcp->window;
    // bpf_trace_printk("Checking RWND for flow (local): %u:%u = %u\n", flow.local_addr, flow.local_port, rwnd_check);
    // bpf_trace_printk("Checking RWND for flow (remote): %u:%u = %u\n", flow.remote_addr, flow.remote_port, rwnd_check);

    return TC_ACT_OK;
}

static int write_active_opt(struct bpf_sock_ops *skops)
{
	struct tcphdr *tcp;
	struct tcp_opt win_scale_opt = {};
    int ret;

    tcp = skops->skb_data;
	if ((void *)(tcp + 1) > skops->skb_data_end)
		return 0;

	if (tcp->syn && tcp->ack) {
    	win_scale_opt.kind = TCPOPT_WINDOW;
        win_scale_opt.len = 0;
        win_scale_opt.data = 0;

		ret = bpf_load_hdr_opt(skops, &win_scale_opt, sizeof(win_scale_opt), 0);
		if (ret != 3 || win_scale_opt.len != 3 ||
		    win_scale_opt.kind != TCPOPT_WINDOW)
        {
            switch (ret) {
                case -ENOMSG:
                    bpf_trace_printk("synack write -ENOMSG\n");
                    break;
                case -EINVAL:
                    bpf_trace_printk("synack write -EINVAL\n");
                    break;
                case -ENOENT:
                    bpf_trace_printk("synack write -ENOENT\n");
                    break;
                case -ENOSPC:
                    bpf_trace_printk("synack write -ENOSPC\n");
                    break;
                case -EFAULT:
                    bpf_trace_printk("synack write -EFAULT\n");
                    break;
                case -EPERM:
                    bpf_trace_printk("synack write -EPERM\n");
                    break;
                default:
                    bpf_trace_printk("synack write default failure\n");
                    break;
            }
            return 0;
        }

        bpf_trace_printk(
            "synack write; %u -> %u; win scale: %u\n",
            ntohs(tcp->source),
            ntohs(tcp->dest),
            win_scale_opt.data);

        bpf_sock_ops_cb_flags_set(skops,
                skops->bpf_sock_ops_cb_flags &
                ~BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG);

        // u8 noop = 1;
        // ret = bpf_store_hdr_opt(skops, &noop,
        //         sizeof(noop), 0);
        // switch (ret) {
        //     case -ENOMSG:
        //         bpf_trace_printk("synack write noop -ENOMSG\n");
        //         break;
        //     case -EINVAL:
        //         bpf_trace_printk("synack write noop -EINVAL\n");
        //         break;
        //     case -ENOENT:
        //         bpf_trace_printk("synack write noop -ENOENT\n");
        //         break;
        //     case -ENOSPC:
        //         bpf_trace_printk("synack write noop -ENOSPC\n");
        //         break;
        //     case -EFAULT:
        //         bpf_trace_printk("synack write noop -EFAULT\n");
        //         break;
        //     case -EPERM:
        //         bpf_trace_printk("synack write noop -EPERM\n");
        //         break;
        //     default:
        //         bpf_trace_printk("synack write noop default ret: %d\n", ret);
        // }
	}

	return 1;
}

// https://elixir.bootlin.com/linux/latest/source/tools/testing/selftests/bpf/test_tcp_hdr_options.h#L113
// static __always_inline void set_hdr_cb_flags(struct bpf_sock_ops *skops)
// {
//     bpf_trace_printk("set_hdr_cb_flags\n");
//     bpf_sock_ops_cb_flags_set(
//         skops,
//         skops->bpf_sock_ops_cb_flags |
//             BPF_SOCK_OPS_PARSE_ALL_HDR_OPT_CB_FLAG |
//             BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG);
//     // bpf_sock_ops_cb_flags_set(
//     //     skops,
//     //     skops->bpf_sock_ops_cb_flags |
//     //         BPF_SOCK_OPS_STATE_CB_FLAG);
//     // bpf_sock_ops_cb_flags_set(
//     //     skops,
//     //     skops->bpf_sock_ops_cb_flags);
// }

static inline void set_hdr_cb_flags(struct bpf_sock_ops *skops, __u32 extra)
{
	bpf_sock_ops_cb_flags_set(skops,
				  skops->bpf_sock_ops_cb_flags |
				  BPF_SOCK_OPS_WRITE_HDR_OPT_CB_FLAG |
				  extra);
}
				//   BPF_SOCK_OPS_PARSE_UNKNOWN_HDR_OPT_CB_FLAG |
                //   BPF_SOCK_OPS_PARSE_ALL_HDR_OPT_CB_FLAG |

// #define TCPHDR_FIN 0x01
#define TCPHDR_SYN 0x02
// #define TCPHDR_RST 0x04
// #define TCPHDR_PSH 0x08
#define TCPHDR_ACK 0x10
// #define TCPHDR_URG 0x20
// #define TCPHDR_ECE 0x40
// #define TCPHDR_CWR 0x80
#define TCPHDR_SYNACK (TCPHDR_SYN | TCPHDR_ACK)

// static inline __u8 skops_tcp_flags(const struct bpf_sock_ops *skops)
// {
// 	return skops->skb_tcp_flags;
// }

// static int active_opt_len(struct bpf_sock_ops *skops)
// {
// 	int err;

// 	/* Reserve more than enough to allow the -EEXIST test in
// 	 * the write_active_opt().
// 	 */
// 	err = bpf_reserve_hdr_opt(skops, 3, 0);  // 12
// 	if (err)
//         return 0;

// 	return 1;
// }

static int handle_hdr_opt_len(struct bpf_sock_ops *skops)
{
    // int err;
    // struct tcphdr *tcp;
	// __u8 tcp_flags = skops_tcp_flags(skops);

	// if ((tcp_flags & TCPHDR_SYNACK) == TCPHDR_SYNACK)
    if ((skops->skb_tcp_flags & TCPHDR_SYNACK) == TCPHDR_SYNACK)
    {
        // err = bpf_reserve_hdr_opt(skops, 3, 0);  // 12
        if (bpf_reserve_hdr_opt(skops, 3, 0))
            return 0;
		// /* Check the SYN from bpf_sock_ops_kern->syn_skb */

        // struct tcp_opt win_scale_opt = {};
        // int ret;
    	// win_scale_opt.kind = TCPOPT_WINDOW;
        // win_scale_opt.len = 0;
        // win_scale_opt.data = 0;

		// /* Search the win scale option written by kernel
		//  * in the SYN packet.
		//  */
		// ret = bpf_load_hdr_opt(skops, &win_scale_opt,
		// 		       sizeof(win_scale_opt), 0);
		// if (ret != 3 || win_scale_opt.len != 3 ||
		//     win_scale_opt.kind != TCPOPT_WINDOW)
        // {
        //     switch (ret) {
        //         case -ENOMSG:
        //             bpf_trace_printk("synack len -ENOMSG\n");
        //             break;
        //         case -EINVAL:
        //             bpf_trace_printk("synack len -EINVAL\n");
        //             break;
        //         case -ENOENT:
        //             bpf_trace_printk("synack len -ENOENT\n");
        //             break;
        //         case -ENOSPC:
        //             bpf_trace_printk("synack len -ENOSPC\n");
        //             break;
        //         case -EFAULT:
        //             bpf_trace_printk("synack len -EFAULT\n");
        //             break;
        //         case -EPERM:
        //             bpf_trace_printk("synack len -EPERM\n");
        //             break;
        //         default:
        //             bpf_trace_printk("synack len default failure\n");
        //     }
        // } else {
        //     bpf_trace_printk("synack len win scale: %u\n", win_scale_opt.data);
        // }

        // bpf_trace_printk(
        //     "synack len; %u -> %u",
        //     skops->local_port,
        //     ntohl(skops->remote_port));
    }
    return 1;
	// /* Passive side should have cleared the write hdr cb by now */
	// if (skops->local_port == passive_lport_h)
	// 	RET_CG_ERR(0);

	// return active_opt_len(skops);
}


// static void active_established(struct bpf_sock_ops *skops) {
//     struct tcp_opt win_scale_opt = {};
//     int ret;
//     win_scale_opt.kind = TCPOPT_WINDOW;
//     win_scale_opt.len = 0;
//     win_scale_opt.data = 0;

//     ret = bpf_load_hdr_opt(skops, &win_scale_opt,
//                     sizeof(win_scale_opt), 0);
//     if (ret != 3 || win_scale_opt.len != 3 ||
//         win_scale_opt.kind != TCPOPT_WINDOW)
//     {
//         switch (ret) {
//             case -ENOMSG:
//                 bpf_trace_printk("synack active est -ENOMSG\n");
//                 break;
//             case -EINVAL:
//                 bpf_trace_printk("synack active est -EINVAL\n");
//                 break;
//             case -ENOENT:
//                 bpf_trace_printk("synack active est -ENOENT\n");
//                 break;
//             case -ENOSPC:
//                 bpf_trace_printk("synack active est -ENOSPC\n");
//                 break;
//             case -EFAULT:
//                 bpf_trace_printk("synack active est -EFAULT\n");
//                 break;
//             case -EPERM:
//                 bpf_trace_printk("synack active est -EPERM\n");
//                 break;
//             default:
//                 bpf_trace_printk("synack active est default failure\n");
//         }
//     } else {
//         bpf_trace_printk("synack active est win scale: %u\n", win_scale_opt.data);
//     }

//     bpf_trace_printk(
//         "synack active est; %u -> %u",
//         skops->local_port,
//         ntohl(skops->remote_port));
// }


int foo(struct bpf_sock_ops *skops)
{
	// int true_val = 1;
    // __u16 passive_lport_h = 0;
    // __u16 passive_lport_n = 0;

	switch (skops->op) {
	case BPF_SOCK_OPS_TCP_LISTEN_CB:
		// passive_lport_h = skops->local_port;
		// passive_lport_n = htons(passive_lport_h);
		// bpf_setsockopt(skops, SOL_TCP, TCP_SAVE_SYN,
		// 	       &true_val, sizeof(true_val));
		set_hdr_cb_flags(skops, 0);
		break;
	case BPF_SOCK_OPS_TCP_CONNECT_CB:
        // bpf_trace_printk("BPF_SOCK_OPS_TCP_CONNECT_CB\n");
		// set_hdr_cb_flags(skops, 0);
		break;
	case BPF_SOCK_OPS_PARSE_HDR_OPT_CB:
        // bpf_trace_printk("BPF_SOCK_OPS_PARSE_HDR_OPT_CB\n");
		return 1;
	case BPF_SOCK_OPS_HDR_OPT_LEN_CB:
        bpf_trace_printk("BPF_SOCK_OPS_HDR_OPT_LEN_CB\n");
		return handle_hdr_opt_len(skops);
	case BPF_SOCK_OPS_WRITE_HDR_OPT_CB:
        bpf_trace_printk("BPF_SOCK_OPS_WRITE_HDR_OPT_CB\n");
		return write_active_opt(skops);
	case BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB:
        // bpf_trace_printk("BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB\n");
		return 1;
    // case BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB:
    //     bpf_trace_printk("BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB\n");
    //     // active_established(skops);
    //     return 1;
	}

	return 1;
}

// int sock_stuff(struct bpf_sock_ops *skops)
// {
//     bpf_trace_printk("sock_stuff %u\n", skops->op);
//     /* ipv4 only */
//     // if (skops->family != AF_INET)
//     // {
//     //     return 1;
//     // }

//     set_hdr_cb_flags(skops);

//     // TODO: Print port...try to filter to just the messgaes related to our connections.
//     switch (skops->op)
//     {
//     case BPF_SOCK_OPS_TIMEOUT_INIT:
//         bpf_trace_printk("BPF_SOCK_OPS_TIMEOUT_INIT\n");
//         break;
//     case BPF_SOCK_OPS_RWND_INIT:
//         bpf_trace_printk("BPF_SOCK_OPS_RWND_INIT\n");
//         break;
//     case BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB\n");
//         break;
//     case BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB\n");
//         break;
//     case BPF_SOCK_OPS_NEEDS_ECN:
//         bpf_trace_printk("BPF_SOCK_OPS_NEEDS_ECN\n");
//         break;
//     case BPF_SOCK_OPS_BASE_RTT:
//         bpf_trace_printk("BPF_SOCK_OPS_BASE_RTT\n");
//         break;
//     case BPF_SOCK_OPS_RTO_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_RTO_CB\n");
//         break;
//     case BPF_SOCK_OPS_RETRANS_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_RETRANS_CB\n");
//         break;
//     case BPF_SOCK_OPS_STATE_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_STATE_CB %u %u\n", skops->state, skops->args[1]);
//         break;
//     case BPF_SOCK_OPS_RTT_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_RTT_CB\n");
//         break;
//     case BPF_SOCK_OPS_PARSE_HDR_OPT_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_PARSE_HDR_OPT_CB\n");
//         break;
//     case BPF_SOCK_OPS_HDR_OPT_LEN_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_HDR_OPT_LEN_CB\n");
//         // return handle_hdr_opt_len(skops);
//         // u8 flags = skops->   skb_tcp_flags;
//         break;
//     case BPF_SOCK_OPS_TCP_LISTEN_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_TCP_LISTEN_CB\n");
//         break;
//     case BPF_SOCK_OPS_TCP_CONNECT_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_TCP_CONNECT_CB\n");
//         // write_window_scale(skops);
//         // set_hdr_cb_flags(skops);
//         break;
//     case BPF_SOCK_OPS_WRITE_HDR_OPT_CB:
//         bpf_trace_printk("BPF_SOCK_OPS_WRITE_HDR_OPT_CB\n");
//         // write_window_scale(skops);
//         // if (skops->skb_tcp_flags & TCP_FLAG_ACK) {
//         //     bpf_trace_printk("outgoing syn!\n");
//         // }
//         break;
//     default:
//         bpf_trace_printk("default\n");
//         break;
//     }
//     return 1;
// }
