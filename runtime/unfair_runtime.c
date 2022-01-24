// Based on tcprtt.py and tcpdrop.py:
//     https://github.com/iovisor/bcc/blob/master/tools/tcprtt.py
//     https://github.com/iovisor/bcc/blob/master/tools/tcpdrop.py

#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/skbuff.h>
#include <net/ip.h>
#include <net/sock.h>
#include <bcc/proto.h>

// #include <bpf/bpf_core_read.h>
// #include <bcc/bpf_helpers.h>
// #include <bcc/bpf_core_read.h>
// #include <bcc/bpf_tracing.h>
// #include <bcc/bpf_endian.h>

struct pkt_t {
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u32 seq;
    u32 srtt_us;
    u32 tsval;
    u32 tsecr;
    u32 total_b;
    u32 ihl_b;
    u32 thl_b;
    u32 payload_b;
};

BPF_PERF_OUTPUT(pkts);

// Need to redefine this because the BCC rewriter does not support rewriting the internal dereferences of skb members.
static struct tcphdr *skb_to_tcphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in tcp_hdr() -> skb_transport_header().
    return (struct tcphdr *)(skb->head + skb->transport_header);
}
static inline struct iphdr *skb_to_iphdr(const struct sk_buff *skb)
{
    // unstable API. verify logic in ip_hdr() -> skb_network_header().
    return (struct iphdr *)(skb->head + skb->network_header);
}

// from include/net/tcp.h:
// #ifndef tcp_flag_byte
// #define tcp_flag_byte(th) (((u_int8_t *)th)[13])
// #endif

// TODO: Can we do something where we accumulate a certain number of packets for a flow and then add them all to the perf output, to reduce the number of invokations of the python function?

// separate data structs for ipv4 and ipv6
// struct ipv4_data_t {
//     u32 pid;
//     u64 ip;
//     u32 saddr;
//     u32 daddr;
//     u16 sport;
//     u16 dport;
//     u8 state;
//     u8 tcpflags;
//     u32 stack_id;
// };

// BPF_STACK_TRACE(stack_traces, 1024);

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    if (sk == NULL)
        return 0;

    // u32 pid = bpf_get_current_pid_tgid() >> 32;
    // pull in details from the packet headers and the sock struct
    // u16 family = sk->__sk_common.skc_family;
    // char state = sk->__sk_common.skc_state;
    u16 sport = 0, dport = 0;
    struct tcphdr *tcp = skb_to_tcphdr(skb);
    struct iphdr *ip = skb_to_iphdr(skb);
    // u8 tcpflags = ((u_int8_t *)tcp)[13];
    sport = tcp->source;
    dport = tcp->dest;
    sport = ntohs(sport);
    dport = ntohs(dport);
    
    struct pkt_t pkt;
    // struct ipv4_data_t data4 = {};

    // data4.pid = pid;
    // data4.ip = 4;
    // data4.saddr = ip->saddr;
    pkt.saddr = ip->saddr;
    // data4.daddr = ip->daddr;
    pkt.daddr = ip->daddr;
    // data4.dport = dport;
    pkt.dport = dport;
    // data4.sport = sport;
    pkt.sport = sport;
    pkt.seq = tcp->seq;
    // data4.state = state;
    // data4.tcpflags = tcpflags;
    // data4.stack_id = stack_traces.get_stackid(ctx, 0);
    // ipv4_events.perf_submit(ctx, &data4, sizeof(data4));

    struct tcp_sock *ts = tcp_sk(sk);
    // struct iphdr *iph;
    // struct tcphdr *tcph;

    // pkt.saddr = 0;
    // pkt.daddr = 0;
    // pkt.sport = 0;
    // pkt.dport = 0;
    // pkt.seq = 0;
    // pkt.srtt = 0;
    // pkt.tsval = 0;
    // pkt.tsecr = 0;
    // pkt.total_b = 0;
    // pkt.payload_b = 0;

    // // u16 dport = 0;
    // // u16 family = sk->__sk_common.skc_family;
    
    // if (sk->__sk_common.skc_family != AF_INET || skb == NULL) {
    //     return 0;
    // }

    // Get connection-level information from the tcp_sock.
    // pkt.saddr = sk->__sk_common.skc_rcv_saddr;
    // pkt.daddr = sk->__sk_common.skc_daddr;
    // pkt.sport = sk->__sk_common.skc_num;
    // dport = sk->__sk_common.skc_dport;
    // pkt.dport = ntohs(dport);

    // Get information related to this specific packet from the sk_buff.
    // Good resources:
    //     http://vger.kernel.org/~davem/skb.html
    //     http://vger.kernel.org/~davem/skb_data.html
    //     https://www.linuxquestions.org/questions/programming-9/access-tcp-header-seq-757115/

    // if (skb->protocol != htons(ETH_P_IP)) {
    //     return 0;
    // }


    // check this is TCP
    // u8 transport_protocol = 0;
    // workaround for reading the sk_protocol bitfield:
    // bpf_probe_read(&transport_protocol, sizeof(transport_protocol), (void *)((long)&sk->sk_wmem_queued) - 3);

    // iph = ip_hdr(skb);
    // pkt.saddr = iph->saddr;

    // __u8 proto = BPF_CORE_READ(iph, protocol);
    // __u8 proto;
    // void* addr = &iph->protocol;
    // bpf_probe_read_kernel(&proto, sizeof(proto), addr);
    //  = iph->protocol;
    // struct iphdr iph;
    // bpf_skb_load_bytes(skb, (void *)iph_ - (void *)skb, (void*)&iph, sizeof(ip_hdr));


    // u8 transport_protocol;
    // bpf_probe_read_user(&transport_protocol, sizeof(transport_protocol), (void *)&iph->protocol);
    // pid_t pid;
    // bpf_core_read(&transport_protocol, sizeof(transport_protocol), &iph->protocol);

    // // if (iph->protocol != IPPROTO_TCP) {
    // if (transport_protocol != IPPROTO_TCP) {
    //     return 0;
    // }
    // tcph = tcp_hdr(skb); // (struct tcphdr*)(skb_network_header(skb) + ip_hdrlen(skb));

    // Parse TCP timestamp option
    // Based on: https://stackoverflow.com/questions/42750552/read-tcp-options-fields?rq=1
    // uint8_t *p = (uint8_t *)tcph + sizeof(struct tcphdr);
    // uint8_t *end = (uint8_t *)tcph + tcp_hdrlen(tcph);
    // while (p < end) {
    //     uint8_t kind = *p++;
    //     // Option types 0 and 1 do not have a size field.
    //     if (kind == 0) {
    //         // End of options.
    //         break;
    //     }
    //     if (kind == 1) {
    //         // No-op option with no length.
    //         continue;
    //     }
    //     // All other option types do have a size field.
    //     uint8_t size = *p++;
    //     if (kind == 8) {
    //         // The TCP timestamp option is type 8.
    //         pkt.tsval = ntohs(*(u32 *)p);
    //         pkt.tsecr = ntohs(*(u32 *)(p + sizeof(u32)));
    //         // We only care about the TCP timestamp option, to abort after finding it.
    //         break;
    //     }
    //     p += (size - 2);
    // }


    // __be32 saddr = ip_hdr(skb)->saddr;

    // __be32 saddr;
    // void *addr = &ip_hdr(skb)->saddr;
    // bpf_probe_read_kernel(&saddr, sizeof(ip_hdr(skb)->saddr), &ip_hdr(skb)->saddr);
    // bpf_probe_read_kernel(&saddr, sizeof(saddr), addr);

    // bpf_skb_load_bytes(skb, addr - (void *)skb, &saddr, sizeof(saddr));

    // pkt.saddr = ntohs(saddr);
    // pkt.saddr = ntohs(iph->saddr);
    // pkt.daddr = ntohs(iph->daddr);
    // pkt.sport = ntohs(tcph->source);
    // pkt.dport = ntohs(tcph->dest);
    // pkt.seq = ntohs(tcph->seq);
    pkt.srtt_us = ts->srtt_us >> 3;
    pkt.tsval = ts->rx_opt.rcv_tsval;
    pkt.tsecr = ts->rx_opt.rcv_tsecr;
    // Add up linear and paged data.
    // pkt.total_b = skb->len + skb->data_len;
    // TCP payload is the total size minus the IP header minus the TCP header.


    u16 total_b;
    total_b = ip->tot_len;
    pkt.total_b = ntohs(total_b);

    u8 ihl;  // = ip->ihl;
    bpf_probe_read(&ihl, sizeof(ihl), &ip);
    ihl = ihl & 0x0f;

    // u64 thl;    
    // bpf_probe_read(&thl, sizeof(thl), &tcp->ack_seq);
    // thl = thl >> 24;
    // thl = thl & 0xfffffffffffffff0;

    u8 thl;
    bpf_probe_read(&thl, sizeof(thl), &tcp->ack_seq + 4);
    thl = (thl & 0xf0) >> 4;

    pkt.ihl_b = (u32)ihl * 4;
    pkt.thl_b = (u32)thl * 4;

    pkt.payload_b = pkt.total_b - pkt.ihl_b - pkt.thl_b;

    pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    return 0;
}
