
#include <linux/tcp.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct pkt_t {
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u32 seq;
    u32 tsval;
    u32 tsecr;
    u32 payload_b;
    u32 total_b;
};

BPF_PERF_OUTPUT(pkts);

// TODO: Can we do something where we accumulate a certain number of packets for a flow and then add them all to the perf output, to reduce the number of invokations of the python function?

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    struct tcp_sock *ts = tcp_sk(sk);
    struct pkt_t pkt;
    struct iphdr iph;
    struct tcphdr tcph;

    u32 srtt = ts->srtt_us >> 3;
    u16 dport = 0;
    u16 family = sk->__sk_common.skc_family;
    
    if (family != AF_INET || skb == NULL) {
        return 0;
    }

    # Get connection-level information from the tcp_sock.
    pkt.saddr = sk->__sk_common.skc_rcv_saddr;
    pkt.daddr = sk->__sk_common.skc_daddr;
    pkt.sport = sk->__sk_common.skc_num;
    dport = sk->__sk_common.skc_dport;
    pkt.dport = ntohs(dport);

    # Get information related to this specific packet from the sk_buff.
    # Good resources:
    #     http://vger.kernel.org/~davem/skb.html
    #     http://vger.kernel.org/~davem/skb_data.html
    #     https://www.linuxquestions.org/questions/programming-9/access-tcp-header-seq-757115/

    if (skb->protocol != htons(ETH_P_IP)) {
        return 0;
    }
    iph = iphdr(skb);
    if (iph->protocol != IPPROTO_TCP) {
        return 0;
    }

    tcph = (struct tcphdr*)(skb_network_header(skb) + ip_hdrlen(skb));
    pkt.seq = tcph->seq;
    pkt.tsval = 0;
    pkt.tsecr = 0;
    pkt.payload_b = 0;
    # Add up linear and paged data.
    pkt.total_b = skb->len + skb->data_len;




    pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    

    return 0;
}
