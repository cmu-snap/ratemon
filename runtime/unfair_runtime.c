
#include <linux/tcp.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct pkt_t {
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
};

BPF_PERF_OUTPUT(pkts);

// TODO: Can we do something where we accumulate a certain number of packets for a flow and then add them all to the perf output, to reduce the number of invokations of the python function?

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    struct tcp_sock *ts = tcp_sk(sk);
    u32 srtt = ts->srtt_us >> 3;
    u16 dport = 0;
    u16 family = sk->__sk_common.skc_family;
    
    if (family == AF_INET) {
        struct pkt_t pkt;
        pkt.saddr = sk->__sk_common.skc_rcv_saddr;
        pkt.daddr = sk->__sk_common.skc_daddr;
        pkt.sport = sk->__sk_common.skc_num;
        dport = sk->__sk_common.skc_dport;
        pkt.dport = ntohs(dport);

        pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    }

    return 0;
}
