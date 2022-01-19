#!/usr/bin/python3
#
# Monitors incoming TCP flows to detect unfairness.

from bcc import BPF
from time import sleep
import argparse


# define BPF program
BPF_TEXT = """
#include <linux/tcp.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct pkt_t {
    u32 saddr;
    u32 daddr;
    u16 lport;
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
        pkt.lport = sk->__sk_common.skc_num;
        dport = sk->__sk_common.skc_dport;
        pkt.dport = ntohs(dport);

        pkts.perf_submit(ctx, &pkt, sizeof(pkt));
    }

    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Squelch unfair flows.")
    parser.add_argument("-i", "--interval", default=1,
                        help="summary interval, seconds")
    parser.add_argument("-d", "--debug", action="store_true",
        help="print debugging info")

    args = parser.parse_args()

    # debug/dump ebpf enable or not
    if args.debug:
        print(BPF_TEXT)

    # load BPF program
    b = BPF(text=BPF_TEXT)
    b.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")

    # process event
    def print_event(cpu, data, size):
        event = b["events"].event(data)
        print(event)

    print("Running")

    # loop with callback to print_event
    b["events"].open_perf_buffer(print_event)
    while 1:
        try:
            sleep(int(args.interval))
            b.perf_buffer_poll()
        except KeyboardInterrupt:
            exit()


if __name__ == "__main__":
    main()
