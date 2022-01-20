#!/usr/bin/python3
#
# Monitors incoming TCP flows to detect unfairness.

from bcc import BPF
from collections import defaultdict

from time import sleep
from argparse import ArgumentParser
from os import path
import struct
import socket

# Maps each flow (four-tuple) to a list of packets for that flow. New packets
# are appended to the ends of these lists. Periodically, a flow's packets are
# consumed by the inference engine and that flow's list is reset to empty.
FLOWS = defaultdict(list)


def ip_str_to_int(ip):
    return struct.unpack("<L", socket.inet_aton(ip))[0]


LOCALHOST = ip_str_to_int("127.0.0.1")


def int_to_ip_str(ip):
    # Use "<" (little endian) instead of "!" (network / big endian) because the IP addresses are already stored in little endian.
    return socket.inet_ntoa(struct.pack("<L", ip))


def flow_to_str(flw):
    saddr, daddr, sport, dport = flw
    return f"{int_to_ip_str(saddr)}:{sport} -> {int_to_ip_str(daddr)}:{dport}"


def main():
    parser = ArgumentParser(description="Squelch unfair flows.")
    parser.add_argument(
        "-i", "--interval", default=1, help="summary interval, seconds", type=int
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="print debugging info"
    )
    args = parser.parse_args()

    # Load BPF text.
    bpf_flp = path.join(path.abspath(path.dirname(__file__)),
                        path.basename(__file__).strip().split(".")[0] + ".c")
    if not path.isfile(bpf_flp):
        print(f"Could not find BPF program: {bpf_flp}")
        return 1
    print(f"Loading BPF program: {bpf_flp}")
    with open(bpf_flp, "r") as fil:
        bpf_text = fil.read()

    # debug/dump ebpf enable or not
    if args.debug:
        print(bpf_text)

    # load BPF program
    b = BPF(text=bpf_text)
    b.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")

    # process event
    def print_event(cpu, data, size):
        pkt = b["pkts"].event(data)

        # Skip packets on the loopback interface.
        if pkt.saddr == LOCALHOST or pkt.daddr == LOCALHOST:
            return

        flw = (pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
        FLOWS[flw].append(1)
        print(flow_to_str(flw))

    print("Running...press Control-C to end")

    # loop with callback to print_event
    b["pkts"].open_perf_buffer(print_event)
    while 1:
        try:
            sleep(args.interval)
            b.perf_buffer_poll()
        except KeyboardInterrupt:
            break

    print("\nFlows:")
    for flow, pkts in sorted(FLOWS.items()):
        print("\t", flow_to_str(flow), sum(pkts))

    # Are saddr and daddr getting reversed? I except to see packets from neptun5 to my local machine.

    return 0


if __name__ == "__main__":
    exit(main())
