#!/usr/bin/python3
#
# Monitors incoming TCP flows to detect unfairness.

from argparse import ArgumentParser
from os import path
import socket
import struct
import sys
import time

from collections import defaultdict

from bcc import BPF

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "..", "model"))
import features
import gen_features
import utils


# Maps each flow (four-tuple) to a list of packets for that flow. New packets
# are appended to the ends of these lists. Periodically, a flow's packets are
# consumed by the inference engine and that flow's list is reset to empty.
FLOWS = defaultdict(list)


def ip_str_to_int(ip):
    return struct.unpack("<L", socket.inet_aton(ip))[0]


LOCALHOST = ip_str_to_int("127.0.0.1")

LIMIT = 100


def int_to_ip_str(ip):
    """ Converts an int storing an IP address into a dotted-quad string. """
    # Use "<" (little endian) instead of "!" (network / big endian) because the
    # IP addresses are already stored in little endian.
    return socket.inet_ntoa(struct.pack("<L", ip))


def flow_to_str(flw):
    """ Converts a flow four-tuple into a string. """
    saddr, daddr, sport, dport = flw
    return f"{int_to_ip_str(saddr)}:{sport} -> {int_to_ip_str(daddr)}:{dport}"


def flow_data_to_str(dat):
    """ Converts a flow data tuple into a string. """
    seq, srtt_us, tsval, tsecr, total_B, ihl_B, thl_B, payload_B, time_us = dat
    return (
        f"seq: {seq}, srtt: {srtt_us} us, tsval: {tsval}, tsecr: {tsecr}, "
        f"total: {total_B} B, IP header: {ihl_B} B, TCP header: {thl_B} B, "
        f"payload: {payload_B} B, time: {time.ctime(time_us / 1e3)}")


def receive_packet(pkt):
    # Skip packets on the loopback interface.
    if pkt.saddr == LOCALHOST or pkt.daddr == LOCALHOST:
        return

    flw = (pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
    dat = (
        pkt.seq, pkt.srtt_us, pkt.tsval, pkt.tsecr, pkt.total_B,
        pkt.ihl_B, pkt.thl_B, pkt.payload_B, pkt.time_us)
    FLOWS[flw].append(dat)
    print(f"{flow_to_str(flw)} --- {flow_data_to_str(dat)}")

    if len(FLOWS[flw]) > LIMIT:
        trigger_inference(flw)


def trigger_inference(flw):
    # Reorganize list of packet metrics into a structured numpy array.
    seqs, srtts_us, tsvals, tsecrs, totals_B, _, _, payloads_B, times_us = zip(*FLOWS[flw])
    pkts = utils.make_empty(len(seqs), additional_dtype=[(features.SRTT_FET, "int32")])
    pkts[features.SEQ_FET] = seqs
    pkts[features.ARRIVAL_TIME_FET] = times_us
    pkts[features.TS_1_FET] = tsvals
    pkts[features.TS_2_FET] = tsecrs
    pkts[features.PAYLOAD_FET] = payloads_B
    pkts[features.WIRELEN_FET] = totals_B
    pkts[features.SRTT_FET] = srtts_us

    fets = gen_features.parse_received_acks(flw, pkts)



def main():
    parser = ArgumentParser(description="Squelch unfair flows.")
    parser.add_argument(
        "-i", "--interval-ms", help="Poll interval (ms)", type=float
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Print debugging info"
    )
    parser.add_argument(
        "-l", "--limit", default=LIMIT, help="The number of packets to accumulate for a flow between inference runs.", type=int)
    args = parser.parse_args()

    assert args.limit > 0, f"\"--limit\" must be greater than 0 but is: {args.limit}"

    # Load BPF text.
    bpf_flp = path.join(path.abspath(path.dirname(__file__)),
                        path.basename(__file__).strip().split(".")[0] + ".c")
    if not path.isfile(bpf_flp):
        print(f"Could not find BPF program: {bpf_flp}")
        return 1
    print(f"Loading BPF program: {bpf_flp}")
    with open(bpf_flp, "r") as fil:
        bpf_text = fil.read()
    if args.debug:
        print(bpf_text)

    # Load BPF program.
    b = BPF(text=bpf_text)
    b.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")

    # This function will be called to process an event.
    def process_event(cpu, data, size):
        receive_packet(b["pkts"].event(data))

    print("Running...press Control-C to end")
    # Loop with callback to process_event().
    b["pkts"].open_perf_buffer(process_event)
    while 1:
        try:
            if args.interval_ms is not None:
                time.sleep(args.interval_ms * 1000)
            b.perf_buffer_poll()
        except KeyboardInterrupt:
            break

    print("\nFlows:")
    for flow, pkts in sorted(FLOWS.items()):
        print("\t", flow_to_str(flow), len(pkts))
    return 0


if __name__ == "__main__":
    exit(main())
