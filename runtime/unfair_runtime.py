#!/usr/bin/python3
"""
Monitors incoming TCP flows to detect unfairness.
"""

from argparse import ArgumentParser
from os import path
import pickle
import socket
import struct
import sys
import threading
import time

from collections import defaultdict

from bcc import BPF

sys.path.insert(0, path.join(path.dirname(
    path.realpath(__file__)), "..", "model"))
import features
import gen_features
import models
import utils


def ip_str_to_int(ip):
    return struct.unpack("<L", socket.inet_aton(ip))[0]


LOCALHOST = ip_str_to_int("127.0.0.1")
DONE = False


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


def receive_packet(lock_i, flows, pkt):
    # Skip packets on the loopback interface.
    if pkt.saddr == LOCALHOST or pkt.daddr == LOCALHOST or pkt.saddr == ip_str_to_int("23.40.28.82"):
        return

    flw = (pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
    dat = (
        pkt.seq, pkt.srtt_us, pkt.tsval, pkt.tsecr, pkt.total_B,
        pkt.ihl_B, pkt.thl_B, pkt.payload_B, pkt.time_us)
    lock_i.acquire()
    flows[flw].append(dat)
    lock_i.release()
    print(f"{flow_to_str(flw)} --- {flow_data_to_str(dat)}")


def inference_loop(lock_i, lock_f, flows, fairness_db, limit, net,
                   disable_inference):
    try:
        check_and_run_inference(
            lock_i, lock_f, flows, fairness_db, limit, net, disable_inference)
    except KeyboardInterrupt:
        return


def check_and_run_inference(lock_i, lock_f, flows, fairness_db, limit, net,
                            disable_inference):
    while not DONE:
        to_remove = []
        to_inf = []

        print("Checking flows...")
        lock_i.acquire()
        print(f"Found {len(flows)} flows")
        for flw, dat in flows.items():
            print(f"{flw} - {len(dat)}")
            if dat:
                # # Remove old flows.
                # if dat[0][-1]:
                #     to_remove.append(flw)
                # Plan to run inference on "full" flows.
                if len(dat) >= limit:
                    to_inf.append(flw)
            else:
                # Remove flows with no packets.
                to_remove.append(flw)
        for flw in to_remove:
            del flows[flw]
        lock_i.release()

        print(f"Checking {len(to_inf)} flows...")
        for flw in to_inf:
            # lock only to extract a flow's data and remove it from flows.
            lock_i.acquire()
            dat = flows[flw]
            del flows[flw]
            lock_i.release()
            print(f"Checking flow {flw}")
            # Do not hold lock while running inference.
            if not disable_inference:
                monitor_flow(lock_f, fairness_db, net, flw, dat)
        time.sleep(1)


def monitor_flow(lock_f, fairness_db, net, flw, dat):
    """
    Runs inference for the provided list of packets.

    Determines appropriate ACK pacing for the flow.

    Updates the flow's fairness record.
    """
    fets = featurize(flw, dat)
    label = inference(net, flw, fets)

    lock_f.acquire()
    prev_decision = fairness_db[flw]
    new_decision = make_decision(flw, label, prev_decision)
    fairness_db[flw] = (label, new_decision)
    lock_f.release()

    print(f"Report for flow {flw}: {label}, {new_decision}")


def inference(net, flw, fets):
    """
    Runs inference on a flow.

    Returns a label: below fair, approximately fair, above fair.
    """
    # Can we skip the dataloader stuff because we aren't doing rebalancing or scaling?


    return net.predict(dat_in, torch=False)


def make_decision(flw, label, prev_decision):
    """
    Makes a flw unfairness mitigation decision based on the provided label and
    previous decision.
    """
    return 0


def featurize(flw, dat):
    """
    Computes features for the provided list of packets.

    Returns a structured numpy array.
    """
    # Reorganize list of packet metrics into a structured numpy array.
    seqs, srtts_us, tsvals, tsecrs, totals_B, _, _, payloads_B, times_us = zip(
        *dat)
    pkts = utils.make_empty(len(seqs), additional_dtype=[
                            (features.SRTT_FET, "int32")])
    pkts[features.SEQ_FET] = seqs
    pkts[features.ARRIVAL_TIME_FET] = times_us
    pkts[features.TS_1_FET] = tsvals
    pkts[features.TS_2_FET] = tsecrs
    pkts[features.PAYLOAD_FET] = payloads_B
    pkts[features.WIRELEN_FET] = totals_B
    pkts[features.SRTT_FET] = srtts_us

    return gen_features.parse_received_acks(flw, pkts)


def main():
    parser = ArgumentParser(description="Squelch unfair flows.")
    parser.add_argument(
        "-i", "--interval-ms", help="Poll interval (ms)", type=float
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Print debugging info"
    )
    parser.add_argument(
        "-l", "--limit", default=100,
        help=("The number of packets to accumulate for a flow between "
              "inference runs."),
        type=int
    )
    parser.add_argument(
        "-s", "--disable-inference", action="store_true",
        help="Disable periodic inference."
    )
    parser.add_argument(
        "--model", choices=models.MODEL_NAMES, help="The model to use.", required=True, type=str)
    parser.add_argument(
        "-f", "--model-file", help="The trained model to use.", required=True, type=str
    )
    args = parser.parse_args()

    assert args.limit > 0, f"\"--limit\" must be greater than 0 but is: {args.limit}"
    assert path.isfile(args.model_file), f"Model does not exist: {args.model_file}"

    net = models.MODELS[args["model"]]()
    with open(args.model_file, "rb") as fil:
        net.net = pickle.load(fil)

    # Maps each flow (four-tuple) to a list of packets for that flow. New
    # packets are appended to the ends of these lists. Periodically, a flow's
    # packets are consumed by the inference engine and that flow's list is
    # reset to empty.
    flows = defaultdict(list)
    # Maps each flow (four-tuple) to a tuple of fairness state:
    #   (is_fair, response)
    # where is_fair is either -1 (no label), 0 (below fair), 1 (approximately
    # fair), or 2 (above fair) and response is a either ACK pacing rate or RWND
    #  value.
    fairness_db = defaultdict(lambda: (-1, 0))

    # Lock for the packet input data structures (e.g., "flows").
    lock_i = threading.Lock()
    # Lock for the fairness_db.
    lock_f = threading.Lock()

    # Set up the inference thread.
    gen = threading.Thread(
        target=inference_loop,
        args=(lock_i, lock_f, flows, fairness_db, args.limit,
              net, args.disable_inference))
    gen.start()

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
    bpf = BPF(text=bpf_text)
    bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")

    # This function will be called to process an event.
    def process_event(cpu, data, size):
        receive_packet(lock_i, flows, bpf["pkts"].event(data))

    print("Running...press Control-C to end")
    # Loop with callback to process_event().
    bpf["pkts"].open_perf_buffer(process_event)
    while True:
        try:
            if args.interval_ms is not None:
                time.sleep(args.interval_ms * 1000)
            bpf.perf_buffer_poll()
        except KeyboardInterrupt:
            break

    print("\nFlows:")
    for flow, pkts in sorted(flows.items()):
        print("\t", flow_to_str(flow), len(pkts))

    global DONE
    DONE = True
    gen.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
