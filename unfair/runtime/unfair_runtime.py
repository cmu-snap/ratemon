#!/usr/bin/python3
"""Monitors incoming TCP flows to detect unfairness."""

from argparse import ArgumentParser
from os import path
import pickle
import socket
import struct
import sys
import threading
import time

from collections import defaultdict

import torch
from bcc import BPF
import numpy as np
import pyroute2

# from pyroute2 import IPRoute

from unfair.model import data, defaults, features, gen_features, models, utils


def ip_str_to_int(ip_str):
    """Convert an IP address string in dotted-quad notation to an integer."""
    return struct.unpack("<L", socket.inet_aton(ip_str))[0]


LOCALHOST = ip_str_to_int("127.0.0.1")
DONE = False
# Flows that have not received a new packet in this many seconds will be
# garbage collected.
OLD_THRESH_SEC = 5 * 60


def int_to_ip_str(ip_int):
    """Convert an IP address int into a dotted-quad string."""
    # Use "<" (little endian) instead of "!" (network / big endian) because the
    # IP addresses are already stored in little endian.
    return socket.inet_ntoa(struct.pack("<L", ip_int))


class Flow:
    """Represents a single flow, identified by a four-tuple.

    A Flow object is created when we first receive a packet for the flow. A Flow object
    is deleted when it has been five minutes since we have checked this flow's
    fairness.

    Must acquire self.lock before accessing members.
    """

    def __init__(self, fourtuple):
        self.lock = threading.RLock()
        self.fourtuple = fourtuple
        self.packets = []
        # The timestamp of the last packet on which we have run inference.
        self.latest_time_us = 0
        self.label = defaults.Class.APPROX_FAIR
        self.decision = (defaults.Decision.NOT_PACED, None)

    @staticmethod
    def flow_to_str(fourtuple):
        """Convert a flow four-tuple into a string."""
        saddr, daddr, sport, dport = fourtuple
        return f"{int_to_ip_str(saddr)}:{sport} -> {int_to_ip_str(daddr)}:{dport}"

    def __str__(self):
        """Create a string representation of this flow."""
        return Flow.flow_to_str(self.fourtuble)


def flow_data_to_str(dat):
    """Convert a flow data tuple into a string."""
    (
        seq,
        srtt_us,
        tsval,
        tsecr,
        total_bytes,
        ihl_bytes,
        thl_bytes,
        payload_bytes,
        time_us,
    ) = dat
    return (
        f"seq: {seq}, srtt: {srtt_us} us, tsval: {tsval}, tsecr: {tsecr}, "
        f"total: {total_bytes} B, IP header: {ihl_bytes} B, "
        f"TCP header: {thl_bytes} B, payload: {payload_bytes} B, "
        f"time: {time.ctime(time_us / 1e3)}"
    )


def receive_packet(flows, flows_lock, pkt):
    """Ingest a new packet, identify its flow, and store it.

    lock_i protects flows.
    """
    print("new packet")
    # Skip packets on the loopback interface.
    # if LOCALHOST in (pkt.flow.saddr, pkt.flow.daddr):
    if LOCALHOST in (pkt.saddr, pkt.daddr):
        return

    # flow = (pkt.flow.saddr, pkt.flow.daddr, pkt.flow.sport, pkt.flow.dport)
    flow = (pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
    dat = (
        pkt.seq,
        pkt.srtt_us,
        pkt.tsval,
        pkt.tsecr,
        pkt.total_bytes,
        pkt.ihl_bytes,
        pkt.thl_bytes,
        pkt.payload_bytes,
        pkt.time_us,
    )

    flows_lock.acquire()
    if flow not in flows:
        flows[flow] = Flow(flow)
    flows[flow].packets.append(dat)
    flows_lock.release()

    print(f"{Flow.flow_to_str(flow)} --- {flow_data_to_str(dat)}")


def check_loop(
    flows, flows_lock, net, args
):
    """Periodically evaluate flow fairness.

    Intended to be run as the target function of a thread.
    """
    try:
        while not DONE:
            check_flows(
                flows, flows_lock, args.limit, net, args.disable_inference, args.debug
            )

            flows_lock.acquire()
            print(
                "Current decisions:\n"
                + "\n".join(
                    f"\t{flow}: {flow.decision}" for flow in flows
                )
            )
            flows_lock.release()
            # time.sleep(1)
    except KeyboardInterrupt:
        return


def check_flows(
    flows, flows_lock, limit, net, disable_inference, debug=False
):
    """Identify flows that are ready to be checked, and check them.

    Remove old and empty flows.
    """
    print("Examining flows...")
    to_remove = []
    to_check = []
    print(f"Found {len(flows)} flows total")
    for fourtuple, flow in flows.items():
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on to the next flow.
        if flow.lock.acquire(blocking=False):
            print(f"{fourtuple} - {len(flow.packets)}")
            if time.time() * 1e6 - flow.latest_time_us > (OLD_THRESH_SEC * 1e6):
                # Remove flows with no packets and flows that have not received
                # a new packet in five seconds.
                to_remove.append(fourtuple)
            if len(flow.packets) >= limit:
                # Plan to run inference on "full" flows.
                to_check.append(fourtuple)  # Garbage collection.
            flow.lock.release()
    # Garbage collection.
    print(f"Removing {len(to_remove)} flows...")
    flows_lock.acquire()
    for fourtuple in to_remove:
        del flows[fourtuple]
    flows_lock.release()

    print(f"Checking {len(to_check)} flows...")
    for fourtuple in to_check:
        print(f"Checking flow {fourtuple}")
        flows[fourtuple].lock.acquire()
        # packets = flows[fourtuple].packets
        # flows[fourtuple].packets = []
        # Do not hold lock while running inference.
        if not disable_inference:
            check_flow(net, flow, debug)
        flows_lock.release()


def featurize(net, flow, pkts, debug=False):
    """Compute features for the provided list of packets.

    Returns a structured numpy array.
    """
    fets = gen_features.parse_received_acks(net.in_spc, flow, pkts, debug)
    data.replace_unknowns(fets, isinstance(net, models.HistGbdtSklearnWrapper))
    return fets


def packets_to_ndarray(pkts):
    """Reorganize a list of packet metrics into a structured numpy array."""
    (
        seqs,
        srtts_us,
        tsvals,
        tsecrs,
        totals_bytes,
        _,
        _,
        payloads_bytes,
        times_us,
    ) = zip(*pkts)
    pkts = utils.make_empty(len(seqs), additional_dtype=[(features.SRTT_FET, "int32")])
    pkts[features.SEQ_FET] = seqs
    pkts[features.ARRIVAL_TIME_FET] = times_us
    pkts[features.TS_1_FET] = tsvals
    pkts[features.TS_2_FET] = tsecrs
    pkts[features.PAYLOAD_FET] = payloads_bytes
    pkts[features.WIRELEN_FET] = totals_bytes
    pkts[features.SRTT_FET] = srtts_us
    return pkts


def make_decision(flow, pkts_ndarray):
    """Make a flow unfairness mitigation decision.

    Base the decision on the flow's label and existing decision. Use the flow's packets
    to calculate any necessary flow metrics, such as the throughput.

    TODO: Instead of passing in using the flow's packets, pass in the features and make
          sure that they include the necessary columns.
    """
    flow.lock.acquire()
    tput_bps = utils.safe_tput_bps(pkts_ndarray, 0, len(pkts_ndarray) - 1)

    if flow.label == defaults.Class.ABOVE_FAIR:
        # This flow is sending too fast. Force the sender to halve its rate.
        new_decision = (defaults.Decision.PACED, tput_bps / 2)
    elif flow.decision == defaults.Decision.PACED:
        # We are already pacing this flow.
        if flow.label == defaults.Class.BELOW_FAIR:
            # If we are already pacing this flow but we are being too aggressive, then
            # let it send faster.
            new_decision = (defaults.Decision.PACED, tput_bps * 1.5)
        else:
            # If we are already pacing this flow and it is behaving as desired, then
            # all is well. Retain the existing pacing decision.
            new_decision = flow.decision
    else:
        # This flow is not already being paced and is not behaving unfairly, so leave
        # it alone.
        new_decision = (defaults.Decision.NOT_PACED, None)

    flow.decision = new_decision
    flow.lock.release()


def condense_labels(labels):
    """Combine multiple labels into a single label.

    For example, smooth the labels by selecting the average label.

    Currently, this simply selects the last label.
    """
    assert len(labels) > 0, "labels cannot be empty"
    return labels[-1]


def check_flow(net, flow, debug=False):
    """Determine whether a flow is unfair and how to mitigate it.

    Runs inference on a flow's packets and determines the appropriate ACK
    pacing for the flow. Updates the flow's fairness record.
    """
    flow.lock.acquire()

    # Discard all but the most recent 100 packets.
    if len(flow.packets) > 100:
        flow.packets = flow.packets[-100:]

    pkts_ndarray = packets_to_ndarray(flow.packets)

    start_time_s = time.time()
    labels = inference(net, flow, pkts_ndarray, debug)
    print(f"Inference took: {(time.time() - start_time_s) * 1e3} ms")

    flow.label = condense_labels(labels)
    make_decision(flow, pkts_ndarray)

    # Clear the flow's packets.
    flow.packets = []

    print(f"Report for flow {flow}: {flow.label}, {flow.decision}")
    flow.lock.release()


def inference(net, flow, pkts, debug=False):
    """Run inference on a flow's packets.

    Returns a label: below fair, approximately fair, above fair.
    """
    preds = net.predict(
        torch.tensor(utils.clean(featurize(net, flow, pkts, debug)), dtype=torch.float)
    )
    return [defaults.Class(pred) for pred in preds]


def _main():
    parser = ArgumentParser(description="Squelch unfair flows.")
    parser.add_argument("-i", "--interval-ms", help="Poll interval (ms)", type=float)
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Print debugging info"
    )
    parser.add_argument(
        "-i",
        "--interface",
        help='The network interface to attach to (e.g., "eno1").',
        required=True,
        type=str,
    )
    parser.add_argument(
        "-l",
        "--limit",
        default=100,
        help=("The number of packets to accumulate for a flow between inference runs."),
        type=int,
    )
    parser.add_argument(
        "-s",
        "--disable-inference",
        action="store_true",
        help="Disable periodic inference.",
    )
    parser.add_argument(
        "--model",
        choices=models.MODEL_NAMES,
        help="The model to use.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f", "--model-file", help="The trained model to use.", required=True, type=str
    )
    parser.add_argument(
        "--reaction-strategy",
        choices=defaults.ReactionStrategy.choices(),
        default=defaults.ReactionStrategy.to_str(defaults.ReactionStrategy.MIMD),
        help="The reaction/feedback strategy to use.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--mitigation-strategy",
        choices=defaults.MitigationStrategy.choices(),
        default=defaults.MitigationStrategy.to_str(
            defaults.MitigationStrategy.RWND_TUNING
        ),
        help="The unfairness mitigation strategy to use.",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    assert args.limit > 0, f'"--limit" must be greater than 0 but is: {args.limit}'
    assert path.isfile(args.model_file), f"Model does not exist: {args.model_file}"

    net = models.MODELS[args.model]()
    with open(args.model_file, "rb") as fil:
        net.net = pickle.load(fil)

    # Maps each flow (four-tuple) to a list of packets for that flow. New
    # packets are appended to the ends of these lists. Periodically, a flow's
    # packets are consumed by the inference engine and that flow's list is
    # reset to empty.
    flows = dict()  # defaultdict(list)
    # Maps each flow (four-tuple) to a tuple of fairness state:
    #   (is_fair, response)
    # where is_fair is either -1 (no label), 0 (below fair), 1 (approximately
    # fair), or 2 (above fair) and response is a either ACK pacing rate or RWND
    #  value.
    # fairness_db = dict()  # defaultdict(lambda: (-1, 0))

    # Lock for the packet input data structures (e.g., "flows"). Only lock when adding or removing flows.
    flows_lock = threading.Lock()

    # Set up the inference thread.
    check_thread = threading.Thread(
        target=check_loop,
        args=(
            flows,
            flows_lock,
            net,
            args
        ),
    )
    check_thread.start()

    # Load BPF text.
    bpf_flp = path.join(
        path.abspath(path.dirname(__file__)),
        path.basename(__file__).strip().split(".")[0] + ".c",
    )
    if not path.isfile(bpf_flp):
        print(f"Could not find BPF program: {bpf_flp}")
        return 1
    print(f"Loading BPF program: {bpf_flp}")
    with open(bpf_flp, "r", encoding="utf-8") as fil:
        bpf_text = fil.read()
    if args.debug:
        print(bpf_text)

    # Load BPF program.
    bpf = BPF(text=bpf_text)
    bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
    # bpf.attach_kprobe(event="tc_egress", fn_name="trace_tc_egress")
    egress_fn = bpf.load_func("handle_egress", BPF.SCHED_ACT)

    # Configure unfairness mitigation strategy.

    ipr = pyroute2.IPRoute()
    ifindex = ipr.link_lookup(ifname=args.interface)
    # ipr.tc("add", "pfifo", 0, "1:")
    # ipr.tc("add-filter", "bpf", 0, ":1", fd=egress_fn.fd, name=egress_fn.name, parent="1:")

    # There can also be a chain of actions, which depend on the return
    # value of the previous action.
    action = dict(kind="bpf", fd=egress_fn.fd, name=egress_fn.name, action="ok")
    # Add the action to a u32 match-all filter
    ipr.tc("add", "htb", ifindex, 0x10000, default=0x200000)
    ipr.tc(
        "add-filter",
        "u32",
        ifindex,
        parent=0x10000,
        prio=10,
        protocol=pyroute2.protocols.ETH_P_ALL,  # Every packet
        target=0x10020,
        keys=["0x0/0x0+0"],
        action=action,
    )

    # This function will be called to process an event from the BPF program.
    def process_event(cpu, dat, size):
        receive_packet(flows, flows_lock, bpf["pkts"].event(dat))

    # Loop with callback to process_event().
    print("Running...press Control-C to end")
    bpf["pkts"].open_perf_buffer(process_event)
    while True:
        try:
            if args.interval_ms is not None:
                time.sleep(args.interval_ms / 1000)
            bpf.perf_buffer_poll()
        except KeyboardInterrupt:
            break

    # print("\nFlows:")
    # for flow, pkts in sorted(flows.items()):
    #     print("\t", flow_to_str(flow), len(pkts))

    global DONE
    DONE = True
    check_thread.join()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
