"""Monitors incoming TCP FLOWS to detect unfairness."""

from argparse import ArgumentParser
import atexit
import multiprocessing
from os import path
import queue
import signal
from struct import unpack
import sys
import threading
import time

import netifaces as ni
import pcapy

from unfair.model import utils
from unfair.runtime import flow_utils, inference, mitigation_strategy, reaction_strategy
from unfair.runtime.mitigation_strategy import MitigationStrategy
from unfair.runtime.reaction_strategy import ReactionStrategy


LOCALHOST = utils.ip_str_to_int("127.0.0.1")
# Flows that have not received a new packet in this many seconds will be
# garbage collected.
# 5 minutes
OLD_THRESH_SEC = 5 * 60

# Maps each flow (four-tuple) to a list of packets for that flow. New
# packets are appended to the ends of these lists. Periodically, a flow's
# packets are consumed by the inference engine and that flow's list is
# reset to empty.
FLOWS = {}
# Lock for the packet input data structures (e.g., "flows"). Only acquire this lock
# when adding, removing, or iterating over flows; no need to acquire this lock when
# updating a flow object.
FLOWS_LOCK = threading.RLock()

NUM_PACKETS = 0
TOTAL_BYTES = 0
MY_IP = None
MANAGER = None


def receive_packet_pcapy(header, packet):
    """Process a packet sniffed by pcapy.

    Extract relevant fields, compute the RTT (for incoming packets), and sort the
    packet into the appropriate flow.

    Inspired by: https://www.binarytides.com/code-a-packet-sniffer-in-python-with-pcapy-extension/
    """
    # Note: We do not need to check that this packet IPv4 and TCP because we already do that
    # with a filter.
    if header is None:
        return

    # The Ethernet header is 14 bytes.
    ehl = 14
    # The basic IPv4 header is 20 bytes.
    ip = unpack("!BBHHHBBH4s4s", packet[ehl : ehl + 20])
    version_ihl = ip[0]

    ihl = (version_ihl & 0xF) * 4
    tcp_offset = ehl + ihl
    # The basic TCP header is 20 bytes.
    tcp = unpack("!HHLLBBHHH", packet[tcp_offset : tcp_offset + 20])
    saddr = int.from_bytes(ip[8], byteorder="little", signed=False)
    daddr = int.from_bytes(ip[9], byteorder="little", signed=False)
    incoming = daddr == MY_IP
    sport = tcp[0]
    dport = tcp[1]

    # Ignore some packets.
    if (
        # Ignore packets on the loopback interface.
        (saddr == LOCALHOST or daddr == LOCALHOST)
        # Accept packets on local ports 9998, 9999, and 10000 only.
        or (incoming and (dport < 9998 or dport > 10000))
        or (not incoming and (sport < 9998 or sport > 10000))
    ):
        return

    thl = (tcp[4] >> 4) * 4
    total_bytes = header.getlen()
    time_s, time_us = header.getts()
    time_s = time_s + time_us / 1e6  # arrival time in microseconds

    # Parse the TCP timestamp option.
    tsval = None
    tsecr = None
    offset = tcp_offset + 20
    while offset < tcp_offset + thl:
        option_type = unpack("!B", packet[offset : offset + 1])[0]
        if option_type == 0:
            break
        if option_type == 1:
            offset += 1
            continue
        option_length = unpack("!B", packet[offset + 1 : offset + 2])[0]
        if option_type == 8:
            tsval, tsecr = unpack("!II", packet[offset + 2 : offset + option_length])
            break
        offset += option_length

    # The fourtuple is always (local addr, remote addr, local port, remote port).
    fourtuple = (
        (daddr, saddr, dport, sport) if incoming else (saddr, daddr, sport, dport)
    )
    with FLOWS_LOCK:
        if fourtuple in FLOWS:
            flow = FLOWS[fourtuple]
        else:
            flow = flow_utils.Flow(fourtuple)
            FLOWS[fourtuple] = flow
    with flow.ingress_lock:
        rtt_s = -1
        if incoming:
            if tsval is not None and tsecr is not None:
                # Use the TCP timestamp option to calculate the RTT.
                if tsecr in flow.sent_tsvals:
                    rtt_s = flow.sent_tsvals[tsecr] - time_s
                    del flow.sent_tsvals[tsecr]

            flow.incoming_packets.append(
                (
                    tcp[2],  # seq
                    rtt_s,
                    total_bytes,
                    total_bytes - (ehl + ihl + thl),  # payload bytes
                    time_s,
                )
            )

            # Only give up credit for processing incoming packets.
            global NUM_PACKETS, TOTAL_BYTES
            NUM_PACKETS += 1
            TOTAL_BYTES += total_bytes
        else:
            # Track outgoing tsval for use later.
            flow.sent_tsvals[tsval] = time_s


def check_loop(args, que, inference_flags, done):
    """Periodically evaluate flow fairness.

    Intended to be run as the target function of a thread.
    """
    try:
        while not done.is_set():
            last_check = time.time()
            check_flows(args, que, inference_flags)

            if args.inference_interval_ms is not None:
                time.sleep(
                    max(
                        0,
                        # The time remaining in the interval, accounting for the time
                        # spent checking the flows.
                        args.inference_interval_ms / 1e3 - (time.time() - last_check),
                    )
                )
    except KeyboardInterrupt:
        print("Cancelled.")
        done.set()


def check_flows(args, que, inference_flags):
    """Identify flows that are ready to be checked, and check them.

    Remove old and empty flows.
    """
    to_remove = []
    to_check = []

    # Need to acquire FLOWS_LOCK while iterating over FLOWS.
    with FLOWS_LOCK:
        for fourtuple, flow in FLOWS.items():
            # Try to acquire the lock for this flow. If unsuccessful, do not block; move
            # on to the next flow.
            if flow.ingress_lock.acquire(blocking=False):
                try:
                    if (
                        len(flow.incoming_packets) > 0
                        # If we have specified a minimum number of packets to run
                        # inference, then check that.
                        and (
                            args.min_packets is None
                            or len(flow.incoming_packets) > args.min_packets
                        )
                    ):
                        # Plan to run inference on this flows.
                        to_check.append(fourtuple)
                    elif flow.latest_time_sec and (
                        time.time() - flow.latest_time_sec > OLD_THRESH_SEC
                    ):
                        # Remove old flows that have not been selected for inference in
                        # a while.
                        to_remove.append(fourtuple)
                finally:
                    flow.ingress_lock.release()
            else:
                print(f"Could not acquire lock for flow: {flow}")
        # Garbage collection.
        for fourtuple in to_remove:
            del FLOWS[fourtuple]
            if fourtuple in inference_flags:
                del inference_flags[fourtuple]

    for fourtuple in to_check:
        flow = FLOWS[fourtuple]
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on
        # to the next flow.
        if flow.ingress_lock.acquire(blocking=False):
            try:
                check_flow(fourtuple, args, que, inference_flags)
            finally:
                flow.ingress_lock.release()
        else:
            print(f"Could not acquire lock for flow: {flow}")


def check_flow(fourtuple, args, que, inference_flags):
    """Determine whether a flow is unfair and how to mitigate it.

    Runs inference on a flow's packets and determines the appropriate ACK
    pacing for the flow. Updates the flow's fairness record.
    """
    flow = FLOWS[fourtuple]
    with flow.ingress_lock:
        # Discard all but the most recent few packets.
        if len(flow.incoming_packets) > args.min_packets:
            flow.incoming_packets = flow.incoming_packets[-args.min_packets :]
        # Record the time when we check this flow.
        flow.latest_time_sec = time.time()

        if fourtuple not in inference_flags:
            inference_flags[fourtuple] = MANAGER.Value(typecode="i", value=0)
        # Submit new packets for inference only if inference is not already scheduled
        # or running for this flow.
        if inference_flags[fourtuple].value == 0:
            inference_flags[fourtuple].value = 1
            try:
                print(
                    f"Scheduling inference on most recent {len(flow.incoming_packets)} "
                    f"packets for flow: {flow}"
                )
                if not args.disable_inference:
                    que.put((fourtuple, flow.incoming_packets), block=False)
            except queue.Full:
                print("Warning: Inference queue full!")
            else:
                # Reset the flow.
                flow.incoming_packets = []
        else:
            print(f"Skipping inference for flow: {flow}")


def pcapy_sniff(interface):
    """Use pcapy to sniff packets from a specific interface."""
    # Set the snapshot length to the maximum size of the Ethernet, IPv4, and TCP
    # headers. Do not put the interface into promiscuous mode. Set the timeout to
    # 1000 ms, which actually allows batching up to 1000 ms
    # of packets from the kernel.
    pcap = pcapy.open_live(interface, 14 + 60 + 60, 0, 1000)

    def pcapy_cleanup():
        """Close the pcapy reader."""
        print("Stopping pcapy sniffing...")
        pcap.close()

    atexit.register(pcapy_cleanup)

    # Drop non-IPv4/TCP packets.
    pcap.setfilter("ip and tcp")

    print("Running...press Control-C to end")
    last_time_s = time.time()
    last_num_packets = NUM_PACKETS
    last_total_bytes = TOTAL_BYTES
    while True:
        receive_packet_pcapy(*pcap.next())

        now_s = time.time()
        delta_time_s = now_s - last_time_s
        if delta_time_s >= 10:
            delta_num_packets = NUM_PACKETS - last_num_packets
            delta_total_bytes = TOTAL_BYTES - last_total_bytes
            last_time_s = now_s
            last_num_packets = NUM_PACKETS
            last_total_bytes = TOTAL_BYTES
            print(
                f"Performance report --- {delta_num_packets / delta_time_s:.2f} pps, "
                f"{8 * delta_total_bytes / delta_time_s / 1e6:.2f} Mbps"
            )


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(description="Squelch unfair flows.")
    parser.add_argument(
        "-p",
        "--poll-interval-ms",
        help="Packet poll interval (sleep time; ms).",
        required=False,
        type=float,
    )
    parser.add_argument(
        "-i",
        "--inference-interval-ms",
        help="Hard interval to enforce between checking flows (start to start; ms).",
        required=False,
        type=float,
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Print debugging info"
    )
    parser.add_argument(
        "--interface",
        help='The network interface to attach to (e.g., "eno1").',
        required=True,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--disable-inference",
        action="store_true",
        help="Disable periodic inference.",
    )
    parser.add_argument(
        "-f", "--model-file", help="The trained model to use.", required=True, type=str
    )
    parser.add_argument(
        "--reaction-strategy",
        choices=reaction_strategy.choices(),
        default=reaction_strategy.to_str(ReactionStrategy.MIMD),
        help="The reaction/feedback strategy to use.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--schedule",
        help=(
            f"A CSV file specifying a pacing schedule to use with the "
            f'"{reaction_strategy.to_str(ReactionStrategy.FILE)}" reaction strategy. '
            "Each line should be of the form: <start time (seconds)>,<RWND>. "
            "*Note* Allow at least 10 seconds for warmup."
        ),
        required=False,
        type=str,
    )
    parser.add_argument(
        "--mitigation-strategy",
        choices=mitigation_strategy.choices(),
        default=mitigation_strategy.to_str(MitigationStrategy.RWND_TUNING),
        help="The unfairness mitigation strategy to use.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--cgroup",
        help=(
            "The cgroup that will contain processes to monitor. "
            "Practically speaking, this is the path to a directory. "
            "BCC and eBPF must know this to monitor/set TCP header options, "
            "(in particular, the TCP window scale)."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--min-packets",
        default=100,
        help="The minimum packets required to run inference on a flow.",
        required=False,
        type=int,
    )
    args = parser.parse_args()
    args.reaction_strategy = reaction_strategy.to_strat(args.reaction_strategy)
    args.mitigation_strategy = mitigation_strategy.to_strat(args.mitigation_strategy)
    assert (
        args.min_packets > 0
    ), f'"--min-packets" must be greater than 0 but is: {args.min_packets}'

    assert (
        args.reaction_strategy != ReactionStrategy.FILE or args.schedule is not None
    ), "Must specify schedule file."
    if args.schedule is not None:
        assert path.isfile(args.schedule), f"File does not exist: {args.schedule}"
        args.schedule = reaction_strategy.parse_pacing_schedule(args.schedule)

    assert path.isdir(args.cgroup), f'"--cgroup={args.cgroup}" is not a directory.'
    return args


def run(args, manager):
    """Core logic."""
    # Create sychronized data structures.
    que = manager.Queue()
    inference_flags = manager.dict()
    # Flag to trigger threads/processes to terminate.
    done = manager.Event()
    done.clear()

    # Create the thread that will monitor flows and decide when to run inference.
    check_thread = threading.Thread(
        target=check_loop,
        args=(args, que, inference_flags, done),
    )
    check_thread.start()

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Create the process that will run inference.
    inference_proc = multiprocessing.Process(
        target=inference.run, args=(args, que, inference_flags, done)
    )
    inference_proc.start()
    signal.signal(signal.SIGINT, original_sigint_handler)

    # Look up my IP address to use when filtering packets.
    global MY_IP
    MY_IP = utils.ip_str_to_int(ni.ifaddresses(args.interface)[ni.AF_INET][0]["addr"])

    # The current thread will sniff packets.
    try:
        pcapy_sniff(args.interface)
    except KeyboardInterrupt:
        print("Cancelled.")
        done.set()

    check_thread.join()
    inference_proc.join()


def _main():
    args = parse_args()
    with multiprocessing.Manager() as manager:
        global MANAGER
        MANAGER = manager
        return run(args, manager)


if __name__ == "__main__":
    sys.exit(_main())
