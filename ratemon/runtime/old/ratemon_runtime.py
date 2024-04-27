"""Monitors incoming TCP flow to detect if they are above their target rate."""

from argparse import ArgumentParser
import atexit
import logging
import multiprocessing
import multiprocessing.managers
from os import path
import queue
import signal
from struct import unpack
import sys
import threading
import time
import typing

import netifaces as ni
import pcapy

from ratemon.model import features, utils
from ratemon.runtime import (
    flow_utils,
    mitigation_strategy,
    policies,
    policy_engine,
    reaction_strategy,
)
from ratemon.runtime.policies import Policy
from ratemon.runtime.mitigation_strategy import MitigationStrategy
from ratemon.runtime.reaction_strategy import ReactionStrategy


LOCALHOST = utils.ip_str_to_int("127.0.0.1")
# Flows that have not received a new packet in this many seconds will be
# garbage collected.
# 1 minute
OLD_THRESH_SEC = 1 * 60

# Maps each flow (four-tuple) to a list of packets for that flow. New
# packets are appended to the ends of these lists. Periodically, a flow's
# packets are consumed by the policy engine and that flow's list is
# reset to empty.
FLOWS = flow_utils.FlowDB()

MY_IP = None
MANAGER = None

LOSS_EVENT_INTERVALS: typing.List[int] = []

EPOCH = 0


def main():
    args = parse_args()
    logging.basicConfig(
        filename=args.log,
        filemode="w",
        format="%(asctime)s %(levelname)s \t| %(message)s",
        level=logging.DEBUG,
    )
    with multiprocessing.Manager() as manager:
        global MANAGER
        MANAGER = manager
        try:
            return run(args)
        except:
            logging.exception("Unknown error in RateMon runtime!")
            return 1


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(description="Constrain flows above their target rate.")
    parser.add_argument(
        "-p",
        "--poll-interval-ms",
        help="Packet poll interval (sleep time; ms).",
        required=False,
        type=float,
    )
    parser.add_argument(
        "-i",
        "--check-interval-ms",
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
        "-f", "--model-file", help="The trained model to use.", required=False, type=str
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
        "--skip-localhost", action="store_true", help="Skip packets to/from localhost."
    )
    parser.add_argument(
        "--log", help="The main log file to write to.", required=True, type=str
    )
    parser.add_argument(
        "--batch-size",
        default=10,
        help="The number of flows to process in parallel.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--listen-ports",
        nargs="+",
        default=[],
        help="List of ports to monitor.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--smoothing-window",
        default=1,
        help="Process this many packets from each flow and smooth the results.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--policy",
        choices=policies.choices(),
        help="Different types of receiver policy, depending on workload.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--schedule",
        help=(
            "(--policy==staticrwnd) A CSV file specifying an RWND tuning schedule to "
            "use. Each line should be of the form: <start time (seconds)>,<RWND>. "
            "*Note* Allow at least 10 seconds for warmup."
        ),
        required=False,
        type=str,
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
        "--mitigation-strategy",
        choices=mitigation_strategy.choices(),
        default=mitigation_strategy.to_str(MitigationStrategy.RWND_TUNING),
        help="The mitigation strategy to use.",
        required=False,
        type=str,
    )
    args = parser.parse_args()
    args.policy = policies.to_policy(args.policy)
    args.reaction_strategy = reaction_strategy.to_strat(args.reaction_strategy)
    args.mitigation_strategy = mitigation_strategy.to_strat(args.mitigation_strategy)

    assert (
        args.policy != Policy.STATIC_RWND or args.schedule is not None
    ), "Must specify schedule file."
    if args.schedule is not None:
        assert path.isfile(args.schedule), f"File does not exist: {args.schedule}"
        args.schedule = reaction_strategy.parse_static_rwnd_schedule(args.schedule)
    assert (
        args.batch_size > 0
    ), f'"--batch-size" must be greater than 0, but is: {args.batch_size}'
    assert (
        args.smoothing_window >= 1
    ), f"Smoothing window must be >= 1, but is: {args.smoothing_window}"

    assert path.isdir(args.cgroup), f'"--cgroup={args.cgroup}" is not a directory.'
    assert (
        args.policy != Policy.FLOWPOLICY
    ) or args.model_file is not None, (
        f"{policies.to_str(Policy.FLOWPOLICY)} requires '--model-file'. "
    )

    return args


def run(args):
    """Core logic."""
    # Need to load the model to check the input features to see the longest window.
    in_spc = policies.get_model_for_policy(args.policy, args.model_file).in_spc
    longest_window = max(
        (
            features.parse_win_metric(fet)[1]
            for fet in in_spc
            if "windowed" in fet and "minRtt" in fet
        ),
        default=0,
    )
    logging.info("Longest minRTT window: %d", longest_window)

    # Fill in feature dependencies.
    all_features = list(
        zip(
            *list(
                set(features.PARSE_PACKETS_FETS)
                | set(
                    features.feature_names_to_dtype(features.fill_dependencies(in_spc))
                )
            )
        )
    )[0]

    global LOSS_EVENT_INTERVALS
    LOSS_EVENT_INTERVALS = list(
        {
            features.parse_win_metric(fet)[1]
            for fet in all_features
            if "windowed" in fet
            and (
                fet.startswith(features.LOSS_EVENT_RATE_FET)
                or fet.startswith(features.SQRT_LOSS_EVENT_RATE_FET)
                or fet.startswith(features.MATHIS_TPUT_LOSS_EVENT_RATE_FET)
            )
        }
    )
    logging.info("Loss event intervals: %s", LOSS_EVENT_INTERVALS)

    # Create sychronized data structures.
    assert isinstance(MANAGER, multiprocessing.managers.SyncManager)
    que = MANAGER.Queue()
    flags = MANAGER.dict()
    # Flag to trigger threads/processes to terminate.
    done = MANAGER.Event()
    done.clear()

    # Create the thread that will monitor flows and decide when to run the policy.
    check_thread = threading.Thread(
        target=check_loop,
        args=(args, longest_window, que, flags, done),
    )
    check_thread.start()

    # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Create the process that will run the policy engine.
    policy_proc = multiprocessing.Process(
        target=policy_engine.run, args=(args, que, flags, done)
    )
    policy_proc.start()
    # signal.signal(signal.SIGINT, original_sigint_handler)

    # Create the thread that will sniff packets from the network interface.
    sniff_thread = threading.Thread(target=pcapy_sniff_main, args=(args, done))
    sniff_thread.start()

    logging.info("Running...press Control-C to end")
    print("Running...press Control-C to end")

    def signal_handler(sig, frame):
        logging.info("Main process: You pressed Ctrl+C!")
        done.set()
        print(
            "RateMon stopping. If RateMon does not stop, then it is likely blocked "
            "waiting to receive a packet. Send a packet to unblock RateMon, "
            "or send SIGKILL."
        )
        # raise RuntimeError()

    signal.signal(signal.SIGINT, signal_handler)

    check_thread.join()
    policy_proc.join()
    sniff_thread.join()
    return 0


def check_loop(args, longest_window, que, flags, done):
    """Periodically evaluate flow fairness.

    Intended to be run as the target function of a thread.
    """
    try:
        while not done.is_set():
            last_check = time.time()
            check_flows(args, longest_window, que, flags)

            if args.check_interval_ms is not None:
                time.sleep(
                    max(
                        0,
                        # The time remaining in the interval, accounting for the time
                        # spent checking the flows.
                        args.check_interval_ms / 1e3 - (time.time() - last_check),
                    )
                )
    except KeyboardInterrupt:
        logging.info("Cancelled.")
        done.set()
    except Exception as exc:
        logging.exception("Other exception")
        raise exc
    finally:
        logging.info("Out of check loop")


def check_flows(args, longest_window, que, flags):
    """Identify flows that are ready to be checked, and check them.

    Remove old and empty flows.
    """
    global EPOCH
    EPOCH += 1

    to_remove = set()
    to_check = set()

    # Need to acquire FLOWS.lock while iterating over FLOWS.
    with FLOWS.lock:
        for fourtuple, flow in FLOWS.items():
            # A fourtuple might add other fourtuples to to_check if
            # we are using ServicePolicy.
            if fourtuple in to_check:
                continue

            # Try to acquire the lock for this flow. If unsuccessful, do not block; move
            # on to the next flow.
            if flow.ingress_lock.acquire(blocking=False):
                try:
                    if flow.incoming_packets:
                        logging.info(
                            (
                                "Flow %s - packets: %d, min_rtt: %.2f us, "
                                "longest window: %d, required span: %.2f s, "
                                "span: %.2f s"
                            ),
                            flow,
                            len(flow.incoming_packets),
                            flow.min_rtt_us,
                            longest_window,
                            flow.min_rtt_us * longest_window / 1e6,
                            (flow.incoming_packets[-1][4] - flow.incoming_packets[0][4])
                            / 1e6,
                        )
                    else:
                        logging.info("No incoming packets for flow %s", flow)

                    if flow.is_ready(args.smoothing_window, longest_window):
                        if args.policy == Policy.SERVICEPOLICY:
                            # Only want to add this flow if all the flows from this
                            # sender are ready.
                            if FLOWS.sender_okay(
                                flow.flowkey.remote_addr,
                                args.smoothing_window,
                                longest_window,
                            ):
                                flows_to_add = FLOWS.get_flows_from_sender(
                                    flow.flowkey.remote_addr
                                )
                                to_check |= flows_to_add
                                logging.info(
                                    (
                                        "Sender %s is ready. "
                                        "Submitting flows to policy engine: %s"
                                    ),
                                    utils.int_to_ip_str(flow.flowkey.remote_addr),
                                    flows_to_add,
                                )

                        else:
                            # Plan to run the policy on this flow.
                            logging.info(
                                "Flow is ready. Submitting to policy engine: %s",
                                fourtuple,
                            )
                            to_check.add(fourtuple)
                    elif flow.latest_time_sec and (
                        time.time() - flow.latest_time_sec > OLD_THRESH_SEC
                    ):
                        # Remove old flows that have not been sent to the policy engine
                        # in a while.
                        to_remove.add(fourtuple)
                finally:
                    flow.ingress_lock.release()
            else:
                logging.warning("Could not acquire lock for flow: %s", flow)
        # Garbage collection.
        for fourtuple in to_remove:
            logging.info("Removing flow: %s", utils.flow_to_str(fourtuple))
            del FLOWS[fourtuple]
            if fourtuple in flags:
                del flags[fourtuple]
            que.put(("remove", fourtuple))

    for fourtuple in to_check:
        flow = FLOWS[fourtuple]
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on
        # to the next flow.
        if flow.ingress_lock.acquire(blocking=False):
            try:
                check_flow(fourtuple, args, longest_window, que, flags, epoch=EPOCH)
            finally:
                flow.ingress_lock.release()
        else:
            logging.warning("Could not acquire lock for flow: %s", flow)


def check_flow(fourtuple, args, longest_window, que, flags, epoch=0):
    """Determine whether a flow is ready to be sent to the policy engine."""
    flow = FLOWS[fourtuple]
    with flow.ingress_lock:
        # Record the time when we check this flow.
        flow.latest_time_sec = time.time()

        if fourtuple not in flags:
            assert isinstance(MANAGER, multiprocessing.managers.SyncManager)
            flags[fourtuple] = MANAGER.Value(typecode="i", value=0)
        # Submit new packets to the policy engine only if this flow has not already
        # been submitted.
        if flags[fourtuple].value == 0:
            flags[fourtuple].value = 1

            if flow.loss_tracker is not None:
                # Calculate packets lost and loss event rate. Do this on all packets
                # because the loss event rate is based on current RTT, not minRTT, so
                # just the packets we send to the policy engine will not be enough. Note
                # that the loss event rate results are just for the last packet.
                (
                    packets_lost,
                    win_to_loss_event_rate,
                ) = flow.loss_tracker.loss_event_rate(flow.incoming_packets)
                logging.info("win_to_loss_event_rate: %s", win_to_loss_event_rate)

                # Discard all but the minimum number of packets required to calculate
                # the longest window's features, and the number of packets required
                # for the smoothing window.
                end_time_us = flow.incoming_packets[-args.smoothing_window][4]
                for idx in range(1, len(flow.incoming_packets) - args.smoothing_window):
                    if (
                        end_time_us - flow.incoming_packets[idx][4]
                        < flow.min_rtt_us * longest_window
                    ):
                        break
                flow.incoming_packets = flow.incoming_packets[idx - 1 :]
                packets_lost = packets_lost[idx - 1 :]
            else:
                packets_lost = 0
                win_to_loss_event_rate = {}

            logging.info(
                "Sending to policy engine the most recent %d packets for flow: %s",
                len(flow.incoming_packets),
                flow,
            )
            try:
                info = (
                    fourtuple,
                    flow.incoming_packets,
                    packets_lost,
                    flow.start_time_us,
                    flow.min_rtt_us,
                    win_to_loss_event_rate,
                )
                if args.policy == Policy.SERVICEPOLICY:
                    info = (
                        *info,
                        epoch,
                        # Sender IP.
                        flow.flowkey.remote_addr,
                        # Num flows to expect.
                        len(FLOWS.get_flows_from_sender(flow.flowkey.remote_addr)),
                    )
                que.put(
                    ("packets", *info),
                    block=False,
                )
            except queue.Full:
                logging.warning("Warning: RateMon policy engine queue is full!")

            flow.incoming_packets = []
        else:
            logging.info(
                (
                    "Not submitting to policy engine because "
                    "flow has already been submitted: %s"
                ),
                flow,
            )


def pcapy_sniff_main(args, done):
    # The current thread will sniff packets.

    # Look up my IP address to use when filtering packets.
    global MY_IP
    MY_IP = utils.ip_str_to_int(ni.ifaddresses(args.interface)[ni.AF_INET][0]["addr"])

    try:
        pcapy_sniff(args, done)
    except KeyboardInterrupt:
        logging.info("Cancelled.")
        done.set()


def pcapy_sniff(args, done):
    """Use pcapy to sniff packets from a specific interface."""
    # Set the snapshot length to the maximum size of the Ethernet, IPv4, and TCP
    # headers. Do not put the interface into promiscuous mode. Set the timeout to
    # 1000 ms, which actually allows batching up to 1000 ms
    # of packets from the kernel.
    pcap = pcapy.open_live(args.interface, 14 + 60 + 60, 0, 1000)

    def pcapy_cleanup():
        """Close the pcapy reader."""
        logging.info("Stopping pcapy sniffing...")
        pcap.close()

    atexit.register(pcapy_cleanup)

    # Drop non-IPv4/TCP packets.
    filt = "ip and tcp and not port 22"
    # Drop packets that we do not care about.
    if args.skip_localhost:
        filt += f" and not host {utils.int_to_ip_str(LOCALHOST)}"
    if args.listen_ports:
        port_filt_l = [
            f"(dst host {utils.int_to_ip_str(MY_IP)} and dst port {port}) or "
            f"(src host {utils.int_to_ip_str(MY_IP)} and src port {port})"
            for port in args.listen_ports
        ]
        filt += f" and ({' or '.join(port_filt_l)})"
    logging.info("Using tcpdump filter: %s", filt)
    pcap.setfilter(filt)

    last_time_s = time.time()
    last_exit_check_s = time.time()
    num_packets = 0
    num_bytes = 0
    last_num_packets = 0
    last_total_bytes = 0
    i = 0
    while True:  # not done.is_set():
        now_s = time.time()

        # Only check done once every 10000 packets or 1 second.
        if i % 10000 == 0 or now_s - last_exit_check_s > 1:
            if done.is_set():
                break
            last_exit_check_s = time.time()

        # Note that this is a blocking call. If we do not receive a packet, then this
        # will never return and we will never check the above exit conditions.
        new_packets, new_bytes = receive_packet_pcapy(*pcap.next())

        num_packets += new_packets
        num_bytes += new_bytes
        delta_time_s = now_s - last_time_s
        if delta_time_s >= 10:
            delta_num_packets = num_packets - last_num_packets
            delta_total_bytes = num_bytes - last_total_bytes
            logging.info(
                "Ingress performance --- %.2f pps, %.2f Mbps",
                delta_num_packets / delta_time_s,
                8 * delta_total_bytes / delta_time_s / 1e6,
            )
            last_time_s = now_s
            last_num_packets = num_packets
            last_total_bytes = num_bytes

        i += 1


def receive_packet_pcapy(header, packet):
    """Process a packet sniffed by pcapy.

    Extract relevant fields, compute the RTT (for incoming packets), and sort the
    packet into the appropriate flow.

    Inspired by:
    https://www.binarytides.com/code-a-packet-sniffer-in-python-with-pcapy-extension/
    """
    # Note: We do not need to check that this packet IPv4 and TCP because we already do
    #       that with a filter.
    if header is None:
        return 0, 0

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

    thl = (tcp[4] >> 4) * 4
    total_bytes = header.getlen()
    # (seconds, microseconds)
    time_tuple = header.getts()
    time_us = time_tuple[0] * 1e6 + time_tuple[1]

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
    with FLOWS.lock:
        if fourtuple in FLOWS:
            flow = FLOWS[fourtuple]
        else:
            flow = flow_utils.Flow(fourtuple, LOSS_EVENT_INTERVALS, time_us)
            FLOWS[fourtuple] = flow

    if tsecr is None or tsval is None:
        logging.warning("Could not determine tsval and tsecr for flow: %s", flow)

    with flow.ingress_lock:
        rtt_us = -1
        if incoming:
            if tsval is not None and tsecr is not None:
                # Use the TCP timestamp option to calculate the RTT.
                if tsecr in flow.sent_tsvals:
                    candidate_rtt_us = time_us - flow.sent_tsvals[tsecr]
                    if candidate_rtt_us > 0:
                        rtt_us = candidate_rtt_us
                        old_min_rtt_us = flow.min_rtt_us
                        flow.min_rtt_us = min(flow.min_rtt_us, rtt_us)
                        if old_min_rtt_us != flow.min_rtt_us:
                            logging.info(
                                "Updated min RTT for flow %s from %d us to %d us",
                                flow,
                                old_min_rtt_us,
                                flow.min_rtt_us,
                            )
                    # del flow.sent_tsvals[tsecr]

            flow.incoming_packets.append(
                (
                    tcp[2],  # seq
                    rtt_us,
                    total_bytes,
                    total_bytes - (ehl + ihl + thl),  # payload bytes
                    time_us,
                )
            )

            # Only give up credit for processing incoming packets.
            return 1, total_bytes
        # Track outgoing tsval for use later.
        flow.sent_tsvals[tsval] = time_us
    return 0, 0


if __name__ == "__main__":
    sys.exit(main())
