"""Monitors incoming TCP FLOWS to detect unfairness."""

from argparse import ArgumentParser
import atexit
import collections
import multiprocessing
import os
from os import path
import pcapy
import queue
import scapy
import socket
from struct import unpack
import sys
import threading
import time

from pyroute2 import IPRoute, protocols
from pyroute2.netlink.exceptions import NetlinkError

from bcc import BPF, BPFAttachType
import netifaces as ni

from unfair.model import defaults, models, utils
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
FLOWS = dict()
# Lock for the packet input data structures (e.g., "flows"). Only acquire this lock
# when adding, removing, or iterating over flows; no need to acquire this lock when
# updating a flow object.
FLOWS_LOCK = threading.RLock()

NUM_PACKETS = 0
TOTAL_BYTES = 0
MY_IP = None


def load_bpf(debug=False):
    """Load the corresponding eBPF program."""
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
    if debug:
        print(bpf_text)

    # Load BPF program.
    return BPF(text=bpf_text)


def receive_packets(packets, last_epoch):
    """Ingest new packets, identify their flows, and store them."""
    # Build a local mapping of flows to packets for this batch so that we do not need
    # to acquire a flow lock for every packet.
    #
    # It is likely that an epoch transition occurred in the middle of the
    # array. Find this point. Record packets after this transition
    # point (older packets) separately from packets before the transition
    # point (newer packets).
    older_flows_cache = collections.defaultdict(list)
    newer_flows_cache = collections.defaultdict(list)
    total_bytes = 0
    current_epoch = last_epoch

    try:
        _, first = next(packets)
        current_epoch = first.epoch
        if current_epoch == last_epoch:
            return 0, 0

        # First packet
        if first.total_bytes != 0:
            total_bytes += first.total_bytes
            newer_flows_cache[
                (first.daddr, first.saddr, first.dport, first.sport)
            ].append(
                (
                    first.seq,
                    first.srtt_us,
                    first.total_bytes,
                    first.payload_bytes,
                    first.time_us,
                )
            )

        # Newer packets
        while True:
            idx, pkt = next(packets)
            if pkt.total_bytes == 0:
                continue
            if pkt.epoch != current_epoch:
                break
            total_bytes += pkt.total_bytes
            newer_flows_cache[(pkt.daddr, pkt.saddr, pkt.dport, pkt.sport)].append(
                (
                    pkt.seq,
                    pkt.srtt_us,
                    pkt.total_bytes,
                    pkt.payload_bytes,
                    pkt.time_us,
                )
            )

        # Middle packet. If this code is reached, then pkt is defined.
        if pkt.epoch != current_epoch and pkt.total_bytes != 0:
            print(f"Found epoch transition at index {idx}")
            total_bytes += pkt.total_bytes
            older_flows_cache[(pkt.daddr, pkt.saddr, pkt.dport, pkt.sport)].append(
                (
                    pkt.seq,
                    pkt.srtt_us,
                    pkt.total_bytes,
                    pkt.payload_bytes,
                    pkt.time_us,
                )
            )

        # Older packets
        while True:
            _, pkt = next(packets)
            if pkt.total_bytes == 0:
                continue
            total_bytes += pkt.total_bytes
            older_flows_cache[(pkt.daddr, pkt.saddr, pkt.dport, pkt.sport)].append(
                (
                    pkt.seq,
                    pkt.srtt_us,
                    pkt.total_bytes,
                    pkt.payload_bytes,
                    pkt.time_us,
                )
            )
    except StopIteration:
        pass

    # Update the master flows database.
    num_packets = 0
    with FLOWS_LOCK:
        # Older flows first this time.
        for fourtuple, flow_packets in older_flows_cache.items():
            if fourtuple in FLOWS:
                flow = FLOWS[fourtuple]
            else:
                flow = flow_utils.Flow(fourtuple)
                FLOWS[fourtuple] = flow
            with flow.ingress_lock:
                num_packets += len(flow_packets)
                flow.packets.extend(flow_packets)

        # Then the newer flows.
        for fourtuple, flow_packets in newer_flows_cache.items():
            if fourtuple in FLOWS:
                flow = FLOWS[fourtuple]
            else:
                flow = flow_utils.Flow(fourtuple)
                FLOWS[fourtuple] = flow
            with flow.ingress_lock:
                num_packets += len(flow_packets)
                flow.packets.extend(flow_packets)

    return current_epoch, num_packets, total_bytes


def check_loop(flow_to_rwnd, args, que, done):
    """Periodically evaluate flow fairness.

    Intended to be run as the target function of a thread.
    """
    try:
        while not done.is_set():
            last_check = time.time()

            check_flows(flow_to_rwnd, args, que)

            # with FLOWS_LOCK:
            #     # Do not bother acquiring the per-flow locks since we are just reading
            #     # data for logging purposes. It is okay if we get inconsistent
            #     # information.
            #     print(
            #         "Current decisions:\n"
            #         + "\n".join(f"\t{flow}: {flow.decision}" for flow in FLOWS.values())
            #     )

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
        done.set()


def check_flows(flow_to_rwnd, args, que):
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
                        len(flow.packets) > 0
                        # If we have specified a minimum number of packets to run
                        # inference, then check that.
                        and (
                            args.min_packets is None
                            or len(flow.packets) > args.min_packets
                        )
                        # Skip flows on the loopback interface.
                        and (
                            LOCALHOST
                            not in (flow.flowkey.local_addr, flow.flowkey.remote_addr)
                        )
                        # Only consider flows on local ports 9998-10000.
                        and (
                            flow.flowkey.local_port >= 9998
                            and flow.flowkey.local_port <= 10000
                        )
                    ):
                        # Plan to run inference on this flows.
                        to_check.append(fourtuple)
                    elif flow.latest_time_sec and (
                        time.time() - flow.latest_time_sec > OLD_THRESH_SEC
                    ):
                        # Remove flows that have not been selected for inference in a
                        # while.
                        to_remove.append(fourtuple)
                finally:
                    flow.ingress_lock.release()
            else:
                print(f"Could not acquire lock for flow: {flow}")
        # Garbage collection.
        for fourtuple in to_remove:
            flowkey = FLOWS[fourtuple].flowkey
            if flowkey in flow_to_rwnd:
                del flow_to_rwnd[flowkey]
            del FLOWS[fourtuple]
            with inference.INFERENCE_FLAGS_LOCK:
                if fourtuple in inference.INFERENCE_FLAGS:
                    del inference.INFERENCE_FLAGS[fourtuple]

    for fourtuple in to_check:
        flow = FLOWS[fourtuple]
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on
        # to the next flow.
        if flow.ingress_lock.acquire(blocking=False):
            try:
                check_flow(fourtuple, args, que)
            finally:
                flow.ingress_lock.release()
        else:
            print(f"Could not acquire lock for flow: {flow}")


def check_flow(fourtuple, args, que):
    """Determine whether a flow is unfair and how to mitigate it.

    Runs inference on a flow's packets and determines the appropriate ACK
    pacing for the flow. Updates the flow's fairness record.
    """
    flow = FLOWS[fourtuple]
    with flow.ingress_lock and inference.INFERENCE_FLAGS_LOCK:
        # Discard all but the most recent few packets.
        if len(flow.packets) > args.min_packets:
            flow.packets = flow.packets[-args.min_packets :]
        # Record the time at which we check this flow.
        flow.latest_time_sec = time.time()

        # Only submit new packets for inference if inference is not already running.
        if inference.INFERENCE_FLAGS[fourtuple].value == 0:
            inference.INFERENCE_FLAGS[fourtuple].value = 1
            try:
                print(
                    f"Scheduling inference on {len(flow.packets)} "
                    f"packets for flow: {flow}"
                )
                if not args.disable_inference:
                    que.put((fourtuple, flow.packets), block=False)
            except queue.Full:
                pass
            else:
                # Clear the flow's packets.
                flow.packets = []
        else:
            print(f"Skipping inference for flow: {flow}")


def poll_ringbuffer(ringbuffer, last_epoch):
    """Poll a custom eBPF ringbuffer."""
    # Peek at the first element.
    current_epoch = ringbuffer[0].epoch
    num_packets = 0
    total_bytes = 0
    if current_epoch != last_epoch:
        last_epoch, num_packets, total_bytes = receive_packets(
            ringbuffer.items_lookup_batch(), last_epoch
        )

        print(f"Found {num_packets} new packets")
        # print(f"Average bytes per packet: {total_bytes / num_packets:.2f}")
    return last_epoch, num_packets, total_bytes


def log_rate(
    last_time_s,
    delta_num_packets,
    delta_total_bytes,
    target_delta_s=10,
):
    """Log the packet processing rate."""
    cur_time_s = time.time()
    delta_s = cur_time_s - last_time_s
    if delta_s > target_delta_s:
        print(
            f"Packets per second: {delta_num_packets / delta_s:.2f}, "
            f"throughput: {delta_total_bytes / delta_s / 1e6:.2f} Mbps"
        )
        last_time_s = cur_time_s
        delta_total_bytes = 0
        delta_num_packets = 0
    return last_time_s, delta_total_bytes, delta_num_packets


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


def run(args):
    """Core logic."""
    # Load BPF program.
    bpf = load_bpf(args.debug)
    bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
    egress_fn = bpf.load_func("handle_egress", BPF.SCHED_ACT)
    flow_to_rwnd = bpf["flow_to_rwnd"]
    ringbuffer = bpf["ringbuffer"]

    # Logic for reading the TCP window scale.
    func_sock_ops = bpf.load_func("read_win_scale", bpf.SOCK_OPS)  # sock_stuff
    filedesc = os.open(args.cgroup, os.O_RDONLY)
    bpf.attach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

    def detach_sockops():
        print("Detaching sock_ops hook...")
        bpf.detach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

    atexit.register(detach_sockops)

    # Configure unfairness mitigation strategy.
    ipr = IPRoute()
    ifindex = ipr.link_lookup(ifname=args.interface)
    assert (
        len(ifindex) == 1
    ), f"Trouble looking up index for interface {args.interface}: {ifindex}"
    ifindex = ifindex[0]
    # ipr.tc("add", "pfifo", 0, "1:")
    # ipr.tc("add-filter", "bpf", 0, ":1", fd=egress_fn.fd, name=egress_fn.name, parent="1:")

    # There can also be a chain of actions, which depend on the return
    # value of the previous action.
    action = dict(kind="bpf", fd=egress_fn.fd, name=egress_fn.name, action="ok")
    try:
        # Add the action to a u32 match-all filter
        ipr.tc("add", "htb", ifindex, 0x10000, default=0x200000)
        ipr.tc(
            "add-filter",
            "u32",
            ifindex,
            parent=0x10000,
            prio=10,
            protocol=protocols.ETH_P_ALL,  # Every packet
            target=0x10020,
            keys=["0x0/0x0+0"],
            action=action,
        )
    except NetlinkError:
        print("Error: Unable to configure TC.")
        return 1

    # Flag to trigger threads/processes to terminate.
    done = multiprocessing.Event()
    done.clear()

    que = multiprocessing.Queue()
    # Set up the inference thread.
    check_thread = threading.Thread(
        target=check_loop,
        args=(flow_to_rwnd, args, que, done),
    )
    check_thread.start()

    inference_proc = multiprocessing.Process(
        target=inference.run, args=(args, que, done, flow_to_rwnd)
    )
    inference_proc.start()

    # NUM_PACKETS = 0
    # TOTAL_BYTES = 0

    # This function will be called to process an event from the BPF program.
    def process_event(cpu, dat, size):
        # When this is called, it is actually in the same thread.
        pkt = bpf["pkts"].event(dat)

        # drop packets from localhost
        if pkt.saddr == 0x7F000001 or pkt.daddr == 0x7F000001:
            return
        # only accept packets from ports 9998, 9999, and 10000
        if pkt.dport not in (9998, 9999, 10000):
            return

        global NUM_PACKETS, TOTAL_BYTES
        NUM_PACKETS += 1
        TOTAL_BYTES += pkt.total_bytes

        # print(pkt.total_bytes)
        # print current process id and thread id
        # print(
        #     f"process_event {multiprocessing.current_process().pid} {threading.current_thread().ident}"
        # )
        return

    bpf["pkts"].open_perf_buffer(process_event)

    print("Running...press Control-C to end")
    last_time_s = time.time()
    last_epoch = -1
    # delta_num_packets = 0
    # delta_total_bytes = 0
    last_num_packets = NUM_PACKETS
    last_total_bytes = TOTAL_BYTES
    try:
        while not done.is_set():
            # if args.poll_interval_ms is not None:
            #     time.sleep(args.poll_interval_ms / 1e3)

            # print(
            #    f"main {multiprocessing.current_process().pid} {threading.current_thread().ident}"
            # )
            bpf.perf_buffer_poll()

            # print change in NUM_PACKETS and TOTAL_BYTES every 10 seconds
            now_s = time.time()
            delta_time_s = now_s - last_time_s
            if delta_time_s >= 10:
                delta_num_packets = NUM_PACKETS - last_num_packets
                delta_total_bytes = TOTAL_BYTES - last_total_bytes
                last_time_s = now_s
                last_num_packets = NUM_PACKETS
                last_total_bytes = TOTAL_BYTES
                print(
                    f"{NUM_PACKETS} packets, {TOTAL_BYTES} bytes, {delta_num_packets / delta_time_s:.2f} pps, {8 * delta_total_bytes / delta_time_s / 1e6:.2f} Mbps"
                )

            # last_epoch, num_packets, total_bytes = poll_ringbuffer(
            #     ringbuffer, last_epoch
            # )

            # delta_num_packets += num_packets
            # delta_total_bytes += total_bytes
            # last_time_s, delta_total_bytes, delta_num_packets = log_rate(
            #     last_time_s, delta_num_packets, delta_total_bytes
            # )
            # if num_packets == 0:
            #     # If we did not find any new packets, then sleep for a while.
            #     time.sleep(0.1)
    except KeyboardInterrupt:
        print("Cancelled.")
        done.set()
    finally:
        print("Cleaning up...")
        ipr.tc("del", "htb", ifindex, 0x10000, default=0x200000)

    # print("\nFlows:")
    # for flow, pkts in sorted(FLOWS.items()):
    #     print("\t", flow_to_str(flow), len(pkts))

    check_thread.join()
    inference_proc.join()


def _main():
    return run(parse_args())


# Inspired by: https://www.binarytides.com/code-a-packet-sniffer-in-python-with-pcapy-extension/
def receive_packet_pcapy(header, packet):
    if header is None:
        return

    # The Ethernet header is 14 bytes.
    ehl = 14
    eth = unpack("!6s6sH", packet[:ehl])
    # Skip packet if network protocol is not IP.
    if socket.ntohs(eth[2]) != 8:
        return
    ip = unpack("!BBHHHBBH4s4s", packet[ehl : 20 + ehl])
    version_ihl = ip[0]
    # Skip packer if IP version is not 4 or protocol is not TCP.
    if (version_ihl >> 4) != 4 or ip[6] != 6:
        return

    ihl = (version_ihl & 0xF) * 4
    tcp_offset = ehl + ihl
    tcp = unpack("!HHLLBBHHH", packet[tcp_offset : tcp_offset + 20])
    saddr = int.from_bytes(ip[8], byteorder="little", signed=False)
    daddr = int.from_bytes(ip[9], byteorder="little", signed=False)
    incoming = daddr == MY_IP
    sport = tcp[0]
    dport = tcp[1]

    # Only accept packets on local ports 9998, 9999, and 10000.
    if (incoming and (dport < 9998 or dport > 10000)) or (
        not incoming and (sport < 9998 or sport > 10000)
    ):
        return

    thl = (tcp[4] >> 4) * 4
    total_bytes = header.getlen()
    time_s, time_us = header.getts()
    time_s = time_s + time_us / 1e6  # arrival time in microseconds

    # Parse TCP timestamp option.
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
        if tsval is not None and tsecr is not None:
            if incoming:
                # Use the TCP timestamp option to calculate the RTT.
                if tsecr in flow.sent_tsvals:
                    rtt_s = flow.sent_tsvals[tsecr] - time_s
                    del flow.sent_tsvals[tsecr]
            else:
                # Track outgoing tsval for use later.
                flow.sent_tsvals[tsval] = time_s

        (flow.incoming_packets if incoming else flow.outgoing_packets).append(
            (
                tcp[2],  # seq
                rtt_s,
                total_bytes,
                total_bytes - (ehl + ihl + thl),  # payload bytes
                time_s,
            )
        )

    global NUM_PACKETS, TOTAL_BYTES
    NUM_PACKETS += 1
    TOTAL_BYTES += total_bytes


def sniff_pcapy(interface):
    # sniff packets using pcapy
    pcap = pcapy.open_live(interface, 100, 0, 1000)
    pcap.setfilter("tcp")

    # get my ip address for interface
    my_ip = ni.ifaddresses(interface)[ni.AF_INET][0]["addr"]
    global MY_IP
    MY_IP = utils.ip_str_to_int(my_ip)
    print(MY_IP)

    last_time_s = time.time()
    last_num_packets = NUM_PACKETS
    last_total_bytes = TOTAL_BYTES

    try:
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
                    f"{NUM_PACKETS} packets, {TOTAL_BYTES} bytes, {delta_num_packets / delta_time_s:.2f} pps, {8 * delta_total_bytes / delta_time_s / 1e6:.2f} Mbps"
                )
                with FLOWS_LOCK:
                    print("Flows:")
                    for flow in FLOWS.keys():
                        print(utils.flow_to_str(flow))

    except KeyboardInterrupt:
        pcap.close()


if __name__ == "__main__":
    # sys.exit(_main())
    sniff_pcapy("ens3")
