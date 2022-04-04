"""Monitors incoming TCP FLOWS to detect unfairness."""

from argparse import ArgumentParser
import atexit
import multiprocessing
import os
from os import path
import queue
import sys
import threading
import time

from pyroute2 import IPRoute, protocols
from pyroute2.netlink.exceptions import NetlinkError

from bcc import BPF, BPFAttachType

from unfair.model import models, utils
from unfair.runtime import flow_utils, inference, mitigation_strategy, reaction_strategy
from unfair.runtime.mitigation_strategy import MitigationStrategy
from unfair.runtime.reaction_strategy import ReactionStrategy


LOCALHOST = utils.ip_str_to_int("127.0.0.1")
# Flows that have not received a new packet in this many seconds will be
# garbage collected.
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


def receive_packets(packets):
    """Ingest a new packet, identify its flow, and store it."""
    # Attempt to acquire FLOWS_LOCK. If unsuccessful, skip this packet.
    total_bytes = 0

    if FLOWS_LOCK.acquire(blocking=False):
        try:
            for idx in range(len(packets)):
                pkt = packets[idx][1]

                # Skip packets that are just empty structs. The total_bytes field is
                # guaranteed never to be zero for valid packets.
                if pkt.total_bytes == 0:
                    continue
                total_bytes += pkt.total_bytes

                # The order needs to be (local addr, remote addr, local port, remote port)
                # to make creating a flow_utils.FlowKey simpler.
                fourtuple = (pkt.daddr, pkt.saddr, pkt.dport, pkt.sport)
                if fourtuple in FLOWS:
                    flow = FLOWS[fourtuple]
                else:
                    flow = flow_utils.Flow(fourtuple)
                    FLOWS[fourtuple] = flow

                # Attempt to acquire the lock for this flow. If unsuccessful, then skip this
                # packet.
                # if flow.ingress_lock.acquire(blocking=False):
                #     try:
                flow.packets.append((
                    pkt.seq,
                    pkt.srtt_us,
                    # pkt.tsval,
                    # pkt.tsecr,
                    pkt.total_bytes,
                    # pkt.ihl_bytes,
                    # pkt.thl_bytes,
                    pkt.payload_bytes,
                    pkt.time_us,
                ))
                    # finally:
                    #     flow.ingress_lock.release()
        finally:
            FLOWS_LOCK.release()
    return total_bytes


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
            # print(f"\t{flow} - {len(flow.packets)} packets")
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
                        # Plan to run inference on "full" flows.
                        to_check.append(fourtuple)
                    elif flow.latest_time_sec and (
                        time.time() - flow.latest_time_sec > OLD_THRESH_SEC
                    ):
                        # Remove flows with no packets and flows that have not received
                        # a new packet in five seconds.
                        to_remove.append(fourtuple)
                finally:
                    flow.ingress_lock.release()
            else:
                print(f"Could not acquire lock for flow {flow}")
        # Garbage collection.
        for fourtuple in to_remove:
            flowkey = FLOWS[fourtuple].flowkey
            if flowkey in flow_to_rwnd:
                del flow_to_rwnd[flowkey]
            del FLOWS[fourtuple]

    for fourtuple in to_check:
        flow = FLOWS[fourtuple]
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on
        # to the next flow.
        if flow.ingress_lock.acquire(blocking=False):
            try:
                if not args.disable_inference:
                    check_flow(fourtuple, args, que)
            finally:
                flow.ingress_lock.release()
        else:
            print(f"Cloud not acquire lock for flow {flow}")


def check_flow(fourtuple, args, que):
    """Determine whether a flow is unfair and how to mitigate it.

    Runs inference on a flow's packets and determines the appropriate ACK
    pacing for the flow. Updates the flow's fairness record.
    """
    flow = FLOWS[fourtuple]
    with flow.ingress_lock and inference.inference_flags_lock:
        print(f"Running inference on {len(flow.packets)} packets for flow {flow}")
        # Discard all but the most recent few packets.
        if len(flow.packets) > args.min_packets:
            flow.packets = flow.packets[-args.min_packets :]
        # Record the time at which we check this flow.
        flow.latest_time_sec = time.time()

        # with inference.inference_flags_lock:
        # Only submit new packets for inference if inference is not already running.
        if inference.inference_flags[fourtuple].value == 0:
            inference.inference_flags[fourtuple].value = 1
            try:
                que.put((fourtuple, flow.packets), block=False)
            except queue.Full:
                pass
            else:
                # Clear the flow's packets.
                flow.packets = []
        else:
            print("Skipping inference")


def poll_ringbuffer(ringbuffer, last_epoch):
    """Poll a custom eBPF ringbuffer."""
    current_epoch = ringbuffer[0].epoch
    found_packets = current_epoch != last_epoch
    total_bytes = 0
    num_packets = 0
    if found_packets:
        new_packets = list(ringbuffer.items_lookup_batch())
        num_packets = len(new_packets)
        print(f"Found {num_packets} new packets")

        # It is likely that an epoch transition occurred in the middle of the
        # array. Find this point. Process all packets after this transition
        # point (older packets) before processing packets before the transition
        # point (newer packets).
        epoch_transition_idx = 0
        for idx, pkt in new_packets:
            if pkt.epoch != current_epoch:
                epoch_transition_idx = idx
                break
                # We found an older epoch. Process these packets first.
                # if epoch_transition_idx is None:
                # receive_packet(pkt)

        total_bytes = receive_packets(new_packets[epoch_transition_idx:])

        if epoch_transition_idx > 0:
            print(
                f"Epoch transition at idx {epoch_transition_idx}: "
                f"{current_epoch} -> {new_packets[epoch_transition_idx][1].epoch}"
            )
            total_bytes += receive_packets(new_packets[:epoch_transition_idx])

            # for idx in range(epoch_transition_idx):
            #     receive_packet(new_packets[idx][1])
        last_epoch = current_epoch
        # print(f"Average bytes per packet: {total_bytes / len(new_packets):.2f}")
    return found_packets, last_epoch, total_bytes, num_packets


def log_rate(
    last_time_s,
    delta_total_bytes,
    delta_num_packets,
    target_delta_s=10,
):
    """Log the packet processing rate."""
    cur_time_s = time.time()
    delta_s = cur_time_s - last_time_s
    if delta_s > target_delta_s:
        # with FLOWS_LOCK:
        #     cur_packets = NUM_PACKETS
        # pps =
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

    print("Running...press Control-C to end")
    last_time_s = time.time()
    last_epoch = -1
    delta_total_bytes = 0
    delta_num_packets = 0
    try:
        while not done.is_set():
            if args.poll_interval_ms is not None:
                time.sleep(args.poll_interval_ms / 1e3)
            found_packets, last_epoch, total_bytes, num_packets = poll_ringbuffer(
                ringbuffer, last_epoch
            )

            delta_total_bytes += total_bytes
            delta_num_packets += num_packets
            last_time_s, delta_total_bytes, delta_num_packets = log_rate(
                last_time_s, delta_total_bytes, delta_num_packets
            )
            if not found_packets:
                # If we did not find any new packets, then sleep for a while.
                time.sleep(0.1)
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


if __name__ == "__main__":
    sys.exit(_main())
