"""Monitors incoming TCP flows to detect unfairness."""

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

NUM_PACKETS = 0
flowkey_map = dict()


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


def receive_packet(flows, flows_lock, pkt, done):
    """Ingest a new packet, identify its flow, and store it.

    This function delegates its main tasks to receive_packet_helper() and instead just
    guards that function by bypassing it if the done event is set and by catching
    KeyboardInterrupt exceptions.

    flows_lock protects flows.
    """
    try:
        # if not done.is_set():
        receive_packet_helper(flows, flows_lock, pkt)
    except KeyboardInterrupt:
        done.set()


def receive_packet_helper(flows, flows_lock, pkt):
    """Ingest a new packet, identify its flow, and store it.

    flows_lock protects flows.
    """
    global NUM_PACKETS

    # Skip packets that are just empty structs.
    if not pkt.valid:
        return
    # Skip packets on the loopback interface.
    if LOCALHOST in (pkt.saddr, pkt.daddr):
        return

    # Attempt to acquire flows_lock. If unsuccessful, skip this packet.
    if flows_lock.acquire(blocking=False):
        try:
            fourtuple = (pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
            if fourtuple in flowkey_map:
                flowkey = flowkey_map[fourtuple]
            else:
                flowkey = flow_utils.FlowKey(pkt.saddr, pkt.daddr, pkt.sport, pkt.dport)
                flowkey_map[fourtuple] = flowkey
            if flowkey in flows:
                flow = flows[flowkey]
            else:
                flow = flow_utils.Flow(flowkey)
                flows[flowkey] = flow

            # Attempt to acquire the lock for this flow. If unsuccessful, then skip this
            # packet.
            if flow.lock.acquire(blocking=False):
                try:
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
                    flow.packets.append(dat)

                    # global NUM_PACKETS
                    NUM_PACKETS += 1
                finally:
                    flow.lock.release()
        finally:
            flows_lock.release()


def check_loop(flows, flows_lock, args, que, done):
    """Periodically evaluate flow fairness.

    Intended to be run as the target function of a thread.
    """
    try:
        while not done.is_set():
            last_check = time.time()

            check_flows(flows, flows_lock, args, que)

            # with flows_lock:
            #     # Do not bother acquiring the per-flow locks since we are just reading
            #     # data for logging purposes. It is okay if we get inconsistent
            #     # information.
            #     print(
            #         "Current decisions:\n"
            #         + "\n".join(f"\t{flow}: {flow.decision}" for flow in flows.values())
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


def check_flows(flows, flows_lock, args, que):
    """Identify flows that are ready to be checked, and check them.

    Remove old and empty flows.
    """
    to_remove = []
    to_check = []

    # Need to acquire flows_lock while iterating over flows.
    with flows_lock:
        for flowkey, flow in flows.items():
            # print(f"\t{flow} - {len(flow.packets)} packets")
            # Try to acquire the lock for this flow. If unsuccessful, do not block; move
            # on to the next flow.
            if flow.lock.acquire(blocking=False):
                try:
                    # TODO: Need some notion of "this is the minimum number or time
                    # interval of packets that I need to run the model".
                    if len(flow.packets) > 0 and (
                        args.min_packets is None or len(flow.packets) > args.min_packets
                    ):
                        # Plan to run inference on "full" flows.
                        to_check.append(flowkey)
                    elif flow.latest_time_sec and (
                        time.time() - flow.latest_time_sec > OLD_THRESH_SEC
                    ):
                        # Remove flows with no packets and flows that have not received
                        # a new packet in five seconds.
                        to_remove.append(flowkey)
                finally:
                    flow.lock.release()
            else:
                print(f"Cloud not acquire lock for flow {flow}")
        # Garbage collection.
        for flowkey in to_remove:
            del flows[flowkey]
            # if flow_key in flow_to_rwnd:
            #     del flow_to_rwnd[flow_key]

    for flowkey in to_check:
        flow = flows[flowkey]
        # Try to acquire the lock for this flow. If unsuccessful, do not block. Move on
        # to the next flow.
        if flow.lock.acquire(blocking=False):
            try:
                if not args.disable_inference:
                    check_flow(flows, flowkey, args, que)
            finally:
                flow.lock.release()
        else:
            print(f"Cloud not acquire lock for flow {flow}")


def check_flow(flows, flowkey, args, que):
    """Determine whether a flow is unfair and how to mitigate it.

    Runs inference on a flow's packets and determines the appropriate ACK
    pacing for the flow. Updates the flow's fairness record.
    """
    flow = flows[flowkey]
    with flow.lock:
        print(f"Running inference on {len(flow.packets)} packets for flow {flow}")
        # Discard all but the most recent few packets.
        if len(flow.packets) > args.min_packets:
            flow.packets = flow.packets[-args.min_packets :]
        # Record the time at which we check this flow.
        flow.latest_time_sec = time.time()
        try:
            que.put((flowkey, flow.packets), block=False)
        except queue.Full:
            pass
        else:
            # Clear the flow's packets.
            flow.packets = []


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

    # Maps each flow (four-tuple) to a list of packets for that flow. New
    # packets are appended to the ends of these lists. Periodically, a flow's
    # packets are consumed by the inference engine and that flow's list is
    # reset to empty.
    flows = dict()
    # Lock for the packet input data structures (e.g., "flows"). Only acquire this lock
    # when adding, removing, or iterating over flows; no need to acquire this lock when
    # updating a flow object.
    flows_lock = threading.RLock()
    # Flag to trigger threads/processes to terminate.
    done = multiprocessing.Event()
    done.clear()

    # que = multiprocessing.Queue()
    # Set up the inference thread.
    # check_thread = threading.Thread(
    #     target=check_loop,
    #     args=(flows, flows_lock, args, que, done),
    # )
    # check_thread.start()

    # inference_proc = multiprocessing.Process(
    #     target=inference.run, args=(args, que, done, flow_to_rwnd)
    # )
    # inference_proc.start()

    print("Running...press Control-C to end")
    last_time_s = time.time()
    last_epoch = -1
    with flows_lock:
        last_packets = NUM_PACKETS
    try:
        while not done.is_set():
            if args.poll_interval_ms is not None:
                time.sleep(args.poll_interval_ms / 1e3)
            last_epoch = poll_ringbuffer(
                ringbuffer, flows, flows_lock, done, last_epoch
            )
            last_time_s, last_packets = log_rate(flows_lock, last_time_s, last_packets)

    except KeyboardInterrupt:
        print("Cancelled.")
        done.set()
    finally:
        print("Cleaning up...")
        ipr.tc("del", "htb", ifindex, 0x10000, default=0x200000)

    # print("\nFlows:")
    # for flow, pkts in sorted(flows.items()):
    #     print("\t", flow_to_str(flow), len(pkts))

    # check_thread.join()
    # inference_proc.join()


def poll_ringbuffer(ringbuffer, flows, flows_lock, done, last_epoch):
    print("poll_ringbuffer")
    current_epoch = ringbuffer[0].epoch
    if current_epoch == last_epoch:
        time.sleep(0.1)
    else:
        new_packets = list(ringbuffer.items_lookup_batch())
        print(f"Found {len(new_packets)} new packets")

        # It is likely that an epoch transition occurred in the middle of the
        # array. Find this point. Process all packets after this transition
        # point (older packets) before processing packets before the transition
        # point (newer packets).
        epoch_transition_idx = None
        for idx, pkt in new_packets:
            print(f"{idx}")
            if pkt.epoch != current_epoch:
                # We found an older epoch. Process these packets first.
                if epoch_transition_idx is None:
                    epoch_transition_idx = idx
                receive_packet(flows, flows_lock, pkt, done)
        if epoch_transition_idx is not None:
            print(
                f"Epoch transition at idx {epoch_transition_idx}: "
                f"{current_epoch} -> {new_packets[epoch_transition_idx][1].epoch}"
            )
            for idx in range(epoch_transition_idx):
                receive_packet(flows, flows_lock, new_packets[idx][1], done)
        last_epoch = current_epoch
    return last_epoch


def log_rate(flows_lock, last_time_s, last_packets):
    print("log_rate")
    cur_time_s = time.time()
    delta_s = cur_time_s - last_time_s
    if delta_s > 10:
        with flows_lock:
            cur_packets = NUM_PACKETS
        pps = (cur_packets - last_packets) / delta_s
        print(
            f"Packets per second: {pps:.2f}, "
            "processed throughput at 1500 B MSS: "
            f"{pps * 1500 * 8 / 1e6:.2f} Mbps"
        )
        last_time_s = cur_time_s
        last_packets = cur_packets
    return last_time_s, last_packets


def _main():
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(_main())
