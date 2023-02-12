"""This module defines a process that will receive packets and run inference on them."""

import collections
import ctypes
import logging
import os
from os import path
import queue
import signal
import time
import traceback

from bcc import BPF, BPFAttachType
import numpy as np
from pyroute2 import IPRoute, protocols
from pyroute2.netlink.exceptions import NetlinkError

from unfair.model import data, defaults, features, gen_features, models, utils
from unfair.runtime import flow_utils, reaction_strategy
from unfair.runtime.reaction_strategy import ReactionStrategy


def predict(net, in_fets, debug=False):
    """Run inference on a flow's packets.

    Returns a label (below fair, approximately fair, above fair), the updated
    min_rtt_us, and the features of the last packet.
    """
    in_fets = utils.clean(in_fets)
    if debug:
        logging.debug(
            "Model input: %s\n%s",
            net.in_spc,
            "\n".join(", ".join(f"{fet}" for fet in row) for row in in_fets),
        )

    pred_start_s = time.time()
    preds = net.predict(in_fets)
    logging.info("Prediction time: %.2f ms", (time.time() - pred_start_s) * 1e3)

    return [defaults.Class(pred) for pred in preds]


def populate_features(
    net, flowkey, start_time_us, min_rtt_us, fets, prev_fets, smoothing_window
):
    """
    Populate the features for a flow.

    Only packets in the smoothing window receive full features and are returned. The
    smoothing window is always the last `smoothing_window` packets.
    """
    assert len(fets) >= smoothing_window, (
        f"Number of packets ({len(fets)}) must be at least as large "
        f"as the smoothing window ({smoothing_window})."
    )
    gen_features.parse_received_packets(
        flowkey,
        start_time_us,
        min_rtt_us,
        fets,
        prev_fets,
        win_metrics_start_idx=len(fets) - smoothing_window,
    )
    # Only keep enough packets to fill the smoothing window.
    in_fets = fets[-smoothing_window:]
    # Replace -1's and with NaNs and convert to an unstructured numpy array.
    data.replace_unknowns(
        in_fets,
        isinstance(net, models.HistGbdtSklearnWrapper),
        assert_no_unknowns=False,
    )
    data.replace_infinite(in_fets)
    return in_fets


def smooth(labels):
    """Combine multiple labels into a single label.

    Select the mode. Break ties by selecting the least fair label.
    """
    assert len(labels) > 0, "Labels cannot be empty."

    # Use https://www.statology.org/numpy-mode/ because we need to support there being
    # multiple modes, otherwise we would use scipy.stats.mode.
    #
    # Find unique values in array along with their counts.
    vals, counts = np.unique(labels, return_counts=True)
    # Find indices of modes.
    mode_value = np.argwhere(counts == np.max(counts))
    # Get actual mode values.
    modes = vals[mode_value].flatten().tolist()
    # The labels are numerically sorted from most fair (lowest label) to least fair
    # (highest label), so selecting the max of the modes will break ties by selecting
    # the less fair label.
    label = max(modes)
    if len(modes) > 1:
        logging.warning("Multiple modes encountered during smoothing: %s", modes)

    return label


def make_decision_sender_fairness(
    args, flowkeys, min_rtt_us, fets, label, flow_to_decisions
):
    """Make a fairness decision for all flows from a sender.

    Take the Mathis fair throughput and divide it equally between the flows.
    """
    logging.info(
        "Label for flows [%s]: %s",
        ", ".join(str(flowkey) for flowkey in flowkeys),
        label,
    )

    mathis_tput_bps_ler = fets[-1][
        features.make_win_metric(
            features.MATHIS_TPUT_LOSS_EVENT_RATE_FET, models.MathisFairness.win_size
        )
    ]
    # Divied the Mathis fair throughput equally between the flows.
    per_flow_tput_bps = mathis_tput_bps_ler / len(flowkeys)

    # Measure the recent loss rate.
    loss_rate = fets[-1][
        features.make_win_metric(features.LOSS_RATE_FET, models.MathisFairness.win_size)
    ]

    if label == defaults.Class.ABOVE_FAIR and loss_rate >= 1e-9:
        logging.info("Mode 1")
        # This sender is sending too fast according to the Mathis model, and we
        # know that the bottleneck is fully utilized because there has been loss
        # recently. Force all flows to slow down to the Mathis model fair rate.
        new_decision = (
            defaults.Decision.PACED,
            per_flow_tput_bps,
            utils.bdp_B(per_flow_tput_bps, min_rtt_us / 1e6),
        )
    elif np.array(
        [
            flow_to_decisions[flowkey][0] == defaults.Decision.PACED
            for flowkey in flowkeys
        ]
    ).any():
        logging.info("Mode 2")
        # The current measurement is that the sender is not unfair, but at
        # least one of its flows is already being paced. If the bottlenck is
        # not fully utilized, then allow the flows to speed up.
        #
        # We use the loss rate to determine whether the bottleneck is fully
        # utilized. If the loss rate is 0, then the bottleneck is not fully
        # utilized. If there is loss, then the bottleneck is fully utilized.

        # Look up the average enforced throughput of the flows that are already
        # being paced.
        avg_enforced_tput_bps = np.average(
            [
                flow_to_decisions[flowkey][1]
                for flowkey in flowkeys
                if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
            ]
        )
        logging.info(
            "Average enforced throughput: %.2f Mbps", avg_enforced_tput_bps / 1e6
        )

        if loss_rate < 1e-9:
            logging.info("Mode 2.1")
            new_tput_bps = reaction_strategy.react_up(
                args.reaction_strategy,
                # Base the new throughput on the current mathis model fair
                # rate. This will prevent the flows from growing quickly.
                # But it will allow a bit of probing that will help drive
                # the loss rate down faster and lead to quicker growth in
                # the Mathis fair rate.
                per_flow_tput_bps,
                # Base the new throughput on the previous enforced throughput.
                # avg_enforced_tput_bps,
                # # Base the new throughput on the observed average per-flow
                # # throughput.
                # # Note: this did not work because a flow that was spuriously aggressive can steal a lot of bandwidth from other flows.
                # fets[-1][
                #     features.make_win_metric(
                #         features.TPUT_FET, models.MathisFairness.win_size
                #     )
                # ]
                # / len(flowkeys),
                # ^^^ Bw probing: We give it move tput. If it actually achieves higher tput, then we give it more.
                # But if it doesn't achieve higher tput, then we don't end up growing forever.
                # With limitless scaling based on the enforce throughput, the throughput will eventually
                # cause losses and lead to a huge drop, basically like timeout+slow start.
            )
            new_decision = (
                defaults.Decision.PACED,
                new_tput_bps,
                utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
            )
        else:
            logging.info("Mode 2.2")
            # We know that the bottleneck is fully utilized because the sender
            # experienced loss recently. Preserve the existing per-flow decisions.
            new_decision = None

    else:
        logging.info("Mode 3")
        # The sender is not unfair or it is unfair but the link is not fully
        # utilized, and none of its flows behaved badly in the past, so leave
        # it alone.
        new_decision = (defaults.Decision.NOT_PACED, None, None)
    return new_decision


def make_decision_flow_fairness(
    args, flowkey, min_rtt_us, fets, label, flow_to_decisions
):
    """Make a fairness decision for a single flow.

    FIXME: Why are the BDP calculations coming out so small? Is the throughput
           just low due to low application demand?
    """
    logging.info("Label for flow %s: %s", flowkey, label)
    if args.reaction_strategy == ReactionStrategy.FILE:
        new_decision = (
            defaults.Decision.PACED,
            None,
            reaction_strategy.get_scheduled_pacing(args.schedule),
        )
    else:
        tput_bps = utils.safe_tput_bps(fets, 0, len(fets) - 1)
        if label == defaults.Class.ABOVE_FAIR:
            # This flow is sending too fast. Force the sender to slow down.
            new_tput_bps = reaction_strategy.react_down(
                args.reaction_strategy,
                # If the flow was already paced, then based the new paced throughput on
                # the old paced throughput.
                flow_to_decisions[flowkey][1]
                if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                else tput_bps,
            )
            new_decision = (
                defaults.Decision.PACED,
                new_tput_bps,
                utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
            )
        elif flow_to_decisions[flowkey][0] == defaults.Decision.PACED:
            # We are already pacing this flow.
            if label == defaults.Class.BELOW_FAIR:
                # If we are already pacing this flow but we are being too
                # aggressive, then let it send faster.
                new_tput_bps = reaction_strategy.react_up(
                    args.reaction_strategy,
                    # If the flow was already paced, then base the new paced
                    # throughput on the old paced throughput.
                    flow_to_decisions[flowkey][1]
                    if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                    else tput_bps,
                )
                new_decision = (
                    defaults.Decision.PACED,
                    new_tput_bps,
                    utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
                )
            else:
                # If we are already pacing this flow and it is behaving as desired,
                # then all is well. Retain the existing pacing decision.
                new_decision = flow_to_decisions[flowkey]
        else:
            # This flow is not already being paced and is not behaving unfairly, so
            # leave it alone.
            new_decision = (defaults.Decision.NOT_PACED, None, None)
    return new_decision


def apply_decision(flowkey, new_decision, flow_to_decisions, flow_to_rwnd):
    """Apply a decision to a flow."""
    logging.info(
        "Decision for flow %s: (%s, target tput: %s, rwnd: %s)",
        flowkey,
        new_decision[0],
        "-" if new_decision[1] is None else f"{new_decision[1] / 1e6:.2f} Mbps",
        "-" if new_decision[2] is None else f"{new_decision[2] / 1e3:.2f} KB",
    )
    if flow_to_decisions[flowkey] != new_decision:
        logging.info("Flow %s changed decision.", flowkey)
        if new_decision[2] is None:
            if flowkey in flow_to_rwnd:
                del flow_to_rwnd[flowkey]
        else:
            new_decision = (new_decision[0], new_decision[1], round(new_decision[2]))
            if new_decision[2] < defaults.MIN_RWND_B:
                logging.info(
                    ("Warning: Flow %s asking for RWND < %d: %d. " "Overriding to %d."),
                    flowkey,
                    defaults.MIN_RWND_B,
                    new_decision[2],
                    defaults.MIN_RWND_B,
                )
                new_decision = (new_decision[0], new_decision[1], defaults.MIN_RWND_B)

            assert new_decision[2] >= 0, (
                "Error: RWND must be non-negative, "
                f"but is {new_decision[2]} for flow {flowkey}."
            )
            flow_to_rwnd[flowkey] = ctypes.c_uint32(new_decision[2])
        flow_to_decisions[flowkey] = new_decision


def make_decision(
    args, flowkeys, min_rtt_us, fets, label, flow_to_decisions, flow_to_rwnd
):
    """Make a flow unfairness mitigation decision.

    Base the decision on the flow's label and existing decision. Use the flow's features
    to calculate any necessary flow metrics, such as the throughput.
    """
    if args.sender_fairness:
        new_decision = make_decision_sender_fairness(
            args, flowkeys, min_rtt_us, fets, label, flow_to_decisions
        )
    else:
        assert len(flowkeys) == 1
        new_decision = make_decision_flow_fairness(
            args, flowkeys[0], min_rtt_us, fets, label, flow_to_decisions
        )

    if new_decision is not None:
        for flowkey in flowkeys:
            apply_decision(flowkey, new_decision, flow_to_decisions, flow_to_rwnd)


def packets_to_ndarray(pkts, dtype, packets_lost, win_to_loss_event_rate):
    """Reorganize a list of packet metrics into a structured numpy array."""
    # Assume that the packets are in order.
    # # For some reason, the packets tend to get reordered after they are timestamped on
    # # arrival. Sort packets by timestamp.
    # pkts = sorted(pkts, key=lambda pkt: pkt[-1])
    (
        seqs,
        rtts_us,
        # tsvals,
        # tsecrs,
        totals_bytes,
        # _,
        # _,
        payloads_bytes,
        times_us,
    ) = zip(*pkts)
    # The final features. -1 implies that a value could not be calculated. Extend the
    # provided dtype with the regular features, which may be required to compute the
    # EWMA and windowed features.
    fets = np.full(len(seqs), -1, dtype=dtype)
    fets[features.SEQ_FET] = seqs
    fets[features.ARRIVAL_TIME_FET] = times_us
    fets[features.PAYLOAD_FET] = payloads_bytes
    fets[features.WIRELEN_FET] = totals_bytes
    fets[features.RTT_FET] = rtts_us
    fets[features.PACKETS_LOST_FET] = packets_lost

    # Fill in the loss event rate.
    for metric in fets.dtype.names:
        if metric.startswith(features.LOSS_EVENT_RATE_FET) and "windowed" in metric:
            fets[metric][-1] = win_to_loss_event_rate[
                features.parse_win_metric(metric)[1]
            ]
    return fets


def load_bpf():
    """Load the corresponding eBPF program."""
    # Load BPF text.
    bpf_flp = path.join(
        path.abspath(path.dirname(__file__)),
        "unfair_runtime.c",
    )
    if not path.isfile(bpf_flp):
        logging.error("Could not find BPF program: %s", bpf_flp)
        return 1
    logging.info("Loading BPF program: %s", bpf_flp)
    with open(bpf_flp, "r", encoding="utf-8") as fil:
        bpf_text = fil.read()
    # Load BPF program.
    return BPF(text=bpf_text)


def configure_ebpf(args):
    """Set up eBPF hooks."""
    if min(args.listen_ports) >= 50000:
        # Use the listen ports to determine the wait time, so that multiple
        # instances of this program do not try to configure themselves at the same
        # time.
        rand_sleep = min(args.listen_ports) - 50000
        logging.info("Waiting %f seconds to prevent race conditions...", rand_sleep)
        time.sleep(rand_sleep)

    try:
        bpf = load_bpf()
    except:
        logging.exception("Error loading BPF program!")
        return None, None
    flow_to_rwnd = bpf["flow_to_rwnd"]

    # Set up a TC egress qdisc, specify a filter the accepts all packets, and attach
    # our egress function as the action on that filter.
    ipr = IPRoute()
    ifindex = ipr.link_lookup(ifname=args.interface)
    assert (
        len(ifindex) == 1
    ), f'Trouble looking up index for interface "{args.interface}": {ifindex}'
    ifindex = ifindex[0]

    logging.info("Attempting to create central qdisc")
    handle = 0x10000
    default = 0x200000
    responsible_for_central_tc = False
    try:
        ipr.tc("add", "htb", ifindex, handle, default=default)
    except NetlinkError:
        logging.warning("Unable to create central qdisc. It probably already exists.")
    else:
        logging.info("Responsible for central TC")
        responsible_for_central_tc = True

    if not responsible_for_central_tc:
        # If someone else is responsible for the egress action, then we will just let
        # them do the work.
        logging.warning("Not configuring TC")
        return flow_to_rwnd, None

    # Read the TCP window scale on outgoing SYN-ACK packets.
    func_sock_ops = bpf.load_func("read_win_scale", bpf.SOCK_OPS)  # sock_stuff
    filedesc = os.open(args.cgroup, os.O_RDONLY)
    bpf.attach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

    # Overwrite advertised window size in outgoing packets.
    egress_fn = bpf.load_func("handle_egress", BPF.SCHED_ACT)
    action = dict(kind="bpf", fd=egress_fn.fd, name=egress_fn.name, action="ok")

    try:
        # Add the action to a u32 match-all filter
        ipr.tc(
            "add-filter",
            "u32",
            ifindex,
            parent=handle,
            prio=10,
            protocol=protocols.ETH_P_ALL,  # Every packet
            target=0x10020,
            keys=["0x0/0x0+0"],
            action=action,
        )
    except:
        logging.exception("Error: Unable to configure TC.")
        return None, None

    def ebpf_cleanup():
        """Clean attached eBPF programs."""
        logging.info("Detaching sock_ops hook...")
        bpf.detach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)
        logging.info("Removing egress TC...")
        ipr.tc("del", "htb", ifindex, handle, default=default)

    logging.info("Configured TC and BPF!")
    return flow_to_rwnd, ebpf_cleanup


def parse_from_inference_queue(
    val, flow_to_rwnd, flow_to_decisions, flow_to_prev_features
):
    """Parse a message from the inference queue."""
    sender_fairness = False
    epoch = num_flows_expected = None
    opcode, fourtuple = val[:2]
    flowkey = flow_utils.FlowKey(*fourtuple)

    if opcode.startswith("inference"):
        (
            pkts,
            packets_lost,
            start_time_us,
            min_rtt_us,
            win_to_loss_event_rate,
        ) = val[2:]

        if opcode.startswith("inference-sender-fairness"):
            sender_fairness = True
            epoch, num_flows_expected = opcode.split("-")[-2:]
            epoch = int(epoch)
            num_flows_expected = int(num_flows_expected)
    elif opcode == "remove":
        logging.info("Inference process: Removing flow %s", flowkey)
        if flowkey in flow_to_rwnd:
            del flow_to_rwnd[flowkey]
        if flowkey in flow_to_decisions:
            del flow_to_decisions[flowkey]
        if flowkey in flow_to_prev_features:
            del flow_to_prev_features[flowkey]
        return None
    else:
        raise RuntimeError(f'Unknown opcode "{opcode}" for flow: {flowkey}')

    return (
        pkts,
        packets_lost,
        start_time_us,
        min_rtt_us,
        win_to_loss_event_rate,
        fourtuple,
        flowkey,
        sender_fairness,
        epoch,
        num_flows_expected,
    )


def build_features(
    args,
    net,
    dtype,
    pkts,
    packets_lost,
    start_time_us,
    min_rtt_us,
    win_to_loss_event_rate,
    inference_flags,
    flow_to_prev_features,
    fourtuple,
    flowkey,
):
    """Build features for a flow."""
    logging.info("Building features for flow: %s", flowkey)
    try:
        # Prepare the numpy array in which we will store the features.
        all_fets = packets_to_ndarray(pkts, dtype, packets_lost, win_to_loss_event_rate)
        # Populate the above numpy array with features and return a pruned
        # version containing only the model input features the packets in
        # the smoothing window.
        in_fets = populate_features(
            net,
            flowkey,
            start_time_us,
            min_rtt_us,
            all_fets,
            flow_to_prev_features.get(flowkey),
            args.smoothing_window,
        )
        # We do not need to keep all of the original packets.
        all_fets = all_fets[-args.smoothing_window :]
        # Update previous fets for this flow.
        flow_to_prev_features[flowkey] = in_fets[-1]
        return all_fets, in_fets
    except AssertionError:
        # Assertion errors mean this batch of packets violated some
        # precondition, but we are safe to skip them and continue.
        logging.warning(
            (
                "Skipping flow %s because feature generation failed "
                "due to a non-fatal assertion failure:\n%s"
            ),
            flowkey,
            traceback.format_exc(),
        )
        logging.info("Clearing inference flag due to error for flow: %s", flowkey)
        inference_flags[fourtuple].value = 0
        return None
    except Exception as exp:
        # An unexpected error occurred. It is not safe to continue. Reraise the
        # exception to kill the process.
        logging.error(
            "Feature generation failed due to an unexpected error:\n%s",
            traceback.format_exc(),
        )
        raise exp


def wait_or_batch(
    net,
    sender_fairness,
    batch,
    waiting_room,
    fourtuple,
    flowkey,
    min_rtt_us,
    all_fets,
    in_fets,
    pkts,
    epoch,
    num_flows_expected,
):
    """Decide whether a flow should wait (if doing sender fairness) or be batched."""
    if not sender_fairness:
        batch.append(([fourtuple], [flowkey], min_rtt_us, all_fets, in_fets))
        logging.info(
            "Adding %d packets from flow %s to batch.",
            len(in_fets),
            flowkey,
        )
        # Additional packets_covered_by_batch
        return len(pkts)

    if (
        flowkey.remote_addr not in waiting_room
        or waiting_room[flowkey.remote_addr][0] < epoch
    ):
        waiting_room[flowkey.remote_addr] = (
            int(epoch),
            int(num_flows_expected),
            [],
        )
    logging.info(
        "Adding %d packets from flow %s to sender fairness waiting room.",
        len(in_fets),
        flowkey,
    )
    waiting_room[flowkey.remote_addr][2].append(
        (fourtuple, flowkey, min_rtt_us, all_fets, in_fets, pkts)
    )

    if len(waiting_room[flowkey.remote_addr][2]) < waiting_room[flowkey.remote_addr][1]:
        # Waiting room is not full yet, so there are no new packets for the batch.
        return 0

    # Empty the waiting room.
    all_flows_from_sender = waiting_room[flowkey.remote_addr][2]
    del waiting_room[flowkey.remote_addr]
    (
        merged_fourtuples,
        merged_flowkeys,
        merged_min_rtt_us,
        merged_all_fets,
        merged_in_fets,
    ) = merge_sender_flows(net, all_flows_from_sender)
    batch.append(
        (
            merged_fourtuples,
            merged_flowkeys,
            merged_min_rtt_us,
            merged_all_fets,
            merged_in_fets,
        )
    )
    logging.info(
        "Sender fairness waiting room for sender %s is full. "
        "Adding %d merged packets to batch.",
        utils.int_to_ip_str(merged_flowkeys[0].remote_addr),
        len(merged_in_fets),
    )
    # Additional packets_covered_by_batch
    return sum(len(flow_info[5]) for flow_info in all_flows_from_sender)


def maybe_run_batch(
    args,
    net,
    batch,
    flow_to_decisions,
    flow_to_rwnd,
    inference_flags,
    max_batch_time_s,
    batch_start_time_s,
):
    """Check if a batch is ready and process it."""
    packets_in_batch = sum(len(in_fets) for _, _, _, _, in_fets in batch)
    # Check if the batch is not ready yet, and if so, return False.
    # If the batch is full, then run inference. Also run inference if it has been a
    # long time since we ran inference last. A "long time" is defined as the max of
    # 1 second and the inference interval (if the inference interval is defined).
    batch_time_s = time.time() - batch_start_time_s
    if not batch or (
        packets_in_batch < args.batch_size and batch_time_s < max_batch_time_s
    ):
        logging.info(
            "Not ready to run batch yet. %d, %.2f >? %.2f",
            packets_in_batch,
            batch_time_s,
            max_batch_time_s,
        )
        return False

    logging.info("Running inference on a batch of %d flow(s).", len(batch))
    try:
        batch_inference(
            args,
            net,
            batch,
            flow_to_decisions,
            flow_to_rwnd,
        )
    except AssertionError:
        # Assertion errors mean this batch of packets violated some
        # precondition, but we are safe to skip them and continue.
        logging.warning(
            "Inference failed due to a non-fatal assertion failure:\n%s",
            traceback.format_exc(),
        )
    except Exception as exp:
        # An unexpected error occurred. It is not safe to continue. Reraise the
        # exception to kill the process.
        logging.error(
            "Inference failed due to an unexpected error:\n%s",
            traceback.format_exc(),
        )
        raise exp
    finally:
        for fourtuples, flowkeys, _, _, _ in batch:
            for fourtuple, flowkey in zip(fourtuples, flowkeys):
                logging.info(
                    "Clearing inference flag due to finished inference for flow: %s",
                    flowkey,
                )
                inference_flags[fourtuple].value = 0
        # Clear the batch.
        batch.clear()

    return True


def loop_iteration(
    args,
    net,
    dtype,
    batch,
    waiting_room,
    flow_to_prev_features,
    flow_to_decisions,
    flow_to_rwnd,
    inference_flags,
    que,
    packets_covered_by_batch,
    max_batch_time_s,
    batch_start_time_s,
    batch_wait_time_s,
):
    """Run one iteration of the inference loop.

    Includes: checking if the current batch is ready and running it, pulling a
    message from the inference queue, computing features, and deciding if the
    flow should wait (sender fairness) or be batched immediately.
    """
    # First, check whether we should run inference on the current batch.
    if maybe_run_batch(
        args,
        net,
        batch,
        flow_to_decisions,
        flow_to_rwnd,
        inference_flags,
        max_batch_time_s,
        batch_start_time_s,
    ):
        batch_proc_time_s = time.time() - batch_start_time_s - batch_wait_time_s
        pps = packets_covered_by_batch / batch_proc_time_s
        logging.info(
            "Inference performance: %.2f ms, %.2f pps, %.2f Mbps",
            batch_proc_time_s * 1e3,
            pps,
            pps * defaults.PACKET_LEN_B * 8 / 1e6,
        )
        # Reset: (packets_covered_by_batch, batch_start_time_s, batch_wait_time_s)
        return (0, time.time(), 0)

    # Get the next message from the inference queue.
    wait_start_time_s = time.time()
    val = None
    try:
        val = que.get(timeout=0.1)
    except queue.Empty:
        return (
            packets_covered_by_batch,
            batch_start_time_s,
            time.time() - wait_start_time_s,
        )
    batch_wait_time_s += time.time() - wait_start_time_s

    # For some reason, the queue returns True or None when the thread on the
    # other end dies.
    if not isinstance(val, tuple):
        return (
            packets_covered_by_batch,
            batch_start_time_s,
            batch_wait_time_s,
        )

    parse_res = parse_from_inference_queue(
        val,
        flow_to_rwnd,
        flow_to_decisions,
        flow_to_prev_features,
    )
    if parse_res is None:
        return (
            packets_covered_by_batch,
            batch_start_time_s,
            batch_wait_time_s,
        )
    (
        pkts,
        packets_lost,
        start_time_us,
        min_rtt_us,
        win_to_loss_event_rate,
        fourtuple,
        flowkey,
        sender_fairness,
        epoch,
        num_flows_expected,
    ) = parse_res

    build_res = build_features(
        args,
        net,
        dtype,
        pkts,
        packets_lost,
        start_time_us,
        min_rtt_us,
        win_to_loss_event_rate,
        inference_flags,
        flow_to_prev_features,
        fourtuple,
        flowkey,
    )
    if build_res is None:
        return (
            packets_covered_by_batch,
            batch_start_time_s,
            batch_wait_time_s,
        )
    all_fets, in_fets = build_res

    packets_covered_by_batch += wait_or_batch(
        net,
        sender_fairness,
        batch,
        waiting_room,
        fourtuple,
        flowkey,
        min_rtt_us,
        all_fets,
        in_fets,
        pkts,
        epoch,
        num_flows_expected,
    )
    return (
        packets_covered_by_batch,
        batch_start_time_s,
        batch_wait_time_s,
    )


def inference_loop(args, flow_to_rwnd, que, inference_flags, done):
    """Receive packets and run inference on them."""
    logging.info("Loading model: %s", args.model_file)
    if args.sender_fairness:
        net = models.MathisFairness()
    else:
        net = models.load_model(args.model_file)
    logging.info("Model features:\n\t%s", "\n\t".join(net.in_spc))
    flow_to_prev_features = {}
    # Maps flowkey to (decision, desired throughput, corresponding RWND)
    flow_to_decisions = collections.defaultdict(
        lambda: (defaults.Decision.NOT_PACED, None, None)
    )
    dtype = features.convert_to_float(
        sorted(
            list(
                # Manually add a feature for the number of packets lost.
                {(features.PACKETS_LOST_FET, "float64")}
                | set(features.PARSE_PACKETS_FETS)
                | set(
                    features.feature_names_to_dtype(
                        features.fill_dependencies(net.in_spc)
                    )
                )
            )
        )
    )
    logging.info("inference dtype %s", dtype)
    # Maximum duration to delay inference to wait to accumulate a batch.
    max_batch_time_s = max(
        0.1,
        (
            args.inference_interval_ms / 1e3
            if args.inference_interval_ms is not None
            else 1
        ),
    )
    logging.info("Inference process ready!")

    # Total packets covered by the current batch. The actual number of packets
    # on which inference will be computed is dependent on the smoothing window
    # and will be smaller than this.
    packets_covered_by_batch = 0
    # Time at which the current batch started.
    batch_start_time_s = time.time()
    # Time spent waiting for the inference queue, which should not count
    # towards time spent computing on the batch.
    batch_wait_time_s = 0
    batch = []
    # Maps sender IP to a tuple of:
    #     (epoch, expected number of flows,
    #      list of flow features from that sender)
    waiting_room = {}

    try:
        while not done.is_set():
            (
                packets_covered_by_batch,
                batch_start_time_s,
                batch_wait_time_s,
            ) = loop_iteration(
                args,
                net,
                dtype,
                batch,
                waiting_room,
                flow_to_prev_features,
                flow_to_decisions,
                flow_to_rwnd,
                inference_flags,
                que,
                packets_covered_by_batch,
                max_batch_time_s,
                batch_start_time_s,
                batch_wait_time_s,
            )
    except queue.Empty:
        return


def merge_fourtuples(fourtuples):
    """Merge multiple fourtuples from one sender by discarding the port numbers."""
    return (
        fourtuples[0]
        if len(fourtuples) == 1
        else (fourtuples[0][0], fourtuples[0][1], 0, 0)
    )


def batch_inference(
    args,
    net,
    batch,
    flow_to_decisions,
    flow_to_rwnd,
):
    """
    Run inference on a batch of flows.

    Note that each flow will occur only once in this batch due to inference_flags
    control.
    """
    num_pkts = sum(len(in_fets) for _, _, _, _, in_fets in batch)
    # Ranges are inclusive.
    flow_to_range = {}

    merged_fourtuples = [
        merge_fourtuples(fourtuples) for fourtuples, _, _, _, _ in batch
    ]

    running = 0
    for fourtuple, (_, _, _, _, in_fets) in zip(merged_fourtuples, batch):
        flow_to_range[fourtuple] = (running, running + len(in_fets))
        running += len(in_fets)

    # Assemble the batch into a single numpy array.
    batch_fets = np.empty(num_pkts, dtype=batch[0][4].dtype)
    for fourtuple, (_, _, _, _, in_fets) in zip(merged_fourtuples, batch):
        start, end = flow_to_range[fourtuple]
        batch_fets[start:end] = in_fets

    # Batch predict! Select only required features (remove unneeded features that were
    # added as dependencies for the requested features).
    labels = predict(net, batch_fets[list(net.in_spc)], args.debug)

    for fourtuple, (_, flowkeys, min_rtt_us, all_fets, in_fets) in zip(
        merged_fourtuples, batch
    ):
        start, end = flow_to_range[fourtuple]
        flw_labels = labels[start:end]
        # Make a decision for this flow.
        make_decision(
            args,
            flowkeys,
            min_rtt_us,
            all_fets,
            smooth(flw_labels),
            flow_to_decisions,
            flow_to_rwnd,
        )


def merge_sender_flows(net, sender_flows):
    """
    Merge multiple flows from the same sender into a single super-flow.

    sender_flows is a list of tuples of the form:
        (fourtuple, flowkey, min_rtt_us, all_fets, in_fets, pkts)
    """
    # Sanity check. Make sure that each flow has the same number of packets,
    # the same remote IP, and the same local IP. Make sure that all of the
    # remote ports are unique. This function only works if the net is a
    # MathisFairness model.
    assert isinstance(net, models.MathisFairness)
    target_num_pkts = sender_flows[0][4].shape[0]
    target_remote_ip = sender_flows[0][1].remote_addr
    target_local_ip = sender_flows[0][1].local_addr
    all_remote_ports = set()
    for _, flowkey, _, all_fets, in_fets, _ in sender_flows:
        assert in_fets.shape[0] == target_num_pkts
        assert all_fets.shape[0] == target_num_pkts
        assert flowkey.remote_addr == target_remote_ip
        assert flowkey.local_addr == target_local_ip
        all_remote_ports.add(flowkey.remote_port)
    assert len(all_remote_ports) == len(sender_flows)

    # Find last smoothing_window packets from across all flows in sender_flows..
    last_smoothing_window_times = []
    for _, _, _, all_fets, _, _ in sender_flows:
        last_smoothing_window_times.extend(all_fets[features.ARRIVAL_TIME_FET].tolist())
    last_smoothing_window_times.sort()
    last_smoothing_window_times = last_smoothing_window_times[-target_num_pkts:]

    # For each flow, interpolate at the times of those last smoothing_window packets.
    sender_flows_interp = []
    for _, _, _, all_fets, in_fets, _ in sender_flows:
        interp = np.empty(target_num_pkts, dtype=in_fets.dtype)
        for fet in in_fets.dtype.names:
            # Interpolate at the times of the last smoothing_window packets.
            interp[fet] = np.interp(
                last_smoothing_window_times,
                all_fets[features.ARRIVAL_TIME_FET],
                in_fets[fet],
            )
        sender_flows_interp.append(interp)

    # Merge the interpolated features across flows.
    merged_in_fets = np.empty(
        target_num_pkts, dtype=features.feature_names_to_dtype(net.in_spc)
    )
    for pkt_idx in range(target_num_pkts):
        # Average RTT across flows.
        average_rtt_us = np.average(
            [
                in_fets[pkt_idx][
                    features.make_win_metric(
                        features.RTT_FET, models.MathisFairness.win_size
                    )
                ]
                for in_fets in sender_flows_interp
            ]
        )
        # The loss rate is a fraction already distributed across all packets,
        # so it is representative of the relative loss that the combined flow
        # aught to experience.
        average_loss_event_rate = np.average(
            [
                in_fets[pkt_idx][
                    features.make_win_metric(
                        features.LOSS_EVENT_RATE_FET, models.MathisFairness.win_size
                    )
                ]
                for in_fets in sender_flows_interp
            ]
        )
        average_loss_rate = np.average(
            [
                in_fets[pkt_idx][
                    features.make_win_metric(
                        features.LOSS_RATE_FET, models.MathisFairness.win_size
                    )
                ]
                for in_fets in sender_flows_interp
            ]
        )
        merged_in_fets[pkt_idx] = (
            average_rtt_us,
            average_loss_event_rate,
            average_loss_rate,
            # Recompute the Mathis throughput.
            utils.safe_mathis_tput_bps(
                defaults.MSS_B, average_rtt_us, average_loss_event_rate
            ),
            utils.safe_mathis_tput_bps(
                defaults.MSS_B, average_rtt_us, average_loss_rate
            ),
            # Sum the throughput across flows.
            np.sum(
                [
                    in_fets[pkt_idx][
                        features.make_win_metric(
                            features.TPUT_FET, models.MathisFairness.win_size
                        )
                    ]
                    for in_fets in sender_flows_interp
                ]
            ),
        )

    logging.info("sender %s merged fets:", utils.int_to_ip_str(target_remote_ip))
    for fet in merged_in_fets.dtype.names:
        logging.info("merged '%s' %s", fet, merged_in_fets[fet])

    return (
        [fourtuple for fourtuple, _, _, _, _, _ in sender_flows],
        [flowkey for _, flowkey, _, _, _, _ in sender_flows],
        # The new min_rtt_us is the min of the min_rtt_us of the sender's flows.
        min(min_rtt_us for _, _, min_rtt_us, _, _, _ in sender_flows),
        # We do not generate all_fets for merged flows.
        merged_in_fets,
        merged_in_fets,
    )


def run(args, que, inference_flags, done):
    """Receive packets and run inference on them.

    This function is designed to be the target of a process.
    """

    def signal_handler(sig, frame):
        logging.info("Inference process: You pressed Ctrl+C!")
        done.set()

    signal.signal(signal.SIGINT, signal_handler)

    cleanup = None
    try:
        flow_to_rwnd, cleanup = configure_ebpf(args)
        if flow_to_rwnd is None:
            return
        inference_loop(args, flow_to_rwnd, que, inference_flags, done)
    except KeyboardInterrupt:
        logging.info("Inference process: You pressed Ctrl+C!")
        done.set()
    except:
        logging.exception("Unknown error in inference process!")
        raise
    finally:
        if cleanup is not None:
            cleanup()
