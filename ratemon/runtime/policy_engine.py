"""This module defines a process that will evaluate a policy on received packets."""

import collections
import ctypes
import logging
import queue
import signal
import time
import traceback

import numpy as np

from ratemon.model import data, defaults, features, gen_features, models, utils
from ratemon.model.defaults import Class
from ratemon.runtime import ebpf, flow_utils, policies
from ratemon.runtime.policies import Policy


def run(args, que, flags, done):
    """Receive packets and evaluate the policy on them.

    This function is designed to be the target of a process.
    """

    def signal_handler(sig, frame):
        logging.info("Policy engine: You pressed Ctrl+C!")
        done.set()

    signal.signal(signal.SIGINT, signal_handler)

    cleanup = None
    try:
        flow_to_rwnd, cleanup = ebpf.configure_ebpf(args)
        if flow_to_rwnd is None:
            return
        main_loop(args, flow_to_rwnd, que, flags, done)
    except KeyboardInterrupt:
        logging.info("Policy engine: You pressed Ctrl+C!")
        done.set()
    except Exception as exc:
        logging.exception("Unknown error in policy engine!")
        raise exc
    finally:
        if cleanup is not None:
            cleanup()


def main_loop(args, flow_to_rwnd, que, flags, done):
    """Receive packets and run evaluate the policy on them."""
    logging.info("Loading model: %s", args.model_file)
    net = policies.get_model_for_policy(args.policy, args.model_file)
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
    logging.info("Policy engine dtype %s", dtype)
    # Maximum duration to delay evaluation to wait to accumulate a batch.
    max_batch_time_s = max(
        0.1,
        (args.check_interval_ms / 1e3 if args.check_interval_ms is not None else 1),
    )
    logging.info("Policy engine ready!")

    # Total packets covered by the current batch. The actual number of packets
    # on which the policy will be computed is dependent on the smoothing window
    # and will be smaller than this.
    packets_covered_by_batch = 0
    # Time at which the current batch started.
    batch_start_time_s = time.time()
    # Time spent waiting for the policy engine queue, which should not count
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
                flags,
                que,
                packets_covered_by_batch,
                max_batch_time_s,
                batch_start_time_s,
                batch_wait_time_s,
            )
    except queue.Empty:
        return


def loop_iteration(
    args,
    net,
    dtype,
    batch,
    waiting_room,
    flow_to_prev_features,
    flow_to_decisions,
    flow_to_rwnd,
    flags,
    que,
    packets_covered_by_batch,
    max_batch_time_s,
    batch_start_time_s,
    batch_wait_time_s,
):
    """Run one iteration of the policy engine main loop.

    Includes: checking if the current batch is ready and running it, pulling a
    message from the policy engine queue, computing features, and deciding if the
    flow should wait (ServicePolicy) or be batched immediately (FlowPolicy).
    """
    # First, check whether we should process the current batch.
    if maybe_run_batch(
        args,
        net,
        batch,
        flow_to_decisions,
        flow_to_rwnd,
        flags,
        max_batch_time_s,
        batch_start_time_s,
    ):
        batch_proc_time_s = time.time() - batch_start_time_s - batch_wait_time_s
        pps = packets_covered_by_batch / batch_proc_time_s
        logging.info(
            "Policy engine performance: %.2f ms, %.2f pps, %.2f Mbps",
            batch_proc_time_s * 1e3,
            pps,
            pps * defaults.PACKET_LEN_B * 8 / 1e6,
        )
        # Reset: (packets_covered_by_batch, batch_start_time_s, batch_wait_time_s)
        return (0, time.time(), 0)

    # Get the next message from the policy engine queue.
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

    parse_res = parse_from_queue(
        args.policy,
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
        flags,
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
        args.policy,
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


def maybe_run_batch(
    args,
    net,
    batch,
    flow_to_decisions,
    flow_to_rwnd,
    flags,
    max_batch_time_s,
    batch_start_time_s,
):
    """Check if a batch is ready and process it."""
    packets_in_batch = sum(len(in_fets) for _, _, _, _, in_fets in batch)
    # Check if the batch is not ready yet, and if so, return False.
    # If the batch is full, then process it. Also process it if it has been a
    # long time since we processed it  last. A "long time" is defined as the max of
    # 1 second and the check interval (if the check interval is defined).
    batch_time_s = time.time() - batch_start_time_s
    if not batch or (
        packets_in_batch < args.batch_size and batch_time_s < max_batch_time_s
    ):
        logging.info(
            "Not ready to run batch yet. %d <? %d, %.2f >? %.2f",
            packets_in_batch,
            args.batch_size,
            batch_time_s,
            max_batch_time_s,
        )
        return False

    logging.info("Running policy engine on a batch of %d flow(s).", len(batch))
    try:
        batch_eval(
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
            "Policy engine failed due to a non-fatal assertion failure:\n%s",
            traceback.format_exc(),
        )
    except Exception as exp:
        # An unexpected error occurred. It is not safe to continue. Reraise the
        # exception to kill the process.
        logging.error(
            "Policy engine failed due to an unexpected error:\n%s",
            traceback.format_exc(),
        )
        raise exp
    finally:
        for fourtuples, flowkeys, _, _, _ in batch:
            for fourtuple, flowkey in zip(fourtuples, flowkeys):
                logging.info(
                    "Clearing flag due to finished processing for flow: %s",
                    flowkey,
                )
                flags[fourtuple].value = 0
        # Clear the batch.
        batch.clear()

    return True


def batch_eval(
    args,
    net,
    batch,
    flow_to_decisions,
    flow_to_rwnd,
):
    """
    Run the policy engine on a batch of flows.

    Note that each flow will occur only once in this batch due to flags
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

    if args.policy == Policy.FLOWPOLICY:
        # Batch predict! Only FlowPolicy actually uses a model. Select only required
        # features (remove unneeded features that were added as dependencies for the
        # requested features).
        labels = predict(net, batch_fets[list(net.in_spc)], args.debug)
    else:
        labels = [Class.NO_CLASS] * len(merged_fourtuples)

    # Make and execute a rate control decision.
    for fourtuple, (_, flowkeys, min_rtt_us, all_fets, _) in zip(
        merged_fourtuples, batch
    ):
        start, end = flow_to_range[fourtuple]
        flw_labels = labels[start:end]
        # Make a decision for this flow.
        new_decision = policies.make_decision(
            args.policy,
            flowkeys,
            net,
            min_rtt_us,
            all_fets,
            smooth(flw_labels),
            flow_to_decisions,
            args.reaction_strategy,
            args.schedule,
        )
        if new_decision is not None:
            for flowkey in flowkeys:
                apply_decision(flowkey, new_decision, flow_to_decisions, flow_to_rwnd)


def merge_fourtuples(fourtuples):
    """Merge multiple fourtuples from one sender by discarding the port numbers."""
    return (
        fourtuples[0]
        if len(fourtuples) == 1
        else (fourtuples[0][0], fourtuples[0][1], 0, 0)
    )


def predict(net, in_fets, debug=False):
    """Evaluate the FlowPolicy model on a batch of packets.

    Returns a list of labels (below target, near target, above target).
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

    return [Class(pred) for pred in preds]


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
                    "Warning: Flow %s asking for RWND < %d: %d. Overriding to %d.",
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


def parse_from_queue(
    policy, val, flow_to_rwnd, flow_to_decisions, flow_to_prev_features
):
    """Parse a message from the policy engine input queue."""
    epoch = num_flows_expected = None
    opcode, fourtuple = val[:2]
    flowkey = flow_utils.FlowKey(*fourtuple)

    if opcode.startswith("packets"):
        (
            pkts,
            packets_lost,
            start_time_us,
            min_rtt_us,
            win_to_loss_event_rate,
        ) = val[2:7]

        if policy == Policy.SERVICEPOLICY:
            assert len(val) == 10
            epoch, _, num_flows_expected = val[7:10]
    elif opcode == "remove":
        logging.info("Policy engine: Removing flow %s", flowkey)
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
    flags,
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
        logging.info("Clearing flag due to error for flow: %s", flowkey)
        flags[fourtuple].value = 0
        return None
    except Exception as exp:
        # An unexpected error occurred. It is not safe to continue. Reraise the
        # exception to kill the process.
        logging.error(
            "Feature generation failed due to an unexpected error:\n%s",
            traceback.format_exc(),
        )
        raise exp


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


def wait_or_batch(
    net,
    policy,
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
    """
    Decide whether a flow should be batched immediately (all but ServicePolicy) or wait
    (ServicePolicy).
    """
    if policy != Policy.SERVICEPOLICY:
        # Immediately put this flow into the batch.
        batch.append(([fourtuple], [flowkey], min_rtt_us, all_fets, in_fets))
        logging.info(
            "Adding %d packets from flow %s to batch.",
            len(in_fets),
            flowkey,
        )
        # Additional packets_covered_by_batch
        return len(pkts)

    # We are using ServicePolicy. We must wait until we have enough data from all flows
    # from this sender.

    # If this flow's sender does not have a waiting room, or if the waiting room is for
    # an earlier epoch, then create a new waiting room.
    if (
        flowkey.remote_addr not in waiting_room
        or waiting_room[flowkey.remote_addr][0] < epoch
    ):
        waiting_room[flowkey.remote_addr] = (
            int(epoch),
            int(num_flows_expected),
            [],
        )

    # Add this flow to its sender's waiting room.
    logging.info(
        "Adding %d packets from flow %s to ServicePolicy waiting room.",
        len(in_fets),
        flowkey,
    )
    waiting_room[flowkey.remote_addr][2].append(
        (fourtuple, flowkey, min_rtt_us, all_fets, in_fets, pkts)
    )

    # Check if the waiting room is full (i.e., has data from all flows from this
    # sender).
    if len(waiting_room[flowkey.remote_addr][2]) < waiting_room[flowkey.remote_addr][1]:
        # Waiting room is not full yet, so there are no new packets for the batch.
        return 0

    # The waiting room is full. Empty it, merge features across flows, and add the
    # combined features to the batch.
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
        "ServicePolicy waiting room for sender %s is full. "
        "Adding %d merged packets to batch.",
        utils.int_to_ip_str(merged_flowkeys[0].remote_addr),
        len(merged_in_fets),
    )
    # Additional packets_covered_by_batch
    return sum(len(flow_info[5]) for flow_info in all_flows_from_sender)


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
    assert isinstance(net, models.ServicePolicyModel)
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
                    features.make_win_metric(features.RTT_FET, net.win_size)
                ]
                for in_fets in sender_flows_interp
            ]
        )
        # The loss rate is a fraction already distributed across all packets, so it is
        # representative of the relative loss that the combined flow aught to
        # experience. I.e., each flow samples from the same loss distribution.
        min_loss_event_rate = np.min(
            [
                in_fets[pkt_idx][
                    features.make_win_metric(features.LOSS_EVENT_RATE_FET, net.win_size)
                ]
                for in_fets in sender_flows_interp
            ]
        )
        average_loss_rate = np.average(
            [
                in_fets[pkt_idx][
                    features.make_win_metric(features.LOSS_RATE_FET, net.win_size)
                ]
                for in_fets in sender_flows_interp
            ]
        )
        merged_in_fets[pkt_idx] = (
            average_rtt_us,
            min_loss_event_rate,
            average_loss_rate,
            # Recompute the Mathis throughput.
            utils.safe_mathis_tput_bps(
                defaults.MSS_B, average_rtt_us, min_loss_event_rate
            ),
            utils.safe_mathis_tput_bps(
                defaults.MSS_B, average_rtt_us, average_loss_rate
            ),
            # Sum the throughput across flows.
            np.sum(
                [
                    in_fets[pkt_idx][
                        features.make_win_metric(features.TPUT_FET, net.win_size)
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
