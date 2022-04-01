"""This module defines a process that will receive packets and run inference on them."""

import collections
import ctypes
from os import path
import pickle
import queue
import time

import torch

from unfair.model import data, defaults, features, gen_features, models, utils
from unfair.runtime import reaction_strategy
from unfair.runtime.reaction_strategy import ReactionStrategy


def load_model(model, model_file):
    """Load the provided trained model."""
    assert path.isfile(model_file), f"Model does not exist: {model_file}"
    net = models.MODELS[model]()
    with open(model_file, "rb") as fil:
        net.net = pickle.load(fil)
    return net


def featurize(flowkey, net, pkts, min_rtt_us, debug=False):
    """Compute features for the provided list of packets.

    Returns a structured numpy array.
    """
    fets, min_rtt_us = gen_features.parse_received_acks(
        net.in_spc, flowkey, pkts, min_rtt_us, debug
    )

    data.replace_unknowns(fets, isinstance(net, models.HistGbdtSklearnWrapper))
    return fets, min_rtt_us


def inference(net, flowkey, pkts, min_rtt_us, debug=False):
    """Run inference on a flow's packets.

    Returns a label: below fair, approximately fair, above fair.
    """
    fets, min_rtt_us = featurize(flowkey, net, pkts, min_rtt_us, debug)
    preds = net.predict(
        torch.tensor(
            utils.clean(fets),
            dtype=torch.float,
        )
    )
    return [defaults.Class(pred) for pred in preds], min_rtt_us


def condense_labels(labels):
    """Combine multiple labels into a single label.

    For example, smooth the labels by selecting the average label.

    Currently, this simply selects the last label.
    """
    assert len(labels) > 0, "Labels cannot be empty."
    return labels[-1]


def make_decision(
    flowkey, label, pkts_ndarray, min_rtt_us, decisions, flow_to_rwnd, args
):
    """Make a flow unfairness mitigation decision.

    Base the decision on the flow's label and existing decision. Use the flow's packets
    to calculate any necessary flow metrics, such as the throughput.

    TODO: Instead of passing in using the flow's packets, pass in the features and make
          sure that they include the necessary columns.
    """
    if args.reaction_strategy == ReactionStrategy.FILE:
        new_decision = (
            defaults.Decision.PACED,
            reaction_strategy.get_scheduled_pacing(args.schedule),
        )
    else:
        tput_bps = utils.safe_tput_bps(pkts_ndarray, 0, len(pkts_ndarray) - 1)

        if label == defaults.Class.ABOVE_FAIR:
            # This flow is sending too fast. Force the sender to halve its rate.
            new_decision = (
                defaults.Decision.PACED,
                reaction_strategy.react_down(
                    args.reaction_strategy,
                    utils.bdp_B(tput_bps, min_rtt_us / 1e6),
                ),
            )
        elif decisions[flowkey] == defaults.Decision.PACED:
            # We are already pacing this flow.
            if label == defaults.Class.BELOW_FAIR:
                # If we are already pacing this flow but we are being too
                # aggressive, then let it send faster.
                new_decision = (
                    defaults.Decision.PACED,
                    reaction_strategy.react_up(
                        args.reaction_strategy,
                        utils.bdp_B(tput_bps, min_rtt_us / 1e6),
                    ),
                )
            else:
                # If we are already pacing this flow and it is behaving as desired,
                # then all is well. Retain the existing pacing decision.
                new_decision = decisions[flowkey]
        else:
            # This flow is not already being paced and is not behaving unfairly, so
            # leave it alone.
            new_decision = (defaults.Decision.NOT_PACED, None)

    # FIXME: Why are the BDP calculations coming out so small? Is the throughput
    #        just low due to low application demand?

    if decisions[flowkey] != new_decision:
        print(f"New decision for flow {flowkey}: {new_decision}")

        if new_decision[1] is None:
            del flow_to_rwnd[flowkey]
        else:
            new_decision = (new_decision[0], round(new_decision[1]))
            assert new_decision[1] > 0, (
                "Error: RWND must be greater than 0, "
                f"but is {new_decision[1]} for flow {flowkey}."
            )
            # if new_decision[1] > 2**16:
            #     print(f"Warning: Asking for RWND >= 2**16: {new_decision[1]}")
            #     new_decision[1] = 2**16 - 1

            flow_to_rwnd[flowkey] = ctypes.c_ushort(new_decision[1])

        decisions[flowkey] = new_decision


def packets_to_ndarray(pkts):
    """Reorganize a list of packet metrics into a structured numpy array."""
    # For some reason, the packets tend to get reordered after they are timestamped on
    # arrival. Sort packets by timestamp.
    pkts = sorted(pkts, key=lambda pkt: pkt[-1])
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


def run(args, que, done, flow_to_rwnd):
    """Receive packets and run inference on them.

    This function is designed to be the target of a process.
    """
    net = load_model(args.model, args.model_file)
    min_rtts_us = collections.defaultdict(lambda: float("int"))
    decisions = collections.defaultdict(lambda: (defaults.Decision.NOT_PACED, None))

    while not done.is_set():
        try:
            flowkey, pkts = que.get(timeout=1)
        except queue.Empty:
            continue

        start_time_s = time.time()
        pkts_ndarray = packets_to_ndarray(pkts)
        try:
            labels, min_rtts_us[flowkey] = inference(
                net, flowkey, pkts, min_rtts_us[flowkey], args.debug
            )
        except AssertionError as exp:
            # FIXME: There is a strange bug when normalizing the packet arrival times
            # that causes the arrival times to not be in order even though we sort the
            # packets. If this (or any other assertion error) occurs, then just skip
            # this batch of packets.
            print(f"Error, skipping batch of packets: {exp}")
            return
        finally:
            print(f"Inference took: {(time.time() - start_time_s) * 1e3:.2f} ms")

        make_decision(
            flowkey,
            condense_labels(labels),
            pkts_ndarray,
            min_rtts_us[flowkey],
            decisions,
            flow_to_rwnd,
            args,
        )
        # print(f"Report for flow {flow}: {flow.label}, {flow.decision}")
