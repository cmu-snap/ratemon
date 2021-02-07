#! /usr/bin/env python3
"""Parses the output of CloudLab experiments. """

import argparse
import collections
import itertools
import math
import multiprocessing
import subprocess
import os
from os import path
import random
import shutil
import time

import numpy as np

import cl_args
import defaults
import utils


# Assemble the output dtype.
#
# These metrics do not change.
REGULAR = [
    ("seq", "int32"),
    ("arrival time us", "int32"),
    ("min RTT us", "int32"),
    ("flow share percentage", "float64"),
    ("interarrival time us", "int32"),
    ("throughput b/s", "float32"),
    ("packets lost since last packet estimate", "int32"),
    ("loss rate at queue", "float64"),
    ("retransmission rate", "float64"),
    ("payload B", "int32"),
    ("total so far B", "int32"),
    ("RTT estimate us", "int32")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("interarrival time us", "float64"),
    ("throughput p/s", "float64"),
    ("RTT estimate us", "float64"),
    ("RTT estimate ratio", "float64"),
    ("RTT true us", "float64"),
    ("RTT true ratio", "float64"),
    ("loss rate estimate", "float64"),
    ("loss rate true", "float64"),
    ("mathis model throughput p/s", "float64"),
    # -1 no applicable (no loss yet), 0 lower than or equal to fair
    # throughput, 1 higher. This is not an EWMA metric itself, but is
    # based on the "mathis model throughput p/s" metric.
    ("mathis model label", "int32")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    ("average interarrival time us", "float64"),
    ("average throughput p/s", "float64"),
    ("average RTT estimate us", "float64"),
    ("average RTT estimate ratio", "float64"),
    ("average RTT true us", "float64"),
    ("average RTT true ratio", "float64"),
    ("loss event rate", "float64"),
    ("1/sqrt loss event rate", "float64"),
    ("loss rate estimate", "float64"),
    ("loss rate true", "float64"),
    ("mathis model throughput p/s", "float64"),
    # -1 no applicable (no loss yet), 0 lower than or equal to fair
    # throughput, 1 higher. This is not a windowed metric itself, but
    # is based on the "mathis model throughput p/s" metric.
    ("mathis model label", "int32")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 1000 for i in range(1, 11)] + [i / 10 for i in range(1, 11)]
# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(11)]
# Mathis model constant.
MATHIS_C = math.sqrt(3 / 2)


def make_ewma_metric(metric, alpha):
    """ Format the name of an EWMA metric. """
    return f"{metric}-ewma-alpha{alpha}"


def make_win_metric(metric, win):
    """ Format the name of a windowed metric. """
    return f"{metric}-windowed-minRtt{win}"


def make_interval_weight(num_intervals):
    """ Used to calculate loss event rate. """
    return [
        (1 if i < num_intervals / 2 else
         2 * (num_intervals - i) / (num_intervals + 2))
        for i in range(num_intervals)]


def compute_weighted_average(curr_event_size, loss_event_intervals,
                             loss_interval_weights):
    """ Used to calculate loss event rate. """
    weight_total = 1 + sum(loss_interval_weights[1:])
    interval_total_0 = (curr_event_size + sum(
        interval * weight
        for interval, weight in zip(
            list(loss_event_intervals)[:-1], loss_interval_weights[1:])))
    interval_total_1 = sum(
        interval * weight
        for interval, weight in zip(
            loss_event_intervals, loss_interval_weights))
    return weight_total / max(interval_total_0, interval_total_1)


def loss_rate(loss_q, win_start_idx, pkt_loss_cur, recv_time_cur,
              recv_time_prev, win_size_us, pkt_idx):
    """ Calculates the loss rate over a window. """
    # If there were packet losses since the last received packet, then
    # add them to the loss queue.
    if pkt_loss_cur > 0 and pkt_idx > 0:
        # The average time between when packets should have arrived,
        # since we received the last packet.
        loss_interval = ((recv_time_cur - recv_time_prev) / (pkt_loss_cur + 1))
        # Look at each lost packet...
        for k in range(pkt_loss_cur):
            # Record the time at which this packet should have
            # arrived.
            loss_q.append(recv_time_prev + (k + 1) * loss_interval)

    # Discard old losses.
    while loss_q and (loss_q[0] < recv_time_cur - win_size_us):
        loss_q.popleft()

    # The loss rate is the number of losses in the window divided by
    # the total number of packets in the window.
    win_losses = len(loss_q)
    return (
        loss_q,
        ((win_losses / (pkt_idx + win_losses - win_start_idx))
         if pkt_idx - win_start_idx > 0 else 0))


def parse_pcap(sim_dir, untar_dir, out_dir, skip_smoothed):
    """ Parse a PCAP file. """
    print(f"Parsing: {sim_dir}")
    sim = utils.Sim(sim_dir)
    tot_flws = sim.cca_1_flws + sim.cca_2_flws
    assert tot_flws > 0, f"No flows to analyze: {sim_dir}"

    # Construct the output filepaths.
    out_flp = path.join(out_dir, f"{sim.name}.npz")
    # If the output file exists, then we do not need to parse this file.
    if path.exists(out_flp):
        print(f"    Already parsed: {sim_dir}")
        return

    # Create a temporary folder to untar experiments.
    if not path.exists(untar_dir):
        os.mkdir(untar_dir)
    exp_dir = path.join(untar_dir, sim.name)
    # If this experiment already has been untarred, then delete the old files.
    if path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    subprocess.check_call(["tar", "-xf", sim_dir, "-C", untar_dir])

    # Process PCAP files from senders and receivers.
    # The final output, with one entry per flow.
    flws = []

    # Create the (super-complicated) dtype. The dtype combines each metric at
    # multiple granularities.
    dtype = (REGULAR + (
        [] if skip_smoothed else (
            [(make_ewma_metric(metric, alpha), typ)
             for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
            [(make_win_metric(metric, win), typ)
             for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)])))

    for flw_idx in range(tot_flws):
        try:
            # Packet lists are of tuples of the form:
            #     (seq, sender, timestamp us, timestamp option)
            sent_pkts = utils.parse_packets(path.join(
                exp_dir, f"client-tcpdump-{sim.name}.pcap"), flw_idx)

            recv_flp = path.join(exp_dir, f"server-tcpdump-{sim.name}.pcap")
            recv_data_pkts = utils.parse_packets(
                recv_flp, flw_idx, direction="data")
            # Ack packets for RTT calculation
            recv_ack_pkts = utils.parse_packets(
                recv_flp, flw_idx, direction="ack")
        except RuntimeError as exc:
            # If this experiment does not have any flows, then skip it.
            print(f"Unable to parse packets for flow {flw_idx}: {exc}")
            flws.append(np.empty(0, dtype=dtype))

            # Create these variables so that there is not a error when they are
            # explicitly deleted after all flows have been processed.
            sent_pkts = None
            recv_data_pkts = None
            recv_ack_pkts = None
            continue

        # The final output. -1 implies that a value could not be calculated.
        output = np.empty(len(recv_data_pkts), dtype=dtype)
        output.fill(-1)

        # If this flow does not have any packets, then skip it.
        skip = False
        if not sent_pkts:
            skip = True
            print(
                f"Warning: No data packets sent for flow {flw_idx} in: "
                f"{sim_dir}")
        if not recv_data_pkts:
            skip = True
            print(
                f"Warning: No data packets received for flow {flw_idx} in: "
                f"{sim_dir}")
        if not recv_ack_pkts:
            skip = True
            print(
                f"Warning: No ACK packets sent for flow {flw_idx} in: "
                f"{sim_dir}")
        if skip:
            flws.append(output)
            continue

        # State that the windowed metrics need to track across packets.
        win_state = {win: {
            # The index at which this window starts.
            "window_start_idx": 0,
            # The "loss event rate".
            "loss_interval_weights": make_interval_weight(8),
            "loss_event_intervals": collections.deque(),
            "current_loss_event_start_idx": 0,
            "current_loss_event_start_time": 0,
            # For "loss rate true".
            "loss_queue_true": collections.deque(),
            # For "loss rate estimated".
            "loss_queue_estimate": collections.deque()
        } for win in WINDOWS}
        # Total number of packet losses up to the current received
        # packet.
        # pkt_loss_total_true = 0
        pkt_loss_total_estimate = 0
        # Loss rate estimation.
        prev_pkt_seq = 0
        highest_seq = 0
        # RTT estimation.
        ack_idx = 0

        for j, recv_pkt in enumerate(recv_data_pkts):
            if j % 1000 == 0:
                print(
                    f"Flow {flw_idx + 1}/{tot_flws}: "
                    f"{j}/{len(recv_data_pkts)} packets")
            # Regular metrics.
            recv_pkt_seq = recv_pkt[0]
            recv_time_cur = recv_pkt[2]

            output[j]["seq"] = recv_pkt_seq
            # Align arrival time to zero such that all features starts at 0s
            output[j]["arrival time us"] = recv_time_cur

            if j > 0:
                # Receiver-side RTT estimation using the TCP timestamp
                # option. Attempt to find a new RTT estimate. Move
                # ack_idx to the first occurance of the timestamp
                # option TSval corresponding to the current packet's
                # TSecr.
                tsval = recv_ack_pkts[ack_idx][3][0]
                tsecr = recv_pkt[3][1]
                ack_idx_old = ack_idx
                while tsval != tsecr and ack_idx < len(recv_ack_pkts) - 1:
                    ack_idx += 1
                    tsval = recv_ack_pkts[ack_idx][3][0]
                if tsval == tsecr:
                    # If we found a timestamp option match, then
                    # update the RTT estimate.
                    rtt_estimate_us = recv_time_cur - recv_ack_pkts[ack_idx][2]
                else:
                    # Otherwise, use the previous RTT estimate and
                    # reset ack_idx to search again for the next
                    # packet.
                    rtt_estimate_us = output[j - 1]["RTT estimate us"]
                    ack_idx = ack_idx_old
                # Update the min RTT estimate.
                min_rtt_us = utils.safe_min(
                    output[j - 1]["min RTT us"], rtt_estimate_us)
                output[j]["min RTT us"] = min_rtt_us
                # Compute the new RTT ratio.
                rtt_estimate_ratio = utils.safe_div(rtt_estimate_us, min_rtt_us)

                # Calculate the inter-arrival time.
                recv_time_prev = recv_data_pkts[j - 1][2]
                interarr_time_us = recv_time_cur - recv_time_prev
            else:
                rtt_estimate_us = -1
                rtt_estimate_ratio = -1
                min_rtt_us = -1
                recv_time_prev = -1
                interarr_time_us = -1

            output[j]["interarrival time us"] = interarr_time_us
            payload_B = recv_pkt[4]
            output[j]["payload B"] = payload_B

            output[j]["total so far B"] = (
                payload_B + (0 if j == 0 else output[j - 1]["total so far B"]))

            output[j]["throughput b/s"] = utils.safe_mul(
                utils.safe_mul(
                    utils.safe_div(1e6, interarr_time_us), payload_B),
                8)
            output[j]["RTT estimate us"] = rtt_estimate_us

            # Receiver-side loss rate estimation. Estimate the losses
            # since the last packet.
            pkt_loss_cur_estimate = math.ceil(
                0
                if recv_pkt_seq == prev_pkt_seq + payload_B
                else (
                    (recv_pkt_seq - highest_seq - payload_B) / payload_B
                    if recv_pkt_seq > highest_seq + payload_B
                    else (
                        1
                        if (recv_pkt_seq < prev_pkt_seq and
                            prev_pkt_seq != highest_seq)
                        else 0)))
            pkt_loss_total_estimate += pkt_loss_cur_estimate
            prev_pkt_seq = recv_pkt_seq
            highest_seq = max(highest_seq, prev_pkt_seq)

            output[j]["packets lost since last packet estimate"] = (
                pkt_loss_cur_estimate)

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                if skip_smoothed:
                    continue

                metric = make_ewma_metric(metric, alpha)
                if "interarrival time us" in metric:
                    new = interarr_time_us
                elif ("throughput p/s" in metric and
                      "mathis model" not in metric):
                    # Do not use the existing interarrival EWMA to
                    # calculate the throughput. Instead, use the true
                    # interarrival time so that the value used to
                    # update the throughput EWMA is not "EWMA-ified"
                    # twice. Divide by 1e6 to convert from
                    # microseconds to seconds.
                    output[j][metric] = utils.safe_div(
                        1e6,
                        output[j][make_ewma_metric(
                            "interarrival time us", alpha)])
                    continue
                elif "RTT estimate us" in metric:
                    new = rtt_estimate_us
                elif "RTT estimate ratio" in metric:
                    new = rtt_estimate_ratio
                elif "RTT true us" in metric:
                    new = rtt_true_us
                elif "RTT true ratio" in metric:
                    new = rtt_true_ratio
                elif "loss rate estimate" in metric:
                    # See comment in case for "loss rate true".
                    new = pkt_loss_cur_estimate / (
                        pkt_loss_cur_estimate + 1)
                elif "loss rate true" in metric:
                    # Divide the pkt_loss_cur_true by
                    # (pkt_loss_cur_true + 1) because over the course
                    # of sending (pkt_loss_cur_true + 1) packets, one
                    # got through and pkt_loss_cur_true were lost.
                    new = pkt_loss_cur_true / (pkt_loss_cur_true + 1)
                elif "mathis model throughput p/s" in metric:
                    # Use the estimated loss rate to compute the
                    # Mathis model fair throughput. Contrary to the
                    # decision for interarrival time, above, here we
                    # use the value of another EWMA (loss rate
                    # estimate) to compute the new value for the
                    # Mathis model throughput EWMA. I believe that
                    # this is desirable because we want to see how the
                    # metric as a whole reacts to a certain degree of
                    # memory.
                    loss_rate_estimate = (
                        pkt_loss_total_estimate / j if j > 0 else -1)
                    # Use "safe" operations in case any of the
                    # supporting values are -1 (unknown).
                    new = (
                        -1 if loss_rate_estimate <= 0 else
                        utils.safe_div(
                            MATHIS_C,
                            utils.safe_div(
                                utils.safe_mul(
                                    min_rtt_us,
                                    utils.safe_sqrt(loss_rate_estimate)),
                                1e6)))
                elif "mathis model label" in metric:
                    # Use the current throughput and the Mathis model
                    # fair throughput to compute the Mathis model
                    # label.
                    output[j][metric] = utils.safe_mathis_label(
                        output[j][make_ewma_metric(
                            "throughput p/s", alpha)],
                        output[j][make_ewma_metric(
                            "mathis model throughput p/s", alpha)])
                    # Continue because the value of this metric is not
                    # an EWMA.
                    continue
                else:
                    raise Exception(f"Unknown EWMA metric: {metric}")
                # Update the EWMA.
                output[j][metric] = utils.safe_update_ewma(
                    -1 if j == 0 else output[j - 1][metric], new, alpha)

            # Windowed metrics.
            for (metric, _), win in itertools.product(WINDOWED, WINDOWS):
                if skip_smoothed:
                    continue

                metric = make_win_metric(metric, win)
                # If we have not been able to estimate the min RTT
                # yet, then we cannot compute any of the windowed
                # metrics.
                if min_rtt_us == -1:
                    continue
                win_size_us = win * min_rtt_us

                # Move the start of the window forward.
                while ((recv_time_cur -
                        recv_data_pkts[win_state[win]["window_start_idx"]][2]) >
                       win_size_us):
                    win_state[win]["window_start_idx"] += 1
                win_start_idx = win_state[win]["window_start_idx"]

                if "average interarrival time us" in metric:
                    new = ((recv_time_cur - recv_data_pkts[win_start_idx][2]) /
                           (j - win_start_idx + 1))
                elif "average throughput p/s" in metric:
                    # We base the throughput calculation on the
                    # average interarrival time over the window.
                    avg_interarr_time_us = output[j][
                        make_win_metric("average interarrival time us", win)]
                    # Divide by 1e6 to convert from microseconds to
                    # seconds.
                    new = utils.safe_div(
                        1, utils.safe_div(avg_interarr_time_us, 1e6))
                elif "average RTT estimate us" in metric:
                    new = utils.safe_mean(
                        output[make_ewma_metric("RTT estimate us", alpha=1.)],
                        win_start_idx, j)
                elif "average RTT estimate ratio" in metric:
                    new = utils.safe_mean(
                        output[
                            make_ewma_metric("RTT estimate ratio", alpha=1.)],
                        win_start_idx, j)
                elif "average RTT true us" in metric:
                    new = utils.safe_mean(
                        output[make_ewma_metric("RTT true us", alpha=1.)],
                        win_start_idx, j)
                elif "average RTT true ratio" in metric:
                    new = utils.safe_mean(
                        output[make_ewma_metric("RTT true ratio", alpha=1.)],
                        win_start_idx, j)
                elif "loss event rate" in metric and "1/sqrt" not in metric:
                    rtt_estimate_us = output[
                        j][make_win_metric("average RTT estimate us", win)]
                    if rtt_estimate_us == -1:
                        # The RTT estimate is -1 (unknown), so we
                        # cannot compute the loss event rate.
                        continue

                    cur_start_idx = win_state[
                        win]["current_loss_event_start_idx"]
                    cur_start_time = win_state[
                        win]["current_loss_event_start_time"]
                    if pkt_loss_cur_estimate > 0:
                        # There was a loss since the last packet.
                        #
                        # The index of the first packet in the current
                        # loss event.
                        new_start_idx = (j + pkt_loss_total_estimate -
                                         pkt_loss_cur_estimate)

                        if cur_start_idx == 0:
                            # This is the first loss event.
                            #
                            # Naive fix for the loss event rate
                            # calculation The described method in the
                            # RFC is complicated for the first event
                            # handling.
                            cur_start_idx = 1
                            cur_start_time = 0
                            new = 1 / j
                        else:
                            # This is not the first loss event. See if
                            # any of the newly-lost packets start a
                            # new loss event.
                            #
                            # The average time between when packets
                            # should have arrived, since we received
                            # the last packet.
                            loss_interval = ((recv_time_cur - recv_time_prev) /
                                             (pkt_loss_cur_estimate + 1))

                            # Look at each lost packet...
                            for k in range(pkt_loss_cur_estimate):
                                # Compute the approximate time at
                                # which the packet should have been
                                # received if it had not been lost.
                                loss_time = (
                                    recv_time_prev + (k + 1) * loss_interval)

                                # If the time of this loss is more
                                # than one RTT from the time of the
                                # start of the current loss event,
                                # then this is a new loss event.
                                if (loss_time - cur_start_time >=
                                        rtt_estimate_us):
                                    # Record the number of packets
                                    # between the start of the new
                                    # loss event and the start of the
                                    # previous loss event.
                                    win_state[
                                        win]["loss_event_intervals"].appendleft(
                                            new_start_idx - cur_start_idx)
                                    # Potentially discard an old event.
                                    if len(win_state[
                                            win]["loss_event_intervals"]) > win:
                                        win_state[
                                            win]["loss_event_intervals"].pop()

                                    cur_start_idx = new_start_idx
                                    cur_start_time = loss_time
                                # Calculate the index at which the
                                # new loss event begins.
                                new_start_idx += 1

                            new = compute_weighted_average(
                                (j + pkt_loss_total_estimate -
                                 cur_start_idx),
                                win_state[win]["loss_event_intervals"],
                                win_state[win]["loss_interval_weights"])
                    elif pkt_loss_total_estimate > 0:
                        # There have been no losses since the last
                        # packet, but the total loss is nonzero.
                        # Increase the size of the current loss event.
                        new = compute_weighted_average(
                            j + pkt_loss_total_estimate - cur_start_idx,
                            win_state[win]["loss_event_intervals"],
                            win_state[win]["loss_interval_weights"])
                    else:
                        # There have never been any losses, so the
                        # loss event rate is 0.
                        new = 0

                    # Record the new values of the state variables.
                    win_state[
                        win]["current_loss_event_start_idx"] = cur_start_idx
                    win_state[
                        win]["current_loss_event_start_time"] = cur_start_time
                elif "1/sqrt loss event rate" in metric:
                    # Use the loss event rate to compute
                    # 1 / sqrt(loss event rate).
                    new = utils.safe_div(
                        1, utils.safe_sqrt(
                            output[j][make_win_metric("loss event rate", win)]))
                elif "loss rate estimate" in metric:
                    # We do not need to check whether recv_time_prev
                    # is -1 (unknown) because the windowed metrics
                    # skip the case where j == 0.
                    win_state[win]["loss_queue_estimate"], new = loss_rate(
                        win_state[win]["loss_queue_estimate"], win_start_idx,
                        pkt_loss_cur_estimate, recv_time_cur, recv_time_prev,
                        win_size_us, j)
                elif "loss rate true" in metric:
                    # We do not need to check whether recv_time_prev
                    # is -1 (unknown) because the windowed metrics
                    # skip the case where j == 0.
                    win_state[win]["loss_queue_true"], new = loss_rate(
                        win_state[win]["loss_queue_true"], win_start_idx,
                        pkt_loss_cur_true, recv_time_cur, recv_time_prev,
                        win_size_us, j)
                elif "mathis model throughput p/s" in metric:
                    # Use the loss event rate to compute the Mathis
                    # model fair throughput.
                    loss_rate_estimate = (
                        pkt_loss_total_estimate / j if j > 0 else -1)
                    new = utils.safe_div(
                        MATHIS_C,
                        utils.safe_div(
                            utils.safe_mul(
                                min_rtt_us,
                                utils.safe_sqrt(
                                    loss_rate_estimate)),
                            1e6))
                elif "mathis model label" in metric:
                    # Use the current throughput and Mathis model
                    # fair throughput to compute the Mathis model
                    # label.
                    new = utils.safe_mathis_label(
                        output[j][make_win_metric(
                            "average throughput p/s", win)],
                        output[j][make_win_metric(
                            "mathis model throughput p/s", win)])
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new

        # Calculate the number of retransmissions. Truncate the sent packets
        # at the last occurence of the last packet to be received.
        #
        # Get the sequence number of the last received packet.
        last_seq = output[-1]["seq"]
        # Find when this packet was sent. Assume that if this packet was
        # retransmitted, then the last retransmission is the one that arrived at
        # the receiver (which may be an incorrect assumption).
        sent_idx = len(sent_pkts) - 1
        while sent_idx >= 0 and sent_pkts[sent_idx][0] != last_seq:
            sent_idx -= 1
        # Convert from index to packet count.
        sent_idx += 1
        sent_seqs = set()
        for sent_pkt in sent_pkts[:sent_idx]:
            sent_seqs.add(sent_pkt[0])
        if sent_idx == 0:
            print(
                "Warning: Did not find when the last received packet was sent.")
        else:
            output[-1]["retransmission rate"] = 1 - (len(sent_seqs) / sent_idx)

        # toks = sim.name.split("-")
        # q_log_flp = path.join(
        #     exp_dir,
        #     "-".join(toks[:-1]) + "-forward-bottleneckqueue-" + toks[-1] +
        #     ".log")

        # # Calculate the raw drop rate at the bottleneck queue.
        # deq_idx = None
        # drop_rate = None

        # print(f"last_seq: {last_seq}")
        # with open(q_log_flp, "r") as fil:
        #     # Find the dequeue log corresponding to the last packet that was
        #     # received.
        #     for line_idx, line in reversed(list(enumerate(fil))):
        #         # Need to filter by flow.

        #         if line.startswith("1"):
        #             seq = int(line.split(",")[3], 16)
        #             #print(f"Checking seq: {seq}")
        #             if seq == last_seq:
        #                 deq_idx = line_idx
        #                 break
        #     if deq_idx is None:
        #         print(
        #             "Warning: Did not find when last received packet was "
        #             "dequeued.")
        #     else:
        #         # Find the most recent stats log before the last received
        #         # packet was dequeued.
        #         for line_idx, line in reversed(list(enumerate(fil[:deq_idx]))):
        #             if line.startswith("stats"):
        #                 enqueued, _, dropped = [
        #                     int(tok) for tok in line.split(":")[1].split(",")]
        #                 drop_rate = dropped / (enqueued + dropped)
        #                 break
        # if drop_rate is None:
        #     print(
        #         "Warning: Did not calculate the loss rate at the bottleneck "
        #         "queue.")
        # else:
        #     output[-1]["loss rate at queue"] = drop_rate
        # print(f"loss rate at queue: {output[-1]['loss rate at queue']}")

        flws.append(output)

    # Save memory by explicitly deleting the sent and received packets
    # after they have been parsed. This happens outside of the above
    # for-loop because only the last iteration's packets are not
    # automatically cleaned up by now (they go out of scope when the
    # *_pkts variables are overwritten by the next loop).
    del sent_pkts
    del recv_data_pkts
    del recv_ack_pkts

    # Sum the total throughput of all flows, and then compute the flow
    # percentage used by each flow
    # extra_time = 20  # Flows could end later than start_time + duration
    # ground_truth_throughput = "throughput p/s-ewma-alpha0.003"
    # arrival_time_key = "arrival time us"
    # x_vals = np.arange((sim.dur_s + extra_time) * 1000, dtype=float) / 1000
    # total_throughput = [0] * ((sim.dur_s + extra_time) * 1000)
    # interp_list = []
    # for flw_dat in flws:
    #     per_flow_throughput = flw_dat[ground_truth_throughput]
    #     interpolated = np.interp(
    #         x_vals,
    #         # Convert from us to s.
    #         xp=flw_dat[arrival_time_key] / 1e6,
    #         fp=per_flow_throughput)
    #     interp_list.append(interpolated)
    #     total_throughput = np.add(total_throughput, interpolated)

    # for k, flw_dat in enumerate(flws):
    #     per_flow_fraction = np.divide(
    #         interp_list[k],
    #         total_throughput)
    #     # Sometimes np.divide could yield weird fraction greater than 1
    #     percentage = list(map(
    #         lambda x: min(1.0, per_flow_fraction[int(x / 1000)]),
    #         flw_dat[arrival_time_key]))
    #     flw_dat["flow share percentage"] = percentage

    if not skip_smoothed:
        total_throughput_p = sim.bw_Mbps * 1e6 / 8 / 1448
        # Use index variables to make sure that no data is being copied.
        # TODO: Is this a correct idea?
        for j in range(len(flws)):
            # I think that these are vector operations that assign an entire column
            # at a time.
            per_flow_throughput = flws[j]["throughput p/s-ewma-alpha0.003"]
            flws[j]["flow share percentage"] = (
                per_flow_throughput / total_throughput_p)

    # Determine if there are any NaNs or Infs in the results. For the
    # results for each flow, look through all features (columns) and
    # make a note of the features that bad values. Flatten these lists
    # of feature names, using a set comprehension to remove
    # duplicates.
    bad_fets = {
        fet for flw_dat in flws
        for fet in flw_dat.dtype.names if not np.isfinite(flw_dat[fet]).all()}
    if bad_fets:
        print(f"    Simulation {untar_dir} has NaNs of Infs in features: "
              f"{bad_fets}")

    # Save the results.
    if path.exists(out_flp):
        print(f"    Output already exists: {out_flp}")
    else:
        print(f"    Saving: {out_flp}")
        np.savez_compressed(
            out_flp, **{str(k + 1): v for k, v in enumerate(flws)})

    # Remove untarred folder
    shutil.rmtree(untar_dir)


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Parses the output of CloudLab experiments.")
    psr.add_argument(
        "--exp-dir",
        help=("The directory in which the experiment results are stored "
              "(required)."), required=True, type=str)
    psr.add_argument(
        "--untar-dir",
        help=("The directory in which the untarred experiment intermediate "
              "files are stored (required)."),
        required=True, type=str)
    psr.add_argument(
        "--random-order", action="store_true",
        help="Parse the simulations in a random order.")
    psr.add_argument(
        "--skip-smoothed-features", action="store_true",
        help="Do not calculate EWMA and windowed features.")
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    exp_dir = args.exp_dir
    untar_dir = args.untar_dir
    out_dir = args.out_dir
    skip_smoothed = args.skip_smoothed_features

    # Find all simulations.
    pcaps = [
        (path.join(exp_dir, sim), untar_dir, out_dir, skip_smoothed)
        for sim in sorted(os.listdir(exp_dir)) if sim.endswith(".tar.gz")]
    if args.random_order:
        # Set the random seed so that multiple instances of this
        # script see the same random order.
        utils.set_rand_seed()
        random.shuffle(pcaps)

    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    if defaults.SYNC:
        for pcap in pcaps:
            parse_pcap(*pcap)
    else:
        # By default, use all available cores.
        with multiprocessing.Pool() as pol:
            pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
