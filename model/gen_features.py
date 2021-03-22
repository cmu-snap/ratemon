#! /usr/bin/env python3
"""Parses the output of CloudLab experiments. """

import argparse
import collections
from contextlib import contextmanager
import itertools
import math
import multiprocessing
import subprocess
import os
from os import path
import random
import shutil
import sys
import time

import json
from matplotlib import pyplot as plt
import numpy as np

import cl_args
import defaults
import features
import utils


# Mathis model constant.
MATHIS_C = math.sqrt(3 / 2)


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


def loss_rate(loss_q, win_start_idx, pkt_loss_cur, recv_time_cur_us,
              recv_time_prev, win_size_us, pkt_idx):
    """ Calculates the loss rate over a window. """
    # If there were packet losses since the last received packet, then
    # add them to the loss queue.
    if pkt_loss_cur > 0 and pkt_idx > 0:
        # The average time between when packets should have arrived,
        # since we received the last packet.
        loss_interval = (
            (recv_time_cur_us - recv_time_prev) / (pkt_loss_cur + 1))
        # Look at each lost packet...
        for k in range(pkt_loss_cur):
            # Record the time at which this packet should have
            # arrived.
            loss_q.append(recv_time_prev + (k + 1) * loss_interval)

    # Discard old losses.
    while loss_q and (loss_q[0] < recv_time_cur_us - win_size_us):
        loss_q.popleft()

    # The loss rate is the number of losses in the window divided by
    # the total number of packets in the window.
    win_losses = len(loss_q)
    return (
        loss_q,
        ((win_losses / (pkt_idx + win_losses - win_start_idx))
         if pkt_idx - win_start_idx > 0 else 0))


def get_time_bounds(pkts, direction="data"):
    """
    Returns the earliest and latest times in a particular direction of each flow
    in a trace. pkts is in the format produced by utils.parse_packets().

    Returns a list of tuples of the form:
        ( time of first packet, time of last packet )
    """
    # [0] selects the data packets (as opposed to the ACKs). [:,1] selects
    # the column pertaining to arrival time. [[0, -1]] Selects the first and
    # last arrival times.
    dir_idx = 1 if direction == "ack" else 0
    return [
        tuple(pkts[flw][dir_idx][:,1][[0, -1]].tolist()) for flw in pkts.keys()]


@contextmanager
def open_exp(exp, exp_flp, untar_dir, out_dir):
    """
    Locks and untars an experiment. Cleans up the lock and untarred files
    automatically.
    """
    lock_flp = path.join(out_dir, f"{exp.name}.lock")
    exp_dir = path.join(untar_dir, exp.name)
    # Keep track of what we do.
    locked = False
    untarred = False
    try:
        # Grab the lock file for this experiment.
        if path.exists(lock_flp):
            print(f"\tParsing already in progress: {exp_flp}")
            yield False, exp_dir
        locked = True
        with open(lock_flp, "w"):
            pass

        # Create a temporary folder to untar experiments.
        if not path.exists(untar_dir):
            os.mkdir(untar_dir)
        # If this experiment has already been untarred, then delete the old
        # files.
        if path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        untarred = True
        subprocess.check_call(["tar", "-xf", exp_flp, "-C", untar_dir])
        yield True, exp_dir
    finally:
        # Remove an entity only if we created it.
        #
        # Remove untarred folder
        if locked and path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        # Remove lock file.
        if untarred and path.exists(lock_flp):
            os.remove(lock_flp)


def parse_opened_exp(exp, exp_flp, exp_dir, out_dir, skip_smoothed):
    """ Parses an experiment. """
    print(f"Parsing: {exp_flp}")
    if exp.tot_flws == 0:
        print(f"\tNo flows to analyze in: {exp_flp}")
    bw_bps = exp.bw_Mbps * 1e6
    bw_limit_bps = bw_bps * 1.01

    # Construct the output filepaths.
    out_flp = path.join(out_dir, f"{exp.name}.npz")
    # If the output file exists, then we do not need to parse this file.
    if path.exists(out_flp):
        print(f"\tAlready parsed: {exp_flp}")
        return

    # Determine the path to the bottleneck queue log file.
    toks = exp.name.split("-")
    q_log_flp = path.join(
        exp_dir,
        "-".join(toks[:-1]) + "-forward-bottleneckqueue-" + toks[-1] +
        ".log")
    q_log = None
    if path.exists(q_log_flp):
        q_log = list(enumerate(utils.parse_queue_log(q_log_flp)))

    # Determine flow src and dst ports.
    params_flp = path.join(exp_dir, f"{exp.name}.json")
    with open(params_flp, "r") as fil:
        params = json.load(fil)
    # List of tuples of the form: (client port, server port)
    flws = [(client_port, flw[4])
                 for flw in params["flowsets"]
                 for client_port in flw[3]]

    flw_to_pkts_client = utils.parse_packets(
        path.join(exp_dir, f"client-tcpdump-{exp.name}.pcap"), flws)
    flw_to_pkts_server = utils.parse_packets(
        path.join(exp_dir, f"server-tcpdump-{exp.name}.pcap"), flws)

    # Transform absolute times into relative times to make life easier.
    #
    # Determine the absolute earliest time observed in the experiment.
    earliest_time_us = min(
        first_time_us
        for bounds in [
            get_time_bounds(flw_to_pkts_client, direction="data"),
            get_time_bounds(flw_to_pkts_client, direction="ack"),
            get_time_bounds(flw_to_pkts_server, direction="data"),
            get_time_bounds(flw_to_pkts_server, direction="ack")]
        for first_time_us, _ in bounds)
    # Subtract the earliest time from all times.
    for flw in flws:
        flw_to_pkts_client[flw][0][:,1] -= earliest_time_us
        flw_to_pkts_client[flw][1][:,1] -= earliest_time_us
        flw_to_pkts_server[flw][0][:,1] -= earliest_time_us
        flw_to_pkts_server[flw][1][:,1] -= earliest_time_us

        assert (flw_to_pkts_client[flw][0][:,1] >= 0).all()
        assert (flw_to_pkts_client[flw][1][:,1] >= 0).all()
        assert (flw_to_pkts_server[flw][0][:,1] >= 0).all()
        assert (flw_to_pkts_server[flw][1][:,1] >= 0).all()

    flws_time_bounds = get_time_bounds(flw_to_pkts_server, direction="data")

    # Process PCAP files from senders and receivers.
    # The final output, with one entry per flow.
    flw_results = {}

    # Create the (super-complicated) dtype. The dtype combines each metric at
    # multiple granularities.
    dtype = (
        features.REGULAR +
        ([] if skip_smoothed else features.make_smoothed_features()))

    for flw_idx, flw in enumerate(flws):
        sent_pkts = flw_to_pkts_client[flw][0]
        recv_data_pkts, recv_ack_pkts = flw_to_pkts_server[flw]

        first_data_time_us = recv_data_pkts[0][1]

        # The final output. -1 implies that a value could not be calculated.
        output = np.full(len(recv_data_pkts), -1, dtype=dtype)

        # If this flow does not have any packets, then skip it.
        skip = False
        if sent_pkts.shape[0] == 0:
            skip = True
            print(
                f"Warning: No data packets sent for flow {flw_idx} in: "
                f"{exp_flp}")
        if recv_data_pkts.shape[0] == 0:
            skip = True
            print(
                f"Warning: No data packets received for flow {flw_idx} in: "
                f"{exp_flp}")
        if recv_ack_pkts.shape[0] == 0:
            skip = True
            print(
                f"Warning: No ACK packets sent for flow {flw_idx} in: "
                f"{exp_flp}")
        if skip:
            flw_results[flw] = output
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
            # For "loss rate estimated".
            "loss_queue_estimate": collections.deque()
        } for win in features.WINDOWS}
        # Total number of packet losses up to the current received
        # packet.
        # pkt_loss_total_true = 0
        pkt_loss_total_estimate = 0
        # Loss rate estimation.
        prev_pkt_seq = 0
        highest_seq = 0
        # RTT estimation.
        ack_idx = 0

        # Determine which packets are retransmissions. Ignore these packets when
        # estimating the RTT.
        #
        # All sequence numbers that have been sent.
        unique_pkts = set()
        # Sequence numbers that have been sent multiple times.
        retrans_pkts = set()
        for sent_pkt in sent_pkts:
            sent_pkt_seq = sent_pkt[0]
            (retrans_pkts
             if sent_pkt_seq in unique_pkts else unique_pkts).add(sent_pkt_seq)

        for j, recv_pkt in enumerate(recv_data_pkts):
            if j % 1000 == 0:
                print(
                    f"\tFlow {flw_idx + 1}/{exp.tot_flws}: "
                    f"{j}/{len(recv_data_pkts)} packets")
            # Regular metrics.
            recv_pkt_seq = recv_pkt[0]
            recv_time_cur_us = recv_pkt[1]

            output[j][features.SEQ_FET] = recv_pkt_seq
            output[j][features.ARRIVAL_TIME_FET] = recv_time_cur_us

            # Count how many flows were active when this packet was captured.
            active_flws = sum(
                1 for first_time_us, last_time_us in flws_time_bounds
                if first_time_us <= recv_time_cur_us <= last_time_us)
            assert active_flws > 0, \
                (f"Error: No active flows detected for packet {j} of "
                 f"flow {flw_idx} in: {exp_flp}")

            output[j][features.ACTIVE_FLOWS_FET] = active_flws
            output[j][features.BW_FAIR_SHARE_FET] = bw_bps / active_flws

            if j > 0:
                prev_min_rtt_us = output[j - 1][features.MIN_RTT_FET]
                prev_rtt_estimate_us = output[j - 1][features.RTT_ESTIMATE_FET]
                if recv_pkt_seq in retrans_pkts:
                    min_rtt_us = prev_min_rtt_us
                else:
                    # Receiver-side RTT estimation using the TCP timestamp
                    # option. Attempt to find a new RTT estimate. Move
                    # ack_idx to the first occurance of the timestamp
                    # option TSval corresponding to the current packet's
                    # TSecr.
                    tsval = recv_ack_pkts[ack_idx][2]
                    tsecr = recv_pkt[3]
                    ack_idx_old = ack_idx
                    while tsval != tsecr and ack_idx < len(recv_ack_pkts) - 1:
                        ack_idx += 1
                        tsval = recv_ack_pkts[ack_idx][2]
                    if tsval == tsecr:
                        # If we found a timestamp option match, then
                        # update the RTT estimate.
                        rtt_estimate_us = (
                            recv_time_cur_us - recv_ack_pkts[ack_idx][1])
                    else:
                        # Otherwise, use the previous RTT estimate and
                        # reset ack_idx to search again for the next
                        # packet.
                        rtt_estimate_us = prev_rtt_estimate_us
                        ack_idx = ack_idx_old
                    # Update the min RTT estimate.
                    min_rtt_us = utils.safe_min(
                        prev_min_rtt_us, rtt_estimate_us)

                # Compute the new RTT ratio.
                rtt_estimate_ratio = utils.safe_div(rtt_estimate_us, min_rtt_us)
                # Calculate the inter-arrival time.
                recv_time_prev = output[j - 1][features.ARRIVAL_TIME_FET]
                interarr_time_us = recv_time_cur_us - recv_time_prev
            else:
                rtt_estimate_us = -1
                rtt_estimate_ratio = -1
                min_rtt_us = -1
                recv_time_prev = -1
                interarr_time_us = -1

            output[j][features.INTERARR_TIME_FET] = interarr_time_us
            output[j][features.INV_INTERARR_TIME_FET] = utils.safe_div(
                1, interarr_time_us)

            payload_B = recv_pkt[4]
            wirelen_B = recv_pkt[5]
            output[j][features.PAYLOAD_FET] = payload_B
            output[j][features.WIRELEN_FET] = wirelen_B
            output[j][features.TOTAL_SO_FAR_FET] = (
                (output[j - 1][features.TOTAL_SO_FAR_FET]
                 if j > 0 else 0) + wirelen_B)

            output[j][features.RTT_ESTIMATE_FET] = rtt_estimate_us
            output[j][features.MIN_RTT_FET] = min_rtt_us
            output[j][features.RTT_RATIO_FET] = utils.safe_div(
                rtt_estimate_us, min_rtt_us)

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

            output[j][features.PACKETS_LOST_FET] = pkt_loss_cur_estimate

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(
                    features.EWMAS, features.ALPHAS):
                if skip_smoothed:
                    continue

                metric = features.make_ewma_metric(metric, alpha)
                if features.INTERARR_TIME_FET in metric:
                    new = interarr_time_us
                elif features.INV_INTERARR_TIME_FET in metric:
                    # Do not use the interarrival time EWMA to calculate the
                    # inverse interarrival time. Instead, use the true inverse
                    # interarrival time so that the value used to update the
                    # inverse interarrival time EWMA is not "EWMA-ified" twice.
                    new = output[j][features.INV_INTERARR_TIME_FET]
                elif features.RTT_ESTIMATE_FET in metric:
                    new = rtt_estimate_us
                elif features.RTT_RATIO_FET in metric:
                    new = rtt_estimate_ratio
                elif features.LOSS_RATE_FET in metric:
                    # See comment in case for "loss rate true".
                    new = pkt_loss_cur_estimate / (
                        pkt_loss_cur_estimate + 1)
                elif features.MATHIS_TPUT_FET in metric:
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
                        utils.safe_mul(
                            output[j][features.WIRELEN_FET],
                            utils.safe_div(
                                MATHIS_C,
                                utils.safe_div(
                                    utils.safe_mul(
                                        min_rtt_us,
                                        utils.safe_sqrt(loss_rate_estimate)),
                                    1e6))))
                else:
                    raise Exception(f"Unknown EWMA metric: {metric}")
                # Update the EWMA.
                output[j][metric] = utils.safe_update_ewma(
                    -1 if j == 0 else output[j - 1][metric], new, alpha)

            # If we cannot estimate the min RTT, then we cannot compute any
            # windowed metrics.
            if min_rtt_us != -1:
                # Move the window start indices later in time. The min RTT
                # estimate will never increase, so we do not need to investigate
                # whether the start of the window moved earlier in time.
                for win in features.WINDOWS:
                    win_size_us = win * min_rtt_us
                    # Keep trying until the window's trailing edge catches up
                    # with its leading edge.
                    while win_state[win]["window_start_idx"] < j:
                        # Look up the arrival time at the start of this window.
                        arr_time_us = output[
                            win_state[win]["window_start_idx"]][
                                features.ARRIVAL_TIME_FET]
                        # Skip this packet if the packet's arrival time is
                        # either unknown or beyond the bounds of the window.
                        if ((arr_time_us == -1) or
                                # We have made sure that both operands in the
                                # subtraction are known.
                                (recv_time_cur_us - arr_time_us > win_size_us)):
                            win_state[win]["window_start_idx"] += 1
                        else:
                            break

            # Windowed metrics.
            for (metric, _), win in itertools.product(
                    features.WINDOWED, features.WINDOWS):
                # If we cannot estimate the min RTT, then we cannot compute any
                # windowed metrics.
                if skip_smoothed or min_rtt_us == -1:
                    continue

                # Calculate windowed metrics only if an entire window has
                # elapsed since the start of the flow.
                win_size_us = win * min_rtt_us
                if recv_time_cur_us - first_data_time_us < win_size_us:
                    continue

                win_start_idx = win_state[win]["window_start_idx"]
                if win_start_idx == j:
                    continue

                metric = features.make_win_metric(metric, win)
                if features.INTERARR_TIME_FET in metric:
                    new = utils.safe_div(
                        utils.safe_sub(
                            recv_time_cur_us,
                            output[win_start_idx][features.ARRIVAL_TIME_FET]),
                        j - win_start_idx)
                elif features.INV_INTERARR_TIME_FET in metric:
                    new = utils.safe_div(
                        1,
                        output[j][features.make_win_metric(
                            features.INTERARR_TIME_FET, win)])
                elif features.TPUT_FET in metric:
                    # Treat the first packet in the window as the beginning of
                    # time. Calculate the average throughput over all but the
                    # first packet.
                    #
                    # Sum up the payloads of the packets in the window.
                    total_bytes = utils.safe_sum(
                        output[features.WIRELEN_FET],
                        start_idx=win_start_idx + 1, end_idx=j)
                    # Divide by the duration of the window.
                    start_time_us = (
                        output[win_start_idx][features.ARRIVAL_TIME_FET]
                        if win_start_idx >= 0 else -1)
                    end_time_us = output[j][features.ARRIVAL_TIME_FET]
                    new = utils.safe_div(
                        utils.safe_mul(total_bytes, 8),
                        utils.safe_div(
                            utils.safe_sub(end_time_us, start_time_us), 1e6))

                    # Report a warning if the throughput does not exceed the
                    # bandwidth.
                    if new != -1 and new > bw_limit_bps:
                        print(
                            f"Warning: Throughput of {new / 1e6:.2f} Mbps is "
                            "higher than experiment bandwidth of "
                            f"{exp.bw_Mbps} Mbps for flow {flw_idx} in: "
                            f"{exp_flp}")
                elif features.TOTAL_TPUT_FET in metric:
                    # This is calcualted at the end.
                    continue
                elif features.TPUT_SHARE_FET in metric:
                    # This is calculated at the end.
                    continue
                elif features.RTT_ESTIMATE_FET in metric:
                    new = utils.safe_mean(
                        output[features.make_ewma_metric(
                            features.RTT_ESTIMATE_FET, alpha=1.)],
                        win_start_idx, j)
                elif features.RTT_RATIO_FET in metric:
                    new = utils.safe_mean(
                        output[features.make_ewma_metric(
                            features.RTT_RATIO_FET, alpha=1.)],
                        win_start_idx, j)
                elif (features.LOSS_EVENT_RATE_FET in metric and
                      "1/sqrt" not in metric):
                    rtt_estimate_us = output[j][features.make_win_metric(
                        features.RTT_ESTIMATE_FET, win)]
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
                            loss_interval = (
                                (recv_time_cur_us - recv_time_prev) /
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
                elif features.SQRT_LOSS_EVENT_RATE_FET in metric:
                    # Use the loss event rate to compute
                    # 1 / sqrt(loss event rate).
                    new = utils.safe_div(
                        1,
                        utils.safe_sqrt(output[j][
                            features.make_win_metric(
                                features.LOSS_EVENT_RATE_FET, win)]))
                elif features.LOSS_RATE_FET in metric:
                    # We do not need to check whether recv_time_prev
                    # is -1 (unknown) because the windowed metrics
                    # skip the case where j == 0.
                    win_state[win]["loss_queue_estimate"], new = loss_rate(
                        win_state[win]["loss_queue_estimate"], win_start_idx,
                        pkt_loss_cur_estimate, recv_time_cur_us, recv_time_prev,
                        win_size_us, j)
                elif features.MATHIS_TPUT_FET in metric:
                    # Use the loss event rate to compute the Mathis
                    # model fair throughput.
                    loss_rate_estimate = (
                        pkt_loss_total_estimate / j if j > 0 else -1)
                    new = utils.safe_mul(
                        output[j][features.WIRELEN_FET],
                        utils.safe_div(
                            MATHIS_C,
                            utils.safe_div(
                                utils.safe_mul(
                                    min_rtt_us,
                                    utils.safe_sqrt(loss_rate_estimate)),
                                1e6)))
                elif features.MATHIS_LABEL_FET in metric:
                    # Use the current throughput and Mathis model fair
                    # throughput to compute the Mathis model label.
                    new = utils.safe_mathis_label(
                        output[j][features.make_win_metric(
                            features.TPUT_FET, win)],
                        output[j][features.make_win_metric(
                            features.MATHIS_TPUT_FET, win)])
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new

        # Calculate the number of retransmissions. Truncate the sent packets
        # at the last occurence of the last packet to be received.
        #
        # Get the sequence number of the last received packet.
        last_seq = output[-1][features.SEQ_FET]
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
                "Warning: Did not find when the last received packet "
                f"(seq: {last_seq}) was sent for flow {flw_idx} in: {exp_flp}")
        else:
            output[-1][features.RETRANS_RATE_FET] = (
                1 - (len(sent_seqs) / sent_idx))

        # Calculate the drop rate at the bottleneck queue.
        client_port = flw[0]
        deq_idx = None
        drop_rate = None
        if q_log is None:
            print(f"Warning: Unable to find bottleneck queue log: {q_log_flp}")
        else:
            # Find the dequeue log corresponding to the last packet that was
            # received.
            for record_idx, record in reversed(q_log):
                if (record[0] == "deq" and record[2] == client_port and
                        record[3] == last_seq):
                    deq_idx = record_idx
                    break
        if deq_idx is None:
            print(
                "Warning: Did not find when the last received packet "
                f"(seq: {last_seq}) was dequeued for flow {flw_idx} in: "
                f"{exp_flp}")
        else:
            # Find the most recent stats log before the last received
            # packet was dequeued.
            for _, record in reversed(q_log[:deq_idx]):
                if record[0] == "stats" and record[1] == client_port:
                    drop_rate = record[4] / (record[2] + record[4])
                    break
        if drop_rate is None:
            print(
                "Warning: Did not calculate the drop rate at the bottleneck "
                f"queue for flow {flw_idx} in: {exp_flp}")
        else:
            output[-1][features.DROP_RATE_FET] = drop_rate

        # Make sure that all output rows were used.
        used_rows = np.sum(output[features.ARRIVAL_TIME_FET] != -1)
        total_rows = output.shape[0]
        assert used_rows == total_rows, \
            (f"Error: Used only {used_rows} of {total_rows} rows for flow "
             f"{flw_idx} in: {exp_flp}")

        flw_results[flw] = output

    # Save memory by explicitly deleting the sent and received packets
    # after they have been parsed. This happens outside of the above
    # for-loop because only the last iteration's packets are not
    # automatically cleaned up by now (they go out of scope when the
    # *_pkts variables are overwritten by the next loop).
    del sent_pkts
    del recv_data_pkts
    del recv_ack_pkts

    if not skip_smoothed:
        # Maps flows to the index of the next packet to process in that flow.
        flw_to_idx = {flw: 0 for flw in flws}
        # Maps window to flow to the index of the packet at the start of that
        # window in that flow.
        win_to_flw_to_start_idx = {
            win: {flw: 0 for flw in flws}
            for win in features.WINDOWS}
        # From a dictionary mapping flows to packet indices, returns a list of
        # flows that still have packets remaining.
        get_remaining_flws = lambda flw_to_idx: [
            flw for flw in flws
            if flw_to_idx[flw] < flw_results[flw].shape[0]]

        # Keep track of the window sizes that results in erroneous throughput
        # calculations (i.e., higher than the experiment bandwidth).
        bad_wins = set()

        remaining_flws = get_remaining_flws(flw_to_idx)
        while len(remaining_flws) > 0:
            flw_to_arr_time_us = {
                flw: flw_results[
                    flw][flw_to_idx[flw]][features.ARRIVAL_TIME_FET]
                for flw in remaining_flws}
            # Pick the earliest packet that we have not processed yet.
            cur_flw = min(flw_to_arr_time_us, key=flw_to_arr_time_us.get)
            end_time_us = flw_to_arr_time_us[cur_flw]
            min_rtt_us = flw_results[
                cur_flw][flw_to_idx[cur_flw]][features.MIN_RTT_FET]

            # If we do not know the current packet's arrival time or min RTT,
            # then we cannot compute any windowed metrics.
            if end_time_us != -1 and min_rtt_us != -1:
                for win in features.WINDOWS:
                    win_size_us = win * min_rtt_us
                    first_flw = None
                    start_time_us = -1
                    total_bytes = 0
                    bytes_used = []
                    # Consider flows that are still covered by the window only.
                    for flw in get_remaining_flws(win_to_flw_to_start_idx[win]):
                        # Maybe move the start of this window in this flow to be
                        # later. Keep trying until the window's trailing edge
                        # catches up with its leading edge.
                        while (win_to_flw_to_start_idx[win][flw] <
                               flw_to_idx[flw]):
                            # Look up the arrival time at the start of this
                            # window in this flow.
                            arr_time_us = flw_results[flw][
                                win_to_flw_to_start_idx[win][flw]][
                                    features.ARRIVAL_TIME_FET]
                            # Skip this packet if the packet's arrival time is
                            # either unknown or beyond the bounds of the window.
                            if ((arr_time_us == -1) or
                                    # We have made sure that both operands in
                                    # the subtraction are known.
                                    (end_time_us - arr_time_us > win_size_us)):
                                win_to_flw_to_start_idx[win][flw] += 1
                            else:
                                break

                        # If the window's trailing edge caught up with its
                        # leading edge, then skip this flow.
                        if (win_to_flw_to_start_idx[win][flw] >=
                            flw_to_idx[flw]):
                            continue

                        # Find the earliest time at which this window starts
                        # across all of the flows.
                        proposed_start_time_us = flw_results[
                            flw][win_to_flw_to_start_idx[win][flw]][
                                features.ARRIVAL_TIME_FET]
                        assert proposed_start_time_us != -1, \
                            "Error: Arrival time is unknown."

                        # Determine the flow to which the earliest packet in the
                        # window belongs.
                        if (start_time_us == -1 or
                                proposed_start_time_us < start_time_us):
                            start_time_us = proposed_start_time_us
                            first_flw = flw

                        bytes_used.extend(
                            flw_results[flw][features.WIRELEN_FET][
                                win_to_flw_to_start_idx[win][flw]:
                                flw_to_idx[flw] + 1].tolist())

                        # Accumulate the bytes received by this flow during this
                        # window.
                        total_bytes = utils.safe_add(
                            total_bytes,
                            utils.safe_sum(
                                flw_results[flw][features.WIRELEN_FET],
                                start_idx=win_to_flw_to_start_idx[win][flw],
                                end_idx=flw_to_idx[flw]))

                    # This window does not cover any flows.
                    if first_flw is None or start_time_us == -1:
                        continue
                    # When calculating the average throughput, we must exclude
                    # the first packet in the window.
                    first_pkt_size = flw_results[
                        first_flw][win_to_flw_to_start_idx[win][first_flw]][
                            features.WIRELEN_FET]
                    total_bytes = utils.safe_sub(
                        total_bytes,
                        0 if first_pkt_size == -1 else first_pkt_size)

                    total_tput_bps = utils.safe_div(
                        utils.safe_mul(total_bytes, 8),
                        utils.safe_div(
                            utils.safe_sub(end_time_us, start_time_us),
                            1e6))

                    flw_results[cur_flw][flw_to_idx[cur_flw]][
                        features.make_win_metric(
                            features.TOTAL_TPUT_FET,
                            win)] = total_tput_bps
                    # Divide the flow's throughput by the total throughput.
                    flw_results[cur_flw][flw_to_idx[cur_flw]][
                        features.make_win_metric(
                            features.TPUT_SHARE_FET, win)] = (
                                utils.safe_div(
                                    flw_results[cur_flw][flw_to_idx[cur_flw]][
                                        features.make_win_metric(
                                            features.TPUT_FET, win)],
                                    total_tput_bps))

                    # Check if this throughput is erroneous.
                    if total_tput_bps > bw_limit_bps:
                        bad_wins.add(win)

            # Move forward one packet.
            flw_to_idx[cur_flw] += 1
            remaining_flws = get_remaining_flws(flw_to_idx)

        for win in sorted(features.WINDOWS):
            if win not in bad_wins:
                print(f"Smallest safe window size: {win}")
                break
        print("Window durations:")
        for win in features.WINDOWS:
            print(
                f"\t{win}",
                win * min(
                    [res[-1][features.MIN_RTT_FET]
                     for res in flw_results.values()]),
                "us")

        # for win in features.WINDOWS:
        #     flw_strs = [str(flw) for flw in flws]

        #     tput_fet = features.make_win_metric(features.TPUT_FET, win)
        #     flw_to_known_idxs = {
        #         flw: (flw_results[flw][tput_fet] != -1).nonzero()
        #         for flw in flws}
        #     xs = [
        #         flw_results[flw][flw_to_known_idxs[flw]][
        #             features.ARRIVAL_TIME_FET] / 1e6
        #         for flw in flws]
        #     ys = [
        #         flw_results[flw][flw_to_known_idxs[flw]][tput_fet] / 1e6
        #         for flw in flws]
        #     for xs_, ys_ in zip(xs, ys):
        #         plt.plot(xs_, ys_, ".", markersize=0.5)

        #     total_tput_fet = features.make_win_metric(
        #         features.TOTAL_TPUT_FET, win)
        #     flw_to_known_idxs = {
        #         flw: (flw_results[flw][total_tput_fet] != -1).nonzero()
        #         for flw in flws}
        #     xs = [
        #         flw_results[flw][flw_to_known_idxs[flw]][
        #             features.ARRIVAL_TIME_FET] / 1e6
        #         for flw in flws]
        #     ys = [
        #         flw_results[flw][flw_to_known_idxs[flw]][total_tput_fet] / 1e6
        #         for flw in flws]
        #     plt.plot(*utils.zip_timeseries(xs, ys))

        #     plt.legend(flw_strs + ["total"], loc="upper left", fontsize=5)
        #     plt.title(tput_fet)
        #     plt.xlabel("time (s)")
        #     plt.ylabel("throughput (Mb/s)")
        #     plt.xlim(left=0)
        #     plt.ylim(bottom=0, top=exp.bw_Mbps * 1.1)
        #     plt.tight_layout()
        #     plt.savefig(path.join(out_dir, f"tput_minRtt{win}.pdf"))
        #     plt.close()

    # Determine if there are any NaNs or Infs in the results. For the results
    # for each flow, look through all features (columns) and make a note of the
    # features that bad values. Flatten these lists of feature names, using a
    # set comprehension to remove duplicates.
    bad_fets = {
        fet for flw_dat in flw_results.values()
        for fet in flw_dat.dtype.names if not np.isfinite(flw_dat[fet]).all()}
    if bad_fets:
        print(
            f"\tExperiment {exp_flp} has NaNs of Infs in features: {bad_fets}")

    # Save the results.
    if path.exists(out_flp):
        print(f"\tOutput already exists: {out_flp}")
    else:
        print(f"\tSaving: {out_flp}")
        np.savez_compressed(
            out_flp,
            **{str(k + 1): v
               for k, v in enumerate(flw_results[flw] for flw in flws)})


def parse_exp(exp_flp, untar_dir, out_dir, skip_smoothed):
    """ Locks, untars, and parses an experiment. """
    exp = utils.Exp(exp_flp)
    with open_exp(exp, exp_flp, untar_dir, out_dir) as (locked, exp_dir):
        if locked:
            parse_opened_exp(exp, exp_flp, exp_dir, out_dir, skip_smoothed)


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
        help="Parse experiments in a random order.")
    psr.add_argument(
        "--skip-smoothed-features", action="store_true",
        help="Do not calculate EWMA and windowed features.")
    psr.add_argument(
        "--parallel", default=multiprocessing.cpu_count(),
        help="The number of files to parse in parallel.", type=int)
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    exp_dir = args.exp_dir
    untar_dir = args.untar_dir
    out_dir = args.out_dir
    skip_smoothed = args.skip_smoothed_features

    # Find all experiments.
    pcaps = [
        (path.join(exp_dir, exp), untar_dir, out_dir, skip_smoothed)
        for exp in sorted(os.listdir(exp_dir)) if exp.endswith(".tar.gz")]
    if args.random_order:
        random.shuffle(pcaps)

    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    if defaults.SYNC:
        for pcap in pcaps:
            parse_exp(*pcap)
    else:
        # By default, use all available cores.
        with multiprocessing.Pool(processes=args.parallel) as pol:
            pol.starmap(parse_exp, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
