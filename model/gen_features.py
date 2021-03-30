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
import time

import json
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
    """ Parses an experiment. Returns the smallest safe window size. """
    print(f"Parsing: {exp_flp}")
    if exp.tot_flws == 0:
        print(f"\tNo flows to analyze in: {exp_flp}")

    # Construct the output filepaths.
    out_flp = path.join(out_dir, f"{exp.name}.npz")
    # If the output file exists, then we do not need to parse this file.
    if path.exists(out_flp):
        print(f"\tAlready parsed: {exp_flp}")
        return -1

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

    # Keep track of the number of erroneous throughputs (i.e., higher than the
    # experiment bandwidth) for each window size.
    win_to_errors = {win: 0 for win in features.WINDOWS}

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
        } for win in features.WINDOWS}
        # Total number of packet losses up to the current received
        # packet.
        pkt_loss_total_estimate = 0
        # Loss rate estimation.
        prev_seq = None
        prev_payload_B = None
        highest_seq = None
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
            recv_seq = recv_pkt[0]
            recv_time_cur_us = recv_pkt[1]
            payload_B = recv_pkt[4]
            wirelen_B = recv_pkt[5]

            output[j][features.SEQ_FET] = recv_seq
            output[j][features.ARRIVAL_TIME_FET] = recv_time_cur_us

            # Count how many flows were active when this packet was captured.
            active_flws = sum(
                1 for first_time_us, last_time_us in flws_time_bounds
                if first_time_us <= recv_time_cur_us <= last_time_us)
            assert active_flws > 0, \
                (f"Error: No active flows detected for packet {j} of "
                 f"flow {flw_idx} in: {exp_flp}")

            output[j][features.ACTIVE_FLOWS_FET] = active_flws
            output[j][features.BW_FAIR_SHARE_FRAC_FET] = utils.safe_div(
                1, active_flws)
            output[j][features.BW_FAIR_SHARE_BPS_FET] = utils.safe_div(
                exp.bw_bps, active_flws)

            if j > 0:
                prev_min_rtt_us = output[j - 1][features.MIN_RTT_FET]
                prev_rtt_estimate_us = output[j - 1][features.RTT_FET]
                if recv_seq in retrans_pkts:
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
            output[j][features.INV_INTERARR_TIME_FET] = utils.safe_mul(
                8 * 1e6 * wirelen_B,
                utils.safe_div(1, interarr_time_us))

            output[j][features.PAYLOAD_FET] = payload_B
            output[j][features.WIRELEN_FET] = wirelen_B
            output[j][features.TOTAL_SO_FAR_FET] = (
                (output[j - 1][features.TOTAL_SO_FAR_FET]
                 if j > 0 else 0) + wirelen_B)

            output[j][features.RTT_FET] = rtt_estimate_us
            output[j][features.MIN_RTT_FET] = min_rtt_us
            output[j][features.RTT_RATIO_FET] = utils.safe_div(
                rtt_estimate_us, min_rtt_us)

            # Receiver-side loss rate estimation. Estimate the number of lost
            # packets since the last packet. Do not try anything complex or
            # prone to edge cases. Consider only the simple case where the last
            # packet and current packet are in order and not retransmissions.
            pkt_loss_cur_estimate = (
                -1 if (
                    prev_seq is None or
                    prev_seq == -1 or
                    prev_payload_B is None or
                    prev_payload_B <= 0 or
                    payload_B <= 0 or
                    highest_seq is None or
                    # The last packet was a retransmission.
                    highest_seq != prev_seq or
                    # The current packet is a retransmission.
                    prev_seq + prev_payload_B > recv_seq)
                else round((recv_seq - prev_payload_B - prev_seq) / payload_B))

            if pkt_loss_cur_estimate != -1:
                pkt_loss_total_estimate += pkt_loss_cur_estimate
            loss_rate_cur = utils.safe_div(
                pkt_loss_cur_estimate,
                utils.safe_add(pkt_loss_cur_estimate, 1))

            output[j][features.PACKETS_LOST_FET] = pkt_loss_cur_estimate
            output[j][features.LOSS_RATE_FET] = loss_rate_cur

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(
                    features.EWMAS, features.ALPHAS):
                if skip_smoothed:
                    continue

                metric = features.make_ewma_metric(metric, alpha)
                if metric.startswith(features.INTERARR_TIME_FET):
                    new = interarr_time_us
                elif metric.startswith(features.INV_INTERARR_TIME_FET):
                    # Do not use the interarrival time EWMA to calculate the
                    # inverse interarrival time. Instead, use the true inverse
                    # interarrival time so that the value used to update the
                    # inverse interarrival time EWMA is not "EWMA-ified" twice.
                    new = output[j][features.INV_INTERARR_TIME_FET]
                elif metric.startswith(features.RTT_FET):
                    new = rtt_estimate_us
                elif metric.startswith(features.RTT_RATIO_FET):
                    new = rtt_estimate_ratio
                elif metric.startswith(features.LOSS_RATE_FET):
                    new = loss_rate_cur
                elif metric.startswith(features.MATHIS_TPUT_FET):
                    # tput = (MSS / RTT) * (C / sqrt(p))
                    new = utils.safe_mul(
                        utils.safe_div(
                            utils.safe_mul(8, output[j][features.PAYLOAD_FET]),
                            utils.safe_div(output[j][features.RTT_FET], 1e6)),
                        utils.safe_div(
                            MATHIS_C,
                            utils.safe_sqrt(loss_rate_cur)))
                else:
                    raise Exception(f"Unknown EWMA metric: {metric}")
                # Update the EWMA. If this is the first value, then use 0 are
                # the old value.
                output[j][metric] = utils.safe_update_ewma(
                    -1 if j == 0 else output[j - 1][metric], new, alpha)

            # If we cannot estimate the min RTT, then we cannot compute any
            # windowed metrics.
            if min_rtt_us != -1:
                # Move the window start indices later in time. The min RTT
                # estimate will never increase, so we do not need to investigate
                # whether the start of the window moved earlier in time.
                for win in features.WINDOWS:
                    win_state[win]["window_start_idx"] = utils.find_bound(
                        output[features.ARRIVAL_TIME_FET],
                        target_us=recv_time_cur_us - (win * min_rtt_us),
                        min_idx=win_state[win]["window_start_idx"],
                        max_idx=j,
                        which="after")

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
                if metric.startswith(features.INTERARR_TIME_FET):
                    new = utils.safe_div(
                        utils.safe_sub(
                            recv_time_cur_us,
                            output[win_start_idx][features.ARRIVAL_TIME_FET]),
                        j - win_start_idx)
                elif metric.startswith(features.INV_INTERARR_TIME_FET):
                    new = utils.safe_mul(
                        8 * 1e6 * wirelen_B,
                        utils.safe_div(
                            1,
                            output[j][features.make_win_metric(
                                features.INTERARR_TIME_FET, win)]))
                elif metric.startswith(features.TPUT_FET):
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
                    if new != -1 and new > exp.bw_bps:
                        win_to_errors[win] += 1
                        # print(
                        #     f"Warning: Throughput of {new / 1e6:.2f} Mbps is "
                        #     "higher than experiment bandwidth of "
                        #     f"{exp.bw_Mbps:.2f} Mbps for window size {win} for "
                        #     f"packet {j} of flow {flw_idx} in: {exp_flp}")
                elif metric.startswith(features.TPUT_SHARE_FRAC_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.TOTAL_TPUT_FET):
                    # This is calcualted at the end.
                    continue
                elif metric.startswith(features.TPUT_FAIR_SHARE_BPS_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.TPUT_TO_FAIR_SHARE_RATIO_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.RTT_FET):
                    new = utils.safe_mean(
                        output[features.RTT_FET], win_start_idx, j)
                elif metric.startswith(features.RTT_RATIO_FET):
                    new = utils.safe_mean(
                        output[features.RTT_RATIO_FET], win_start_idx, j)
                elif metric.startswith(features.LOSS_EVENT_RATE_FET):
                    rtt_estimate_us = output[j][features.make_win_metric(
                        features.RTT_FET, win)]
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
                elif metric.startswith(features.SQRT_LOSS_EVENT_RATE_FET):
                    # Use the loss event rate to compute
                    # 1 / sqrt(loss event rate).
                    new = utils.safe_div(
                        1,
                        utils.safe_sqrt(output[j][
                            features.make_win_metric(
                                features.LOSS_EVENT_RATE_FET, win)]))
                elif metric.startswith(features.LOSS_RATE_FET):
                    win_losses = utils.safe_sum(
                        output[features.PACKETS_LOST_FET], win_start_idx + 1, j)
                    new = utils.safe_div(
                        win_losses, win_losses + (j - win_start_idx))
                elif metric.startswith(features.MATHIS_TPUT_FET):
                    # tput = (MSS / RTT) * (C / sqrt(p))
                    new = utils.safe_mul(
                        utils.safe_div(
                            utils.safe_mul(8, output[j][features.PAYLOAD_FET]),
                            utils.safe_div(output[j][features.RTT_FET], 1e6)),
                        utils.safe_div(
                            MATHIS_C,
                            utils.safe_sqrt(
                                output[j][features.make_win_metric(
                                    features.LOSS_EVENT_RATE_FET, win)])))
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new

            prev_seq = recv_seq
            prev_payload_B = payload_B
            highest_seq = (
                prev_seq if highest_seq is None else max(highest_seq, prev_seq))
            # In the event of sequence number wraparound, reset the sequence
            # number tracking.
            #
            # TODO: Test sequence number wraparound logic.
            if recv_seq + payload_B > 2**32:
                print(
                    "Warning: Sequence number wraparound detected for packet "
                    f"{j} of flow {flw} in: {exp_flp}")
                highest_seq = None
                prev_seq = None

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
        # Maps window the index of the packet at the start of that window.
        win_to_start_idx = {win: 0 for win in features.WINDOWS}

        # Merge the flow data into a unified timeline.
        combined = []
        for flw in flws:
            num_pkts = flw_results[flw].shape[0]
            merged = np.empty(
                (num_pkts,), dtype=[
                    (features.WIRELEN_FET, "int32"),
                    (features.MIN_RTT_FET, "int32"),
                    ("client port", "int32"),
                    ("server port", "int32"),
                    ("index", "int32")])
            merged[features.WIRELEN_FET] = flw_results[flw][features.WIRELEN_FET]
            merged[features.MIN_RTT_FET] = flw_results[flw][features.MIN_RTT_FET]
            merged["client port"].fill(flw[0])
            merged["server port"].fill(flw[1])
            merged["index"] = np.arange(num_pkts)
            combined.append(merged)
        zipped_arr_times, zipped_dat = utils.zip_timeseries(
            [flw_results[flw][features.ARRIVAL_TIME_FET] for flw in flws],
            combined)

        for j in range(zipped_arr_times.shape[0]):
            min_rtt_us = zipped_dat[j][features.MIN_RTT_FET]
            if min_rtt_us == -1:
                continue

            for win in features.WINDOWS:
                # The bounds should never go backwards, so start the
                # search at the current bound.
                win_to_start_idx[win] = utils.find_bound(
                        zipped_arr_times,
                        target_us=(
                            zipped_arr_times[j] -
                            (win * zipped_dat[j][features.MIN_RTT_FET])),
                        min_idx=win_to_start_idx[win],
                        max_idx=j,
                        which="after")
                # If the window's trailing edge caught up with its
                # leading edge, then skip this flow.
                if win_to_start_idx[win] >= j:
                    continue

                total_tput_bps = utils.safe_div(
                    utils.safe_mul(
                        # Accumulate the bytes received by this flow during this
                        # window. When calculating the average throughput, we
                        # must exclude the first packet in the window.
                        utils.safe_sum(
                            zipped_dat[features.WIRELEN_FET],
                            start_idx=win_to_start_idx[win] + 1,
                            end_idx=j),
                        8 * 1e6),
                    utils.safe_sub(
                        zipped_arr_times[j],
                        zipped_arr_times[win_to_start_idx[win]]))
                # Extract the flow to which this packet belongs, as well as its
                # index in its flow.
                flw = tuple(zipped_dat[j][["client port", "server port"]].tolist())
                index = zipped_dat[j]["index"]
                flw_results[flw][index][features.make_win_metric(
                    features.TOTAL_TPUT_FET, win)] = total_tput_bps
                # Use the total throughput and the number of active flows to
                # calculate the throughput fair share.
                flw_results[flw][index][features.make_win_metric(
                    features.TPUT_FAIR_SHARE_BPS_FET, win)] = utils.safe_div(
                        total_tput_bps,
                        flw_results[flw][index][features.ACTIVE_FLOWS_FET])
                # Divide the flow's throughput by the total throughput.
                tput_share = utils.safe_div(
                    flw_results[flw][index][
                        features.make_win_metric(features.TPUT_FET, win)],
                    total_tput_bps)
                flw_results[flw][index][features.make_win_metric(
                    features.TPUT_SHARE_FRAC_FET, win)] = tput_share
                # Calculate the ratio of tput share to bandwidth fair share.
                flw_results[flw][index][features.make_win_metric(
                    features.TPUT_TO_FAIR_SHARE_RATIO_FET, win)] = (
                        utils.safe_div(
                            tput_share,
                            flw_results[flw][index][
                                features.BW_FAIR_SHARE_FRAC_FET]))
                # Check if this throughput is erroneous.
                if total_tput_bps > exp.bw_bps:
                    win_to_errors[win] += 1

    print("\tWindow durations:")
    for win in features.WINDOWS:
        print(
            f"\t\t{win}:",
            win * min(
                [res[-1][features.MIN_RTT_FET]
                 for res in flw_results.values()]),
            "us")
    print("\tWindow errors:")
    for win in features.WINDOWS:
        print(f"\t\t{win}:", win_to_errors[win])
    smallest_safe_win = 0
    for win in sorted(features.WINDOWS):
        if win_to_errors[win] == 0:
            print(f"\tSmallest safe window size: {win}")
            smallest_safe_win = win
            break
    else:
        print(f"Warning: No safe window sizes in: {exp_flp}")

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

    return smallest_safe_win


def parse_exp(exp_flp, untar_dir, out_dir, skip_smoothed):
    """ Locks, untars, and parses an experiment. """
    exp = utils.Exp(exp_flp)
    with open_exp(exp, exp_flp, untar_dir, out_dir) as (locked, exp_dir):
        if locked:
            return parse_opened_exp(
                exp, exp_flp, exp_dir, out_dir, skip_smoothed)


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
        smallest_safe_wins = {parse_exp(*pcap) for pcap in pcaps}
    else:
        # By default, use all available cores.
        with multiprocessing.Pool(processes=args.parallel) as pol:
            smallest_safe_wins = set(pol.starmap(parse_exp, pcaps))
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")

    # Remove return values from experiments that were not parsed.
    smallest_safe_wins = [win for win in smallest_safe_wins if win != -1]
    if 0 in smallest_safe_wins:
        print("Some experiments had no safe window sizes.")
    print("Smallest globally-safe window size:", max(smallest_safe_wins))


if __name__ == "__main__":
    main()
