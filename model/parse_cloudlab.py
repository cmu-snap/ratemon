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
import time

import numpy as np
import matplotlib.pyplot as plt
import shutil

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
    ("flow share percentage", "float64")
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


# The final dtype combines each metric at multiple granularities.
DTYPE = (REGULAR +
         [(make_ewma_metric(metric, alpha), typ)
          for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
         [(make_win_metric(metric, win), typ)
          for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)])


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


def parse_pcap(sim_dir, untar_dir, out_dir):
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

    # Create a temporary folder to untar experiments
    untar_dir = path.join(untar_dir, sim.name)
    if path.exists(untar_dir):
        # Delete the folder and then untar experiments
        shutil.rmtree(untar_dir)
    os.mkdir(untar_dir)
    subprocess.check_call(["tar", "-xf", sim_dir, "-C", untar_dir])

    # Process PCAP files from senders and receivers.
    #
    # The final output, with one entry per flow.
    flws = []

    # Only process the last flow (unfair flow)
    for flw_idx in range(tot_flws - 1, tot_flws):
        # Packet lists are of tuples of the form:
        #     (seq, sender, timestamp us, timestamp option)
        sent_pkts = utils.parse_packets(
            path.join(untar_dir, f"client-tcpdump-{sim.name}.pcap"), flw_idx)

        recv_flp = path.join(untar_dir, f"server-tcpdump-{sim.name}.pcap")
        recv_data_pkts = utils.parse_packets(
            recv_flp, flw_idx, direction="data")
        # Ack packets for RTT calculation
        recv_ack_pkts = utils.parse_packets(recv_flp, flw_idx, direction="ack")

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

        # The final output. -1 implies that a value was unable to be
        # calculated.
        output = np.empty(len(recv_data_pkts), dtype=DTYPE)
        output.fill(-1)
        # Total number of packet losses up to the current received
        # packet.
        pkt_loss_total_true = 0
        pkt_loss_total_estimate = 0
        # Loss rate estimation.
        prev_pkt_seq = 0
        highest_seq = 0
        # RTT estimation.
        ack_idx = 0
        # Update experiment start time
        start_time = recv_data_pkts[0][2]
        # Compute the end time of the first flow
        end_time = utils.parse_packets(recv_flp, 0, direction="data")[-1][2]
        end_idx = len(recv_data_pkts)

        for j, recv_pkt in enumerate(recv_data_pkts):
            if j % 1000 == 0:
                print(f"Flow {flw_idx}: {j}/{len(recv_data_pkts)} packets")
            # Regular metrics.
            recv_pkt_seq = recv_pkt[0]
            recv_time_cur = recv_pkt[2]

            if recv_time_cur > end_time:
                # Stop parsing when the first flow ends
                end_idx = j
                break

            output[j]["seq"] = recv_pkt_seq
            # Align arrival time to zero such that all features starts at 0s
            output[j]["arrival time us"] = recv_time_cur - start_time

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
                    rtt_estimate_us = output[
                        j - 1][make_ewma_metric("RTT estimate us", alpha=1.)]
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

            # Calculate the true packet loss rate. Count the number of
            # dropped packets by checking if the sequence numbers at
            # sender and receiver are the same. If not, the packet is
            # dropped, and the pkt_loss_total_true counter increases
            # by one to keep the index offset at sender
            sent_pkt_seq = sent_pkts[j + pkt_loss_total_true][0]
            pkt_loss_total_true_prev = pkt_loss_total_true
            while sent_pkt_seq != recv_pkt_seq:
                # Packet loss
                pkt_loss_total_true += 1
                if j + pkt_loss_total_true < len(sent_pkts):
                    sent_pkt_seq = sent_pkts[j + pkt_loss_total_true][0]
                else:
                    # Couldn't find the packet in sender's trace
                    # Reduce pkt_loss_total_true by one so that
                    # the offset is still correct
                    print(f"Couldn't find {recv_pkt_seq}")
                    pkt_loss_total_true = pkt_loss_total_true_prev - 1
                    break
            # Calculate how many packets were lost since receiving the
            # last packet.
            pkt_loss_cur_true = max(
                0, pkt_loss_total_true - pkt_loss_total_true_prev)

            # Receiver-side loss rate estimation. Estimate the losses
            # since the last packet.
            payload_B = sent_pkts[j + pkt_loss_total_true][4]
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

            # Calculate the true RTT and RTT ratio. Look up the send
            # time of this packet to calculate the true
            # sender-receiver delay. Assume that, on the reverse path,
            # packets will experience no queuing delay.
            one_way_us = sim.rtt_us / 2
            rtt_true_us = (
                recv_time_cur - sent_pkts[j + pkt_loss_total_true][2] +
                one_way_us)
            rtt_true_ratio = rtt_true_us / (2 * one_way_us)

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
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
        flws.append(output[:end_idx])

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

    ground_truth_throughput = "throughput p/s-ewma-alpha0.003"
    flow_share = "flow share percentage"
    total_throughput_p = sim.bw_Mbps * (1e6 / 8 / 1448)
    for flw_dat in flws:
        per_flow_throughput = flw_dat[ground_truth_throughput]
        flw_dat[flow_share] = per_flow_throughput / total_throughput_p

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
        help=(
            "The directory in which the untarped experiment results are stored "
            "(required)."),
        required=True, type=str)
    psr.add_argument(
        "--random-order", action="store_true",
        help="Parse the simulations in a random order.")
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    exp_dir = args.exp_dir
    untar_dir = args.untar_dir
    out_dir = args.out_dir

    # Find all simulations.
    pcaps = [
        (path.join(exp_dir, sim), untar_dir, out_dir)
        for sim in sorted(os.listdir(exp_dir))]
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
        with multiprocessing.Pool() as pol:
            pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
