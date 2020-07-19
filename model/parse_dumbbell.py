#! /usr/bin/env python3
"""Parses the pcap file of dumbbell topology. """

import argparse
import collections
import itertools
import math
import multiprocessing
import os
from os import path
import random
import time

import numpy as np

import utils


# Whether to parse PCAP files synchronously or in parallel.
SYNC = False
# Assemble the output dtype.
#
# These metrics do not change.
REGULAR = [
    ("seq", "int32"),
    ("arrival time us", "int32"),
    ("interarrival time us", "int32"),
    ("RTT estimated", "int32"),
    ("RTT true", "int32"),
    ("RTT ratio true", "float64")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("interarrival time us", "float64"),
    ("throughput p/s", "float64"),
    ("RTT ratio estimated", "float64"),
    ("loss rate true", "float64"),
    ("loss rate estimated", "float64"),
    ("queue occupancy", "float64"),
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
    ("average RTT ratio estimated", "float64"),
    ("loss event rate", "float64"),
    ("1/sqrt loss event rate", "float64"),
    ("loss rate true", "float64"),
    ("loss rate estimated", "float64"),
    ("queue occupancy", "float64"),
    ("mathis model throughput p/s", "float64"),
    # -1 no applicable (no loss yet), 0 lower than or equal to fair
    # throughput, 1 higher. This is not a windowed metric itself, but
    # is based on the "mathis model throughput p/s" metric.
    ("mathis model label", "int32")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 10 for i in range(1, 11)]
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
    # The "queue occupancy" metric is over the estimated RTT instead
    # of the minimum RTT.
    return (f"{metric}-windowed-"
            f"{'' if metric == 'queue occupancy' else 'min'}Rtt{win}")


# The final dtype combines each metric at multiple granularities.
DTYPE = (REGULAR +
         [(make_ewma_metric(metric, alpha), typ)
          for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
         [(make_win_metric(metric, win), typ)
          for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)])


def update_ewma(prev_ewma, new_val, alpha):
    """ Update an exponentially weighted moving average. """
    return alpha * new_val + (1 - alpha) * prev_ewma


def make_interval_weight(num_intervals):
    return [
        1 if i < num_intervals / 2 else 2 * (i - 1) / (i + 2)
        for i in range(num_intervals)]


def compute_weighted_average(curr_event_size, loss_event_intervals,
                             loss_interval_weights):
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


def parse_pcap(sim_dir, out_dir):
    """Parse a PCAP file.
    Writes a npz file that contains the sequence number, RTT ratio,
    interarrival time, loss rate, and queue occupency percentage in
    the last n RTT (default set to be 2 RTT).
    """
    print(f"Parsing: {sim_dir}")
    sim = utils.Sim(sim_dir)
    assert sim.unfair_flws > 0, f"No unfair flows to analyze: {sim_dir}"

    # Construct the output filepaths.
    out_flp = path.join(out_dir, f"{sim.name}.npz")
    # If the output file exists, then we do not need to parse this file.
    if path.exists(out_flp):
        print(f"    Already parsed: {sim_dir}")
        return

    # Process PCAP files from unfair senders and receivers.
    #
    # The final output, with one entry per unfair flow.
    unfair_flws = []
    for unfair_idx in range(sim.unfair_flws):
        # We will estimate the min RTT using the first TCP timestamp
        # option echo.
        min_rtt_us = -1
        min_rtt_s = -1

        # Packet lists are of tuples of the form:
        #     (seq, sender, timestamp us, timestamp option)
        sent_pkts = utils.parse_packets(
            path.join(sim_dir, f"{sim.name}-{unfair_idx + 2}-0.pcap"),
            sim.payload_B, direction="data")
        recv_pcap_flp = path.join(
            sim_dir,
            (f"{sim.name}-{unfair_idx + 2 + sim.unfair_flws + sim.other_flws}-0"
             ".pcap"))
        recv_pkts = utils.parse_packets(
            recv_pcap_flp, sim.payload_B, direction="data")
        # Ack packets for RTT calculation
        ack_pkts = utils.parse_packets(
            recv_pcap_flp, sim.payload_B, direction="ack")

        # State that the windowed metrics need to track across packets.
        win_state = {win: {
            # The index at which this window starts.
            "window_start_idx": 0,
            # The "loss event rate".
            "loss_interval_weights": make_interval_weight(win),
            "loss_event_intervals": collections.deque(),
            "current_loss_event_start_idx": 0,
            "current_loss_event_start_time": 0,
            # For "loss rate true".
            "loss_queue_true": collections.deque(),
            # For "loss rate estimated".
            "loss_queue_estimated": collections.deque()
        } for win in WINDOWS}

        # Final output.
        output = np.zeros(len(recv_pkts), dtype=DTYPE)
        # Total number of packet losses up to the current received
        # packet.
        pkt_loss_total_true = 0
        pkt_loss_total_estimated = 0
        # Loss rate estimation.
        prev_pkt_seq = 0
        highest_seq = 0
        # RTT estimation.
        curr_ts_option = 0
        ack_ts_idx = 0
        # The running RTT estimate. If we cannot determine an RTT
        # estimate for a new packet, then the value from the previous
        # packet is used.
        rtt_estimate_us = 0

        for j, recv_pkt in enumerate(recv_pkts):
            # Regular metrics.
            recv_pkt_seq = recv_pkt[0]
            output[j]["seq"] = recv_pkt_seq
            recv_time_cur = recv_pkt[2]
            output[j]["arrival time us"] = recv_time_cur
            if j > 0:
                # The time at which the previous packet was
                # received.
                recv_time_prev = recv_pkts[j - 1][2]
                interarr_time_us = recv_time_cur - recv_time_prev
            else:
                recv_time_prev = 0
                interarr_time_us = 0
            output[j]["interarrival time us"] = interarr_time_us

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
                sent_pkt_seq = sent_pkts[j + pkt_loss_total_true][0]
            # Calculate how many packets were lost since receiving the
            # last packet.
            pkt_loss_cur_true = pkt_loss_total_true - pkt_loss_total_true_prev

            # Calculate the true RTT ratio. Since this will not be
            # used in practice, we can calculate the min RTT using the
            # simulation's parameters.
            one_way_us = sim.btl_delay_us + 2 * sim.edge_delays[unfair_idx]
            output[j]["RTT ratio true"] = (
                # Look up the send time of this packet to calculate
                # the true sender-receiver delay.
                (recv_time_cur - sent_pkts[j + pkt_loss_total_true][2] +
                 # Assume that, on the reverse path, packets will
                 # experience no queuing delay.
                 one_way_us) /
                # Compare to the minimum RTT.
                (2 * one_way_us))

            # Receiver-side loss rate estimation. Estimate the losses
            # since the last packet.
            pkt_loss_cur_estimated = math.ceil(
                0 if recv_pkt_seq == prev_pkt_seq + sim.payload_B
                else (
                    ((recv_pkt_seq - highest_seq - sim.payload_B) /
                     sim.payload_B)
                    if recv_pkt_seq > highest_seq + sim.payload_B
                    else (
                        1
                        if (recv_pkt_seq < prev_pkt_seq and
                            prev_pkt_seq != highest_seq)
                        else 0)))

            pkt_loss_total_estimated += pkt_loss_cur_estimated
            prev_pkt_seq = recv_pkt_seq
            highest_seq = max(highest_seq, prev_pkt_seq)

            # Receiver-side RTT estimation using TCP timestamp option.
            recv_ts_option = recv_pkt[3]
            if recv_ts_option != curr_ts_option:
                curr_ts_option = recv_ts_option
                # Move ack_ts_idx to the first occurance of the timestamp option
                while ack_pkts[ack_ts_idx][3] != recv_ts_option:
                    ack_ts_idx += 1
                rtt_estimate_us = recv_time_cur - ack_pkts[ack_ts_idx][2]
            if min_rtt_us == -1:
                min_rtt_us = rtt_estimate_us
                min_rtt_s = min_rtt_us / 1e6

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                metric = make_ewma_metric(metric, alpha)
                if j > 0:
                    if "interarrival time us" in metric:
                        new = interarr_time_us
                    elif "throughput p/s" in metric:
                        interarr_time_us = output[j][make_ewma_metric(
                            "interarrival time us", alpha)]
                        # Divide by 1e6 to convert from microseconds
                        # to seconds.
                        new = (0 if interarr_time_us == 0
                               else 1 / (interarr_time_us / 1e6))
                    elif "RTT ratio estimated" in metric:
                        new = rtt_estimate_us
                    elif "loss rate true" in metric:
                        # Divide the pkt_loss_cur_true by
                        # (pkt_loss_cur_true + 1) because over the course of
                        # sending (pkt_loss_cur_true + 1) packets, one got
                        # through and pkt_loss_cur_true were lost.
                        new = pkt_loss_cur_true / (pkt_loss_cur_true + 1)
                    elif "loss rate estimated" in metric:
                        # See comment in case for "loss rate true".
                        new = pkt_loss_cur_estimated / (
                            pkt_loss_cur_estimated + 1)
                    elif "queue occupancy" in metric:
                        # Queue occupancy is calculated using the
                        # router logs, below.
                        continue
                    elif "mathis model throughput p/s" in metric:
                        # Use the estimated loss rate to compute the
                        # Mathis model fair throughput.
                        loss_rate_estimated = output[
                            j][make_ewma_metric("loss rate estimated", alpha)]
                        # If we have not been able to estimate the min
                        # RTT yet, then we cannot compute the Mathis
                        # model fair throughput.
                        new = (
                            0 if (min_rtt_s == -1 or
                                  min_rtt_s == 0 or
                                  loss_rate_estimated <= 0)
                            else (MATHIS_C /
                                  (min_rtt_s * math.sqrt(loss_rate_estimated))))
                    elif "mathis model label" in metric:
                        # Use the current throughput and the Mathis
                        # model fair throughput to compute the Mathis
                        # model label.
                        output[j][metric] = utils.get_mathis_label(
                            output[j][make_ewma_metric(
                                "throughput p/s", alpha)],
                            output[j][make_ewma_metric(
                                "mathis model throughput p/s", alpha)])
                        # Continue because the value of this metric is
                        # not an EWMA.
                        continue
                    else:
                        raise Exception(f"Unknown EWMA metric: {metric}")
                    new_ewma = update_ewma(output[j - 1][metric], new, alpha)
                else:
                    new_ewma = 0
                output[j][metric] = new_ewma

            # Windowed metrics.
            for (metric, _), win in itertools.product(WINDOWED, WINDOWS):
                metric = make_win_metric(metric, win)
                # If we have not been able to estimate the min RTT
                # yet, then we cannot compute any of the windowed
                # metrics.
                if min_rtt_us == -1:
                    output[j][metric] = 0
                    continue
                win_size_us = win * min_rtt_us

                # Move the start of the window forward.
                while ((recv_time_cur -
                        recv_pkts[win_state[win]["window_start_idx"]][2]) >
                       win_size_us):
                    win_state[win]["window_start_idx"] += 1
                win_start_idx = win_state[win]["window_start_idx"]

                if "average interarrival time us" in metric:
                    new = ((recv_time_cur - recv_pkts[j - win_start_idx][2]) /
                           (j - win_start_idx + 1))
                elif "average throughput p/s" in metric:
                    avg_interarr_time_us = output[j][
                        make_win_metric("average interarrival time us", win)]
                    # Divide by 1e6 to convert interarrival time to seconds.
                    new = (0 if avg_interarr_time_us == 0
                           else 1 / (avg_interarr_time_us / 1e6))
                elif "average RTT ratio estimated" in metric:
                    # For each packet in the window, extract the EWMA
                    # metric for RTT ratio with alpha = 1.0, which
                    # corresponds to an EWMA with no history.
                    new = (0 if j == win_start_idx else
                           np.mean(output[make_ewma_metric(
                               "RTT ratio estimated", alpha=1.)][
                                   win_start_idx:j + 1]))
                elif "loss event rate" in metric:
                    cur_start_idx = win_state[
                        win]["current_loss_event_start_idx"]
                    cur_start_time = win_state[
                        win]["current_loss_event_start_time"]

                    # If there was a loss since the last packet...
                    if pkt_loss_cur_estimated > 0:
                        # The index of the first packet in the current
                        # loss event.
                        new_start_idx = (j + pkt_loss_total_estimated -
                                         pkt_loss_cur_estimated)

                        # This is the first loss event.
                        if cur_start_idx == 0:
                            cur_start_idx = new_start_idx
                            cur_start_time = sent_pkts[cur_start_idx][2]
                            new = 1 / (pkt_loss_cur_estimated + 1)
                        else:
                            # This is not the first loss event. See if
                            # any of the newly-lost packets start a
                            # new loss event.
                            # The average time between when packets
                            # should have arrived, since we received
                            # the last packet.
                            loss_interval = ((recv_time_cur - recv_time_prev) /
                                             (pkt_loss_cur_estimated + 1))

                            # Look at each lost packet...
                            for k in range(pkt_loss_cur_estimated):
                                # Compute the approximate time at
                                # which the packet should have been
                                # received if it had not been lost.
                                loss_time = (
                                    recv_time_prev + (k + 1) * loss_interval)

                                # If the time of this loss is more
                                # than an RTT from the time of the
                                # start of the current loss event,
                                # then this is a new loss event.
                                if (loss_time - cur_start_time >=
                                        output[j]["RTT estimated"]):
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
                                (j + pkt_loss_total_estimated -
                                 cur_start_idx),
                                win_state[win]["loss_event_intervals"],
                                win_state[win]["loss_interval_weights"])
                    else:
                        # If there have been no losses since the
                        # last packet...
                        if pkt_loss_total_estimated > 0:
                            # ...and the total loss is nonzero, then
                            # increase the size of the current loss
                            # event.
                            new = compute_weighted_average(
                                (j + pkt_loss_total_estimated -
                                 cur_start_idx),
                                win_state[win]["loss_event_intervals"],
                                win_state[win]["loss_interval_weights"])
                        else:
                            # ...and there have never been any losses,
                            # then the loss event rate is 0.
                            new = 0
                    # Record the new values of the state variables.
                    win_state[
                        win]["current_loss_event_start_idx"] = cur_start_idx
                    win_state[
                        win]["current_loss_event_start_time"] = cur_start_time
                elif "1/sqrt loss event rate" in metric:
                    # Use the loss event rate to compute
                    # 1 / sqrt(loss event rate). Perform this calculation only
                    # if the packet's loss event rate is greater than 0.
                    loss_event_rate = output[
                        j][make_win_metric("loss event rate", win)]
                    new = (0 if loss_event_rate <= 0 else
                           1 / math.sqrt(loss_event_rate))
                elif "loss rate true" in metric:
                    win_state[win]["loss_queue_true"], new = loss_rate(
                        win_state[win]["loss_queue_true"], win_start_idx,
                        pkt_loss_cur_true, recv_time_cur, recv_time_prev,
                        win_size_us, j)
                elif "loss rate estimated" in metric:
                    win_state[win]["loss_queue_estimated"], new = loss_rate(
                        win_state[win]["loss_queue_estimated"], win_start_idx,
                        pkt_loss_cur_estimated, recv_time_cur, recv_time_prev,
                        win_size_us, j)
                elif "queue occupancy" in metric:
                    # Queue occupancy is calculated using the router
                    # logs, below.
                    continue
                elif "mathis model throughput p/s" in metric:
                    # Use the loss event rate to compute the Mathis
                    # model fair throughput.
                    loss_event_rate = output[j][make_win_metric("loss event rate", win)]
                    new = (0 if min_rtt_s == 0 or loss_event_rate <= 0 else
                           MATHIS_C / (min_rtt_s * math.sqrt(loss_event_rate)))
                elif "mathis model label" in metric:
                    # Use the current throughput and Mathis model
                    # fair throughput to compute the Mathis model
                    # label.
                    new = utils.get_mathis_label(
                        output[j][make_win_metric(
                            "average throughput p/s", win)],
                        output[j][make_win_metric(
                            "mathis model throughput p/s", win)])
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new
        unfair_flws.append(output)

    # Save memory by explicitly deleting the sent and received packets
    # after they have been parsed. This happens outside of the above
    # for-loop because only the last iteration's sent and received
    # packets are not automatically cleaned up by now (they go out of
    # scope when the sent_pkts and recv_pkts variables are overwritten
    # by the next loop).
    del sent_pkts
    del recv_pkts

    # Process pcap files from the bottleneck router to determine queue
    # occupency. Packet lists are of tuples of the form:
    #     (seq, sender, timestamp us, timestamp option)
    router_pkts = utils.parse_packets(
        path.join(sim_dir, f"{sim.name}-1-0.pcap"), sim.payload_B,
        direction="data")
    # State pertaining to each flow.
    flw_state = {
        flw: {
            # Index of the output array where the queue occupency
            # results should be appended.
            "output_idx": 0,
            # The number of other flows' packets that have arrived
            # since the last packet for this flow.
            "packets_since_last": 0,
            # The number of packets from this flow currently in the
            # window.
            "window_flow_packets": {win: 0 for win in WINDOWS}
        } for flw in range(sim.unfair_flws)}
    # The index of the first packet in the window, for every window
    # size.
    win_start_idxs = {win: 0 for win in WINDOWS}

    # Loop over all of the packets receiver by the bottleneck
    # router. Note that we process all flows at once.
    for j, router_pkt in enumerate(router_pkts):
        _, sender, curr_time, _ = router_pkt
        # Process only packets that are part of one of the unfair
        # flows. Discard packets that did not make it to the receiver
        # (e.g., at the end of the experiment).
        if (sender < sim.unfair_flws and
                flw_state[sender]["output_idx"] < unfair_flws[sender].shape[0]):
            # We cannot move this above the if-statement condition
            # because it is valid only if sender < sim.unfair_flws.
            output_idx = flw_state[sender]["output_idx"]

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                metric = make_ewma_metric(metric, alpha)
                if j > 0:
                    if "interarrival time us" in metric:
                        # The interarrival time is calculated using
                        # the sender and/or receiver logs, above.
                        continue
                    if "throughput p/s" in metric:
                        # The throughput is calculated using the
                        # sender and/or receiver logs, above.
                        continue
                    if "RTT ratio estimated" in metric:
                        # The RTT ratio is calculated using the sender
                        # and/or receiver logs, above.
                        continue
                    if "loss rate true" in metric:
                        # The true loss rate is calculated using the sender
                        # and/or receiver logs, above.
                        continue
                    if "loss rate estimated" in metric:
                        # The estiamted loss rate is calculated using
                        # the sender and/or receiver logs, above.
                        continue
                    if "queue occupancy" in metric:
                        # Extra safety check to avoid divide-by-zero
                        # errors.
                        assert j > 0, \
                            ("Cannot calculate queue occupancy EWMA for the "
                             "first packet.")
                        # The instanteneous queue occupancy is 1
                        # divided by the number of packets that have
                        # entered the queue since the last packet from
                        # the same flow. This is the fraction of
                        # packets added to the queue corresponding to
                        # this flow, over the time since when the
                        # flow's last packet arrived.
                        new = 1 / flw_state[sender]["packets_since_last"]
                    elif "mathis model throughput p/s" in metric:
                        # The Mathis model fair throughput is
                        # calculated using the sender and/or receiver
                        # logs, above.
                        continue
                    elif "mathis model label" in metric:
                        # The Mathis model label is calculated using
                        # the sender and/or receiver logs, above.
                        continue
                    else:
                        raise Exception(f"Unknown EWMA metric: {metric}")
                    new_ewma = update_ewma(
                        unfair_flws[sender][output_idx - 1][metric],
                        new, alpha)
                else:
                    new_ewma = 0
                unfair_flws[sender][output_idx][metric] = new_ewma

            # Windowed metrics.
            for (metric, _), win in itertools.product(WINDOWED, WINDOWS):
                metric = make_win_metric(metric, win)
                if "average interarrival time us" in metric:
                    # The average interarrival time is calculated
                    # using the sender and/or receiver logs, above.
                    continue
                if "average throughput p/s" in metric:
                    # The average throughput is calculated using the
                    # sender and/or receiver logs, above.
                    continue
                if "average RTT ratio estimated" in metric:
                    # The average RTT ratio is calculated using the
                    # sender and/or receiver logs, above.
                    continue
                if "loss event rate" in metric:
                    # The loss event rate is calcualted using the
                    # sender and/or receiver logs, above.
                    continue
                if "1/sqrt loss event rate" in metric:
                    # The reciprocal of the square root of the loss
                    # event rate is calculated using the sender and/or
                    # receiver logs, above.
                    continue
                if "loss rate true" in metric:
                    # The true loss rate is calculated using the
                    # sender and/or receiver logs, above.
                    continue
                if "loss rate estimated" in metric:
                    # The estimated loss rate is calcualted using the
                    # sender and/or reciever logs, above.
                    continue
                if "queue occupancy" in metric:
                    win_start_idx = win_start_idxs[win]
                    # By definition, the window now contains one more
                    # packet from this flow.
                    win_flw_pkts = (
                        flw_state[sender]["window_flow_packets"][win] + 1)

                    # The current length of the window.
                    win_cur_us = curr_time - router_pkts[win_start_idx][2]
                    # Calculate the target length of the window.
                    win_target_us = (
                        win * unfair_flws[sender][output_idx]["RTT estimated"])

                    # If the current window size is greater than the
                    # target window size, then shrink the window.
                    while win_cur_us > win_target_us:
                        # If the packet that will be removed from
                        # the window is from this flow, then we
                        # need to decrease our record of the
                        # number of this flow's packets in the
                        # window by one.
                        if router_pkts[win_start_idx][1] == sender:
                            win_flw_pkts -= 1
                        # Move the start of the window forward.
                        win_start_idx += 1
                        win_cur_us = curr_time - router_pkts[win_start_idx][2]

                    # If the current window size is smaller than the
                    # target window size, then grow the window.
                    while (win_start_idx > 0 and
                           win_cur_us < win_target_us):
                        # Move the start of the window backward.
                        win_start_idx -= 1
                        win_cur_us = curr_time - router_pkts[win_start_idx][2]
                        # If the new packet that was added to the
                        # window is from this flow, then we need
                        # to increase our record of the number of
                        # this flow's packets in the window by
                        # one.
                        if router_pkts[win_start_idx][1] == sender:
                            win_flw_pkts += 1

                    # The queue occupancy is the number of this flow's
                    # packets in the window divided by the total
                    # number of packets in the window.
                    new = win_flw_pkts / (j - win_start_idx + 1)
                    # Record the new values of the state variables.
                    win_start_idxs[win] = win_start_idx
                    flw_state[
                        sender]["window_flow_packets"][win] = win_flw_pkts
                elif "mathis model throughput p/s" in metric:
                    # The Mathis model fair throughput is calculated
                    # using the sender and/or receiver logs, above.
                    continue
                elif "mathis model label" in metric:
                    # The Mathis model label is calculated using the
                    # sender and/or receiver logs, above.
                    continue
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                unfair_flws[sender][output_idx][metric] = new
            flw_state[sender]["output_idx"] += 1
            # For the current packet's flow, the number of packets
            # since the last packet in this flow is now 1.
            flw_state[sender]["packets_since_last"] = 1
        # For each unfair flow except the current packet's flow,
        # increment the number of packets since the last packet from
        # that flow.
        for flw in range(sim.unfair_flws):
            if flw != sender:
                flw_state[flw]["packets_since_last"] += 1

    # Determine if there are any NaNs or Infs in the results. For the
    # results for each unfair flow, look through all features
    # (columns) and make a note of the features that bad
    # values. Flatten these lists of feature names, using a set
    # comprehension to remove duplicates.
    bad_fets = {
        fet for flw_dat in unfair_flws
        for fet in flw_dat.dtype.names if not np.isfinite(flw_dat[fet]).all()}
    if bad_fets:
        print(f"    Simulation {sim_dir} has NaNs of Infs in features: "
              f"{bad_fets}")

    # Save the results.
    if path.exists(out_flp):
        print(f"    Output already exists: {out_flp}")
    else:
        print(f"    Saving: {out_flp}")
        np.savez_compressed(
            out_flp, **{str(k + 1): v for k, v in enumerate(unfair_flws)})


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Parses the output of gen_training_data.py.")
    psr.add_argument(
        "--exp-dir",
        help=("The directory in which the experiment results are stored "
              "(required)."), required=True, type=str)
    psr.add_argument(
        "--out-dir",
        help="The directory in which to store output files (required).",
        required=True, type=str)
    psr.add_argument(
        "--random-order", action="store_true",
        help="Parse the simulations in a random order.")
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir

    # Find all simulations.
    pcaps = [(path.join(exp_dir, sim), out_dir)
             for sim in sorted(os.listdir(exp_dir))]
    if args.random_order:
        random.shuffle(pcaps)
    pcaps = pcaps[:1]

    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    if SYNC:
        for pcap in pcaps:
            parse_pcap(*pcap)
    else:
        with multiprocessing.Pool() as pol:
            pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
