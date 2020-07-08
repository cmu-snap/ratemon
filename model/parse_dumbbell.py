#! /usr/bin/env python3
"""Parses the pcap file of dumbbell topology. """

import argparse
import itertools
import multiprocessing
import os
from os import path
import time
from collections import deque
import math
from statistics import mean

import numpy as np

import utils


# Whether to parse PCAP files synchronously or in parallel.
SYNC = False

# Assemble the output dtype.
#
# These metrics do not change.
REGULAR = [
    ("seq", "int32"),
    ("arrival time", "int32"),
    ("inter-arrival time", "int32"),
    ("true RTT ratio", "float64"),
    ("loss event rate", "float64"),
    ("loss event rate sqrt", "float64"),
    ("mathis model throughput", "float64"),
    # -1 no applicable (no loss yet), 0 lower than fair throughput, 1 higher
    ("mathis model label", "int32")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("inter-arrival time ewma", "float64"),
    ("throughput ewma", "float64"),
    ("RTT ratio ewma", "float64"),
    ("loss rate ewma", "float64"),
    ("queue occupancy ewma", "float64")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    ("average inter-arrival time windowed", "float64"),
    ("average throughput windowed", "float64"),
    ("average RTT ratio windowed", "float64"),
    ("loss rate windowed", "float64"),
    ("queue occupancy windowed", "float64")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 10 for i in range(1, 11)]
# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(0, 11)]
# Number of RTTs for computing loss event rate
NUM_INTERVALS = 8

def make_ewma_metric(metric, alpha):
    """ Format the name of an EWMA metric. """
    return f"{metric}-alpha{alpha}"


def make_win_metric(metric, win):
    """ Format the name of a windowed metric. """
    return f"{metric}-minRtt{win}"


# The final dtype combines each metric at multiple granularities.
DTYPE = (REGULAR +
         [(make_ewma_metric(metric, alpha), typ)
          for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
         [(make_win_metric(metric, win), typ)
          for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)])


def update_ewma(prev_ewma, new_val, alpha):
    """ Update an exponentially weighted moving average. """
    return alpha * new_val + (1 - alpha) * prev_ewma

def make_interval_weight():
    weight = [1.0] * NUM_INTERVALS
    for i in range(NUM_INTERVALS):
        if i < NUM_INTERVALS / 2:
            weight[i] = 1.0
        else:
            weight[i] = 2 * (i - 1) / (1.0 * (i + 2))
    return weight

def compute_weighted_average(curr_event_size, loss_event_intervals,
                             loss_interval_weight):
    i_tot0 = curr_event_size
    i_tot1 = 0
    w_tot = 1.0
    for i in range(len(loss_event_intervals) - 1):
        i_tot0 += loss_event_intervals[i] * loss_interval_weight[i + 1]
        w_tot += loss_interval_weight[i + 1]

    for i in range(len(loss_event_intervals)):
        i_tot1 += loss_event_intervals[i] * loss_interval_weight[i]

    i_tot = max(i_tot0, i_tot1)
    i_mean = i_tot / w_tot
    return 1.0 / i_mean

def parse_pcap(sim_dir, out_dir):
    """Parse a PCAP file.
    Writes a npz file that contains the sequence number, RTT ratio,
    inter-arrival time, loss rate, and queue occupency percentage in
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

    # Process PCAP files from unfair senders and receivers

    unfair_flws = []
    min_rtts_us = []
    loss_interval_weight = make_interval_weight()

    for unfair_idx in range(sim.unfair_flws):
        one_way_us = sim.btl_delay_us + 2 * sim.edge_delays[unfair_idx] * 1.0
        min_rtt_us = one_way_us * 2
        min_rtts_us.append(min_rtt_us)

        min_rtt_s = min_rtt_us / 1e6

        # (Seq, timestamp)
        send_pkts = utils.parse_packets_endpoint(
            path.join(sim_dir, f"{sim.name}-{unfair_idx + 2}-0.pcap"),
            sim.payload_B)
        recv_pkts = utils.parse_packets_endpoint(
            path.join(
                sim_dir,
                (f"{sim.name}-"
                 f"{unfair_idx + 2 + sim.unfair_flws + sim.other_flws}-0"
                 ".pcap")),
            sim.payload_B)

        # State that the windowed metrics need to track across packets.
        windowed_state = {win: {
            "window_start": 0,
            "loss_queue": deque()
        } for win in WINDOWS}
        # Number of packet losses up to the current received packet.
        packet_loss = 0
        # Final output.
        output = np.empty(len(recv_pkts), dtype=DTYPE)
        # Loss event rate data
        loss_event_intervals = deque()
        curr_event_start_idx = 0
        curr_event_start_time = 0
        # Mathis Model loss queue
        mathis_loss_queue = deque()
        mathis_fair_throughput = 0.0
        mathis_8rtt_window_start = 0
        continuous_loss_interval = []

        # negative_gaps = 0
        # big_gaps = 0
        # big_gap_packets = 0
        # negative_gap_packets = 0
        # strange_gaps = 0
        # max_seq = 0
        # correct_gaps = 0
        # other = 0
        # retransmits = 0
        # for j, recv_pkt in enumerate(recv_pkts):
        #     cur_seq = recv_pkt[0]
        #     if cur_seq <= max_seq:
        #         retransmits += 1
        #     max_seq = max(max_seq, cur_seq)
        #     if j == 0:
        #         continue
        #     seq_diff = cur_seq - recv_pkts[j - 1][0]
        #     seq_diff_packets = abs(seq_diff / 1380)
        #     if seq_diff < 0:
        #         print(f"negative diff: {seq_diff}")
        #         negative_gaps += 1
        #         negative_gap_packets += seq_diff_packets
        #     elif 0 <= seq_diff < 1380:
        #         print(f"stange diff: {seq_diff}")
        #         strange_gaps += 1
        #     elif seq_diff == 1380:
        #         correct_gaps += 1
        #     elif 1380 < seq_diff:
        #         print(f"big diff: {seq_diff}")
        #         big_gaps += 1
        #         big_gap_packets += seq_diff_packets
        #     else:
        #         other += 1
        # num_sent = len(send_pkts)
        # num_recv = len(recv_pkts)
        # bdp = sim.bw_Mbps * 1e6 * (2 * sim.edge_delays[unfair_idx] + sim.btl_delay_us) * 1e-6 / (1380 * 8)
        # total_cap = (10e9 * sim.edge_delays[unfair_idx] * 1e-6 * 2 + sim.bw_Mbps * 1e6 * sim.btl_delay_us * 1e-6) / (1380 * 8)
        # aligned = np.zeros((num_sent, ), dtype=[("send seq", "int"), ("received?", "int")])
        # assert num_sent >= num_recv
        # send_idx = 0
        # recv_idx = 0
        # num_valid = None
        # for send_pkt in send_pkts:
        #     send_seq = send_pkt[0]
        #     aligned[send_idx][0] = send_seq
        #     if recv_idx < num_recv:
        #         recv_seq = recv_pkts[recv_idx][0]
        #         if send_seq == recv_seq:
        #             aligned[send_idx][1] = 1
        #             recv_idx += 1
        #     else:
        #         if num_valid is None:
        #             num_valid = send_idx
        #     send_idx += 1
        # print("Send seq, received?")
        # for send_seq, was_received in aligned.tolist():
        #     print(send_seq, "-----------------------------------------------------" if not was_received else "1")
        # missing = abs(np.sum(aligned['received?'] - 1))
        # tail = num_sent - num_valid
        # print(f"missing: {missing}")
        # print(f"tail: {tail}")
        # print(f"missing not tail: {missing - tail}")
        # print(f"BDP no queue: {bdp}")
        # print(f"BDP with queue: {bdp + sim.queue_p}")
        # print(f"total capacity no queue: {total_cap}")
        # print(f"total capacity with queue: {total_cap + sim.queue_p}")
        # print(f"num packets sent: {num_sent}")
        # print(f"num packets received: {num_recv}")
        # print(f"missing packets: {num_sent - num_recv}")
        # print(f"Big gaps: {big_gaps}")
        # print(f"Negative gaps: {negative_gaps}")
        # print(f"strange gaps: {strange_gaps}")
        # print(f"correct gaps: {correct_gaps}")
        # print(f"other gaps: {other}")
        # print(f"Big gap packets: {big_gap_packets}")
        # print(f"Negative gap packets: {negative_gap_packets}")
        # print(f"big/negative gap diff: {big_gap_packets - negative_gap_packets}")
        # print(f"end-to-end seq diff in packets: {(recv_pkts[-1][0] - recv_pkts[0][0]) / 1380}")
        # print(f"max seq in packets: {max_seq / 1380}")
        # print(f"retransmits: {retransmits}")

        for j, recv_pkt in enumerate(recv_pkts):
            # Regular metrics.
            recv_pkt_seq = recv_pkt[0]
            output[j]["seq"] = recv_pkt_seq
            curr_recv_time = recv_pkt[1]
            output[j]["arrival time"] = curr_recv_time
            if j > 0:
                interarrival_time = curr_recv_time - recv_pkts[j - 1][1]
            else:
                interarrival_time = 0
            output[j]["inter-arrival time"] = interarrival_time

            # Process packet loss -
            # Count the number of dropped packets by checking if the
            # sequence numbers at sender and receiver are the same. If
            # not, the packet is dropped, and the packet_loss counter
            # increases by one to keep the index offset at sender
            send_pkt_seq = send_pkts[j + packet_loss][0]
            prev_loss = packet_loss
            while send_pkt_seq != recv_pkt_seq:
                # Packet loss
                packet_loss += 1
                send_pkt_seq = send_pkts[j + packet_loss][0]
            curr_loss = packet_loss - prev_loss

            # if curr_loss > 0:
            #     print(f"Lost packets: {curr_loss}")
            # continue

            # Calculate the true RTT ratio.
            output[j]["true RTT ratio"] = (
                # Look up the send time of this packet to calculate
                # the true sender-receiver delay.
                (curr_recv_time - send_pkts[j + packet_loss][1] +
                 # Assume that, on the reverse path, packets will
                 # experience no queuing delay.
                 one_way_us) /
                # Compare to the minimum RTT.
                (2 * one_way_us))

            # Calculate loss event rate
            if packet_loss == 0:
                output[j]["loss event rate"] = 0.0
            if curr_loss > 0:
                curr_loss_start = j + packet_loss - curr_loss
                if curr_event_start_idx == 0:
                    # First loss event
                    curr_event_start_idx = curr_loss_start
                    curr_event_start_time = send_pkts[curr_event_start_idx][1]
                    loss_event_rate = 1 / (1.0 * (curr_loss + 1))
                    output[j]["loss event rate"] = loss_event_rate
                else:
                    # See if any loss packets start a new interval
                    prev_recv_time = recv_pkts[j - 1][1]
                    loss_interval = (
                        curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                    for k in range(0, curr_loss):
                        loss_time = prev_recv_time + (k + 1) * loss_interval
                        if (loss_time - curr_event_start_time >=
                                output[j]["true RTT ratio"] * 2 * one_way_us):
                            # Start of a new event - first store the previous event
                            loss_event_intervals.appendleft(curr_loss_start + k -
                                                            curr_event_start_idx)
                            if len(loss_event_intervals) > NUM_INTERVALS:
                                loss_event_intervals.pop()
                            curr_event_start_time = loss_time
                            curr_event_start_idx = curr_loss_start + k
                    curr_event_size = j + packet_loss - curr_event_start_idx
                    output[j]["loss event rate"] = compute_weighted_average(curr_event_size,
                                                                            loss_event_intervals,
                                                                            loss_interval_weight)
            else:
                if packet_loss == 0:
                    output[j]["loss event rate"] = 0.0
                else:
                    # Increase current event size
                    curr_event_size = j + packet_loss - curr_event_start_idx
                    output[j]["loss event rate"] = compute_weighted_average(curr_event_size,
                                                                            loss_event_intervals,
                                                                            loss_interval_weight)

            # Compute Mathis Model Estimation
            # First compute loss rate in last eight min RTT
            prev_recv_time = recv_pkts[j - 1][1]
            loss_interval = (
                curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
            for k in range(0, curr_loss):
                mathis_loss_queue.append(prev_recv_time + (k + 1) * loss_interval)

            # Pop out early loss
            while (mathis_loss_queue and
                   mathis_loss_queue[0] < (curr_recv_time - 8 * min_rtt_us)):
                mathis_loss_queue.popleft()
            mathis_loss_count = len(mathis_loss_queue)

            # Update 8rtt window start
            while curr_recv_time - recv_pkts[mathis_8rtt_window_start][1] > 8 * min_rtt_us:
                mathis_8rtt_window_start += 1

            # Loss rate over 8rtt window
            mathis_loss_count = len(mathis_loss_queue)
            if mathis_loss_count > 0:
                mathis_loss_rate = (mathis_loss_count /
                                    (1.0 * j - mathis_8rtt_window_start + mathis_loss_count))

                # Throughput computed by mathis equation (in number of packets per second)
                mathis_throughput = 1.2247 / (1.0 * min_rtt_s * math.sqrt(mathis_loss_rate))

                continuous_loss_interval.append(mathis_throughput)

                mathis_fair_throughput = mean(continuous_loss_interval)
            else:
                if j == 0:
                    mathis_fair_throughput = 0.0
                else:
                    continuous_loss_interval.clear()

            output[j]["mathis model throughput"] = mathis_fair_throughput

            if mathis_fair_throughput == 0.0:
                mathis_label = -1
            else:
                curr_throughput = (j - mathis_8rtt_window_start) / (8.0 * min_rtt_s)
                if curr_throughput > mathis_fair_throughput:
                    mathis_label = 1
                else:
                    mathis_label = 0
            output[j]["mathis model label"] = mathis_label

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                metric = make_ewma_metric(metric, alpha)
                if j > 0:
                    if "inter-arrival time" in metric:
                        new = interarrival_time

                        tput_metric = make_ewma_metric("throughput ewma", alpha)
                        new_tput = 0 if interarrival_time == 0 else 1 / interarrival_time
                        output[j][tput_metric] = update_ewma(
                            output[j - 1][tput_metric], new_tput, alpha)
                    elif "throughput" in metric:
                        # The throughput is calculated during the
                        # inter-arrival time calculation, above.
                        continue
                    elif "RTT ratio" in metric:
                        # TODO: RTT ratio EWMA
                        new = 0
                    elif "loss rate" in metric:
                        # Divide curr_loss by (curr_loss + 1) because
                        # over the course of sending (curr_loss + 1)
                        # packets, one got through and curr_loss were
                        # lost.
                        new = curr_loss / (curr_loss + 1)
                    elif "queue occupancy" in metric:
                        # Queue occupancy is calculated using the
                        # router's PCAP files, below.
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
                window_size = win * min_rtt_us
                state = windowed_state[win]

                if "average inter-arrival time" in metric:
                    # This is calculated as part of the loss rate
                    # calculation, below.
                    continue
                if "average throughput" in metric:
                    # The average throughput is calculated as part of
                    # the loss rate calculation, below.
                    continue
                if "average RTT ratio" in metric:
                    # TODO: Average RTT ratio over a window.
                    new = 0
                elif "loss rate" in metric:
                    # Process packet loss.
                    if (curr_loss > 0 and j > 0):
                        prev_recv_time = recv_pkts[j - 1][1]
                        loss_interval = (
                            curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                        for k in range(0, curr_loss):
                            state["loss_queue"].append(
                                prev_recv_time + (k + 1) * loss_interval)
                    # Pop out earlier loss.
                    while (state["loss_queue"] and
                           (state["loss_queue"][0] <
                            curr_recv_time - window_size)):
                        state["loss_queue"].popleft()

                    # Update window_start.
                    while ((curr_recv_time -
                            recv_pkts[state["window_start"]][1]) > window_size):
                        state["window_start"] += 1

                    # If it's processing the first packet
                    # (j == window_start == 0) or no other packets were
                    # received within the RTT window, then output 0
                    # for the loss rate.
                    if j - state["window_start"] > 0:
                        # len_loss_q = len(state["loss_queue"])
                        # num_pkts = len_loss_q + j - state["window_start"]
                        # loss_rate = len_loss_q / num_pkts
                        # print(f"len_loss_q: {len_loss_q}, num_pkts: {num_pkts}, loss rate: {loss_rate}")
                        new = (len(state["loss_queue"]) / (1.0 * (
                            len(state["loss_queue"]) + j -
                            state["window_start"])))
                    else:
                        new = 0

                    # Calculate the average inter-arrival time.
                    avg_interarrival_time = (
                        (curr_recv_time -
                         recv_pkts[j - state["window_start"]][1]) /
                        (j - state["window_start"] + 1))
                    output[j][make_win_metric(
                        "average inter-arrival time windowed", win)] = avg_interarrival_time
                    new_tput = 0 if avg_interarrival_time == 0 else 1 / avg_interarrival_time
                    output[j][make_win_metric(
                        "average throughput windowed", win)] = new_tput
                elif "queue occupancy" in metric:
                    # Queue occupancy is calculated using the router's
                    # PCAP files, below.
                    continue
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new
        valid = np.where(output["loss event rate"] > 0)
        output["loss event rate sqrt"][valid] = np.reciprocal(
            np.sqrt(output["loss event rate"][valid]))
        unfair_flws.append(output)

        # print(f"total losses detected: {packet_loss}")
        # raise Exception()

    # Save memory by explicitly deleting the sent and received packets
    # after they've been parsed. This happens outside of the above
    # for-loop because only the last iteration's sent and received
    # packets are not automatically cleaned up by now (they go out of
    # scope when the send_pkts and recv_pkts variables are overwritten
    # by the next loop).
    del send_pkts
    del recv_pkts

    # Process pcap files from the bottleneck router to determine queue
    # occupency. router_pkts is a list of tuples of the form:
    #     (sender, timestamp)
    router_pkts = utils.parse_packets_router(
        path.join(sim_dir, f"{sim.name}-1-0.pcap"), sim.payload_B)
    # Index of the output array where the queue occupency result
    # should be appended. Once for each unfair flow, since the
    # per-flow output is stored separately.
    output_idxs = [0] * sim.unfair_flws
    # Track the number of other flows' packets that have arrived since
    # the last packet for each flow.
    pkts_since_last = {sender: 0 for sender in range(sim.unfair_flws)}
    # State that the windowed metrics need to track across
    # packets.
    windowed_state = {win: {
        "window_pkt_count": 0,
        "window_start": 0
    } for win in WINDOWS}
    # Loop over all of the packets receiver by the bottleneck
    # router. Note that we process all flows at once.
    for j, router_pkt in enumerate(router_pkts):
        # The flow to which this packet belongs.
        sender, curr_time = router_pkt
        # Process only packets that are part of one of the unfair
        # flows. Discard packets that did not make it to the receiver
        # (e.g., at the end of the experiment).
        if (sender < sim.unfair_flws and
                output_idxs[sender] < unfair_flws[sender].shape[0]):
            # We cannot move this above the if-statement condition
            # because it is valid only if sender < sim.unfair_flws.
            output_idx = output_idxs[sender]

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                metric = make_ewma_metric(metric, alpha)
                if j > 0:
                    if "inter-arrival time" in metric:
                        # The inter-arrival time is calculated using
                        # the sender and receiver logs, above.
                        continue
                    if "throughput" in metric:
                        # The throughput is calculated using the
                        # sender and receiver logs, above.
                        continue
                    if "RTT ratio" in metric:
                        # The RTT ratio is calculated using the sender
                        # and receiver logs, above.
                        continue
                    if "loss rate" in metric:
                        # The loss rate is calculated using the sender
                        # and receiver logs, above.
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
                        new = 1 / pkts_since_last[sender]
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
                state = windowed_state[win]
                if "average inter-arrival time" in metric:
                    # The average inter-arrival time is calculated
                    # using the sender and receiver logs, above.
                    continue
                if "average throughput" in metric:
                    # The average throughput is calculated using the
                    # sender and receiver logs, above.
                    continue
                if "average RTT ratio" in metric:
                    # The average RTT ratio time is calculated using
                    # the sender and receiver logs, above.
                    continue
                if "loss rate" in metric:
                    # The loss rate is calculated using the sender
                    # and receiver logs, above.
                    continue
                if "queue occupancy" in metric:
                    state["window_pkt_count"] += 1
                    # Update window_start index for this flow, if the
                    # window is larger than the expected window
                    # size. Also decrement packet count if the packet
                    # belongs to the unfair flow.
                    window_start = state["window_start"]
                    # Use RTT ratio from previous step to compute
                    # actual RTT (Assuming that the one-way delay for
                    # the ACK sending back to sender would be min
                    # one-way delay).
                    rtt_ratio = unfair_flws[sender][output_idx]["true RTT ratio"]
                    actual_rtt_us = min_rtts_us[sender] * (rtt_ratio + 0.5)
                    window_size = win * actual_rtt_us
                    # To avoid the window size bouncing back and
                    # forth, only allow the window to grow/shrink in
                    # one direction.
                    if curr_time - router_pkts[window_start][1] > window_size:
                        while (curr_time - router_pkts[window_start][1] >
                               window_size):
                            # Check if this packet is part of the same flow.
                            if router_pkts[window_start][0] == sender:
                                state["window_pkt_count"] -= 1
                            window_start += 1
                    # Grow the window size to eariler packets
                    elif curr_time - router_pkts[window_start][1] < window_size:
                        while (window_start > 0 and
                               (curr_time - router_pkts[window_start][1] <
                                window_size)):
                            # Check if this packet is part of the same flow.
                            if router_pkts[window_start][0] == sender:
                                state["window_pkt_count"] += 1
                            window_start -= 1

                    # Record the new value of window_start
                    state["window_start"] = window_start
                    # Get queue occupency
                    new = state["window_pkt_count"] / (j - window_start + 1)
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                unfair_flws[sender][output_idx][metric] = new
            output_idxs[sender] += 1

        # For each unfair flow except the current packet's flow,
        # increment the number of packets since the last packet from
        # that flow.
        for flw in range(sim.unfair_flws):
            if flw != sender:
                pkts_since_last[flw] += 1
        # For the current packet's flw, the number of packets since
        # the last packet in this flow is now 1.
        pkts_since_last[sender] = 1

    # Determine if there are any NaNs in the results. For the results
    # for each unfair flow, look through all features (columns) and
    # make a note of the features that contain NaNs. Flatten these
    # lists of feature names, using a set comprehension to remove
    # duplicates.
    nan_fets = {
        fet for fets in
        [fet_ for flw_dat in unfair_flws
         for fet_ in flw_dat.dtype.names if np.isnan(flw_dat[fet_]).any()]
        for fet in fets}
    # If there are NaNs, then we do not want to save these results.
    if nan_fets:
        print((f"    Discarding simulation {sim_dir} because it has NaNs in "
               f"features: {nan_fets}"))
        return

    # Write to output
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
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir

    # Find all simulations.
    pcaps = [(path.join(exp_dir, sim), out_dir)
             for sim in sorted(os.listdir(exp_dir))]
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
