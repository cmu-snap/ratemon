#! /usr/bin/env python3
"""Parses the pcap file of dumbbell topology. """

import argparse
import itertools
import multiprocessing
import os
from os import path
import time
from collections import deque

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
    ("true RTT ratio", "float")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("inter-arrival time ewma", "float"),
    ("RTT ratio ewma", "float"),
    ("loss rate ewma", "float"),
    ("queue occupancy ewma", "float")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    ("average inter-arrival time windowed", "float"),
    ("average RTT ratio windowed", "float"),
    ("loss rate windowed", "float"),
    ("queue occupancy windowed", "float")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 10 for i in range(1, 11)]
# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(0, 11)]


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

    for unfair_idx in range(sim.unfair_flws):
        one_way_us = sim.btl_delay_us + 2 * sim.edge_delays[unfair_idx] * 1.0
        min_rtt_us = one_way_us * 2
        min_rtts_us.append(min_rtt_us)

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

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(EWMAS, ALPHAS):
                metric = make_ewma_metric(metric, alpha)
                if j > 0:
                    if "inter-arrival time" in metric:
                        new = interarrival_time
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
                        new = (len(state["loss_queue"]) / (1.0 * (
                            len(state["loss_queue"]) + j -
                            state["window_start"])))
                    else:
                        new = 0

                    # Calculate the average inter-arrival time.
                    output[j][make_win_metric(
                        "average inter-arrival time windowed", win)] = (
                            (curr_recv_time -
                             recv_pkts[j - state["window_start"]][1]) /
                            (j - state["window_start"] + 1))
                elif "queue occupancy" in metric:
                    # Queue occupancy is calculated using the router's
                    # PCAP files, below.
                    continue
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new
        unfair_flws.append(output)
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
