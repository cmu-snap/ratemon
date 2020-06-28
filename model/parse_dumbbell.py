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
    ("RTT ratio", "float")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("RTT ratio ewma", "float"),
    ("inter-arrival time ewma", "float"),
    ("loss rate ewma", "float"),
    ("queue occupancy ewma", "float")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    ("average inter-arrival time windowed", "float"),
    ("loss rate windowed", "float"),
    ("queue occupancy windowed", "float")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 10 for i in range(11)]
# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(0, 11, 2)]
# The final dtype combines each metric at multiple granularities.
DTYPE = (REGULAR +
         [(f"{name}-alpha{alpha}", typ)
          for (name, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
         [(f"{name}-minRtt{win}", typ)
          for (name, typ), win in itertools.product(WINDOWED, WINDOWS)])


def update_ewma(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev


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
                f"{sim.name}-{unfair_idx + 2 + sim.unfair_flws + sim.other_flws}-0.pcap"),
            sim.payload_B)

        # State that the windowed metrics need to track across packets.
        windowed_state = {win: {
            "window_start": 0,
            "loss_queue": deque(),
            "window_pkt_count": 0,
            "router_window_start": 0

            } for win in WINDOWS}

        # Number of packet loss up to the current processing packet
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

            # Process packet loss
            curr_loss = packet_loss - prev_loss
            if (curr_loss > 0 and j > 0):
                prev_recv_time = recv_pkts[j - 1][1]
                loss_interval = (curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                # Update all windowed metrics
                for (metric, _). win in itertools.product(WINDOWED, WINDOWS):
                    state = windowed_state[win]
                    for k in range(0, curr_loss):
                        state["loss_queue"].append(prev_recv_time + (k + 1) * loss_interval)

            curr_send_time = send_pkts[j + packet_loss][1]
            output[j]["RTT ratio"] = (curr_recv_time - curr_send_time) / one_way_us

            # EWMA metrics.
            for (metric, ), alpha in itertools.product(EWMAS, ALPHAS):
                if j > 0:
                    if "RTT ratio" in metric:
                        new = output[j]["RTT ratio"]
                    elif "inter-arrival time" in metric:
                        new = interarrival_time
                    elif "loss rate" in metric:
                        new = curr_loss / (curr_loss + 1)
                    else:
                        raise Exception(f"Unknown EWMA metric: {metric}")
                    new_ewma = update_ewma(output[j - 1][metric], new, alpha)
                else:
                    new_ewma = 0
                output[j][metric] = new_ewma

            # Windowed metrics.
            for (metric, _). win in itertools.product(WINDOWED, WINDOWS):
                window_size = win * min_rtt_us
                state = windowed_state[win]

                # Update window_start
                while curr_recv_time - recv_pkts[state["window_start"]][1] > window_size:
                    state["window_start"] += 1

                if "average inter-arrival time" in metric:
                    new = ((curr_recv_time - recv_pkts[j - state["window_start"]]) /
                           (1.0 * (j - state["window_start"] + 1)))
                elif "loss rate" in metric:
                    # If it's processing the first packet (j == window_start == 0) or no
                    # other packets received within the RTT window, then output 0 for
                    # loss rate.
                    if j - state["window_start"] > 0:
                        new = (len(state["loss_queue"]) /
                               (1.0 * (len(state["loss_queue"]) + j - state["window_start"])))
                    else:
                        new = 0
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

    # Process pcap files from routers to determine queue occupency
    # (sender, timestamp)
    router_pkts = utils.parse_packets_router(
        path.join(sim_dir, f"{sim.name}-1-0.pcap"), sim.payload_B)
    # Index of the output array where the queue occupency result should be appended
    output_index = [0] * sim.unfair_flws

    for i, router_pkt in enumerate(router_pkts):
        sender = router_pkt[0]

        if sender < sim.unfair_flws and output_index[sender] < len(unfair_flws[sender]):

            curr_time = router_pkt[1]

            for (metric, _). win in itertools.product(WINDOWED, WINDOWS):
                state = windowed_state[win]
                state['window_pkt_count'] += 1

                # Update window_start index for this flow, if the window is larger than
                # the expected window size. Also decrement packet count if the packet
                # belongs to the unfair flow
                window_start = state['router_window_start']

                # Use RTT ratio from previous step to compute actual RTT
                # (Assuming that the one-way delay for the ACK sending back to sender
                #  would be min one-way delay)
                rtt_ratio = output[output_index[sender]]["RTT ratio"]
                actual_rtt_us = min_rtts_us[sender] * (rtt_ratio + 0.5)
                window_size = win * actual_rtt_us

                # To avoid the window size bouncing back and forth,
                # only allow the window to grow/shrink in one direction
                if curr_time - router_pkts[window_start][1] > window_size:
                    while (curr_time - router_pkts[window_start][1] >
                           window_size):
                        if router_pkts[window_start][0] == sender:
                            state['window_pkt_count'] -= 1
                        window_start += 1
                # Grow the window size to eariler packets
                elif curr_time - router_pkts[window_start][1] < window_size:
                    while (window_start > 0 and curr_time - router_pkts[window_start][1] <
                           window_size):
                        if router_pkts[window_start][0] == sender:
                            state['window_pkt_count'] += 1
                        window_start -= 1

                state['router_window_start'] = window_start
                # Get queue occupency
                queue_occupency = state['window_pkt_count'] / float(i - window_start + 1)

                # Add this record to output array
                unfair_flws[sender][output_index[sender]][metric]["queue occupancy windowed"] = (
                    queue_occupency)
            output_index[sender] = output_index[sender] + 1

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
