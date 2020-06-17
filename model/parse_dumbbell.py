#! /usr/bin/env python3
"""Parses the pcap file of dumbbell topology. """

import argparse
import multiprocessing
import os
from os import path
import time
from collections import deque

import numpy as np
import scapy.layers.inet

import utils


# Whether to parse PCAP files synchronously or in parallel.
SYNC = False
# RTT window to consider queue occupency
RTT_WINDOW = 2
# The dtype of the output.
DTYPE = [("seq", "int32"),
         ("RTT ratio", "float"),
         ("inter-arrival time", "float"),
         ("loss rate", "float"),
         ("queue occupancy", "float")]


def parse_pcap(sim_dir, out_dir, rtt_window):
    """Parse a PCAP file.

    Writes a npz file that contains the sequence number, RTT ratio,
    inter-arrival time, loss rate, and queue occupency percentage in
    the last n RTT (default set to be 2 RTT).
    """
    print(f"Parsing: {sim_dir}")
    sim = path.basename(sim_dir)
    _, btl_delay_us, _, _, payload_B, unfair_flows, \
             other_flows, edge_delays = utils.parse_sim_name(sim)
    assert unfair_flows > 0, f"No unfair flows to analyze: {sim_dir}"

    # Construct the output filepaths.
    out_flps = [
        path.join(
            out_dir,
            f"{sim}-{rtt_window}rttW-{i + 1}flowNum.npz")
        for i in range(unfair_flows)]
    # For all of the output filepaths, check if the file already
    # exists. If all of the output files exist, then we do not need to
    # parse this file.
    if np.vectorize(lambda p: path.exists(p))(np.array(out_flps)).all():
        print(f"    Already parsed: {sim_dir}")
        return

    # Process PCAP files from unfair senders and receivers

    output_list = []
    rtt_list = []

    for i in range(0, unfair_flows):
        rtt_us = (btl_delay_us + 2 * edge_delays[i]) * 2
        rtt_list.append(rtt_us)
        one_way_us = float(rtt_us / 2.0)

        # (Seq, timestamp)
        send_pkts = parse_packets_endpoint(
            path.join(sim_dir, f"{sim}-{i + 2}-0.pcap"), payload_B)
        recv_pkts = parse_packets_endpoint(
            path.join(sim_dir, f"{sim}-{i + 2 + unfair_flows + other_flows}-0.pcap"),
            payload_B)

        packet_loss = 0
        window_size = rtt_us * rtt_window
        window_start = 0
        loss_queue = deque()
        output = np.empty(len(recv_pkts), dtype=DTYPE)

        for j, recv_pkt in enumerate(recv_pkts):
            recv_pkt_seq = recv_pkt[0]
            send_pkt_seq = send_pkts[j + packet_loss][0]

            # Count the number of dropped packets by checking if the
            # sequence numbers at sender and receiver are the same. If
            # not, the packet is dropped, and the packet_loss counter
            # increases by one to keep the index offset at sender
            prev_loss = packet_loss
            while send_pkt_seq != recv_pkt_seq:
                # Packet loss
                packet_loss += 1
                send_pkt_seq = send_pkts[j + packet_loss][0]

            curr_loss = packet_loss - prev_loss
            curr_recv_time = recv_pkt[1]
            curr_send_time = send_pkts[j + packet_loss][1]

            # Process packet loss
            if (curr_loss > 0 and j > 0):
                prev_recv_time = recv_pkts[j - 1][1]
                loss_interval = (curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                for k in range(0, curr_loss):
                    loss_queue.append(prev_recv_time + (k + 1) * loss_interval)

            # Pop out earlier loss
            while loss_queue and loss_queue[0] < curr_recv_time - window_size:
                loss_queue.popleft()

            # Update window_start
            while curr_recv_time - recv_pkts[window_start][1] > window_size:
                window_start += 1

            # Fill into output
            output[j][0] = recv_pkt_seq
            output[j][1] = (curr_recv_time - curr_send_time) / one_way_us

            # If it's processing the first packet (j == window_start == 0)
            # Or no other packets received within the RTT window,
            # output 0 for RTT ratio and inter-arrival time
            if j - window_start > 0:
                output[j][2] = (curr_recv_time - recv_pkts[window_start][1]) / \
                               (1.0 * (j - window_start))
                output[j][3] = len(loss_queue) / (1.0 * (len(loss_queue) + j - window_start))
            else:
                output[j][2] = 0
                output[j][3] = 0

        output_list.append(output)

    # Process pcap files from routers to determine queue occupency
    # (sender, timestamp)
    router_pkts = parse_packets_routerpath.join(sim_dir, f"{sim}-1-0.pcap"), payload_B)
    # Number of packets sent by the unfair flows within RTT window
    # Note that RTT window could be different for flows with different RTT
    window_pkt_count = [0] * unfair_flows
    # Start of the window for each flow (index in router_pkts)
    # Since different flows could have different RTTs, each flow
    # needs to keep track of their own window start
    router_window_start = [0] * unfair_flows
    # Index of the output array where the queue occupency result should be appended
    output_index = [0] * unfair_flows

    for i, router_pkt in enumerate(router_pkts):
        sender = router_pkt[0]

        if sender < unfair_flows:
            curr_time = router_pkts[1]
            window_pkt_count[sender] += 1

            # Update window_start index for this flow, if the window is larger than
            # the expected window size. Also decrement packet count if the packet
            # belongs to the unfair flow
            window_start = router_window_start[sender]
            while (curr_time - router_pkts[window_start][1] >
                   rtt_list[sender] * rtt_window):
                if router_pkts[window_start][0] == sender:
                    window_pkt_count[sender] -= 1
                window_start += 1

            router_window_start[sender] = window_start
            # Get queue occupency
            queue_occupency = window_pkt_count[sender] / float(i - window_start + 1)

            # Add this record to output array
            if output_index[sender] < len(output_list[sender]):
                output_list[sender][output_index[sender]][4] = queue_occupency
                output_index[sender] = output_index[sender] + 1

    # Write to output
    for i, out_flp in enumerate(out_flps):
        print(f"    Saving: {out_flp}")
        np.savez_compressed(out_flp, output_list[i])


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
        "--rtt-window", default=RTT_WINDOW,
        help=("Size of the RTT window to calculate receiving rate and loss "
              "rate (default: 2 * RTT)"), type=int)
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir
    rtt_window = args.rtt_window

    # Find all simulations.
    pcaps = [(path.join(exp_dir, sim), out_dir, rtt_window)
             for sim in os.listdir(exp_dir)]
    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    if SYNC:
        [parse_pcap(*pcap) for pcap in pcaps]
    else:
        with multiprocessing.Pool() as pol:
            pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
