#! /usr/bin/env python3
"""Parses the pcap file of dumbbell topology. """

import argparse
import json
import multiprocessing
import os
from os import path
import time
from collections import deque

import numpy as np
import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp

from utils import *


# RTT window to consider queue occupency
RTT_WINDOW = 2
# The dtype of the output.
DTYPE = [("seq", "int32"),
         ("RTT ratio", "float"),
         ("inter-arrival time", "float"),
         ("loss rate", "float"),
         ("queue occupancy", "float")]

def parse_pcap(flp, out_dir, rtt_window):
    """Parse a PCAP file.

    Writes to a npz file that contains the sequence number, RTT ratio, inter-arrival time,
    loss rate, and queue occupency percentage in the last n RTT (default set to be 2 RTT).
    """
    print(f"Parsing: {flp}")
    # E.g., 64Mbps-40000us-100p-1unfair-8other-1380B-2s-1-1.pcap
    bw_Mbps, btl_delay_us, queue_p, dur_s, packet_size_B, unfair_flows, \
             other_flows, edge_delays = parse_filename(flp)

    # Process PCAP files from unfair senders and receivers

    output_list = []
    rtt_list = []

    for i in range(0, unfair_flows):
        send_flp = flp[:-8] + str(i + 2) + "-0.pcap"
        recv_flp = flp[:-8] + str(i + 2 + unfair_flows + other_flows) + "-0.pcap"

        rtt_us = (btl_delay_us + 2 * edge_delays[i]) * 2
        rtt_list.append(rtt_us)
        one_way_us = float(rtt_us / 2.0)

        send_pkts = parse_packets(send_flp, packet_size_B)

        recv_pkts = parse_packets(recv_flp, packet_size_B)

        packet_loss = 0
        window_size = rtt_us * rtt_window
        window_start = 0
        loss_queue = deque()
        output = np.empty(len(recv_pkts), dtype=DTYPE)

        for j in range(0, len(recv_pkts)):
            recv_pkt_seq = recv_pkts[j][1][scapy.layers.inet.TCP].seq
            send_pkt_seq = send_pkts[j + packet_loss][1][scapy.layers.inet.TCP].seq

            # Count the number of dropped packets by checking if the sequence numbers
            # at sender and receiver are the same. If not, the packet is dropped, and
            # the packet_loss counter increases by one to keep the index offset at sender
            prev_loss = packet_loss
            while send_pkt_seq != recv_pkt_seq:
                # Packet loss
                packet_loss += 1
                send_pkt_seq = send_pkts[j + packet_loss][1][scapy.layers.inet.TCP].seq

            curr_loss = packet_loss - prev_loss

            curr_recv_time = parse_time_us(recv_pkts[j][0])
            curr_send_time = parse_time_us(send_pkts[j + packet_loss][0])

            # Process packet loss
            if (curr_loss > 0 and j > 0):
                prev_recv_time = parse_time_us(recv_pkts[j-1][0])
                loss_interval = (curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                for k in range(0, curr_loss):
                    loss_queue.append(prev_recv_time + (k + 1) * loss_interval)

            # Pop out earlier loss
            while loss_queue and loss_queue[0] < curr_recv_time - (window_size):
                loss_queue.popleft()

            # Update window_start
            while curr_recv_time - parse_time_us(recv_pkts[window_start][0]) > window_size:
                window_start += 1

            # Fill into output
            output[j][0] = recv_pkt_seq
            output[j][1] = (curr_recv_time - curr_send_time) / one_way_us

            # If it's processing the first packet (j == window_start == 0)
            # Or no other packets received within the RTT window,
            # output 0 for RTT ratio and inter-arrival time
            if j - window_start > 0:
                output[j][2] = (curr_recv_time - parse_time_us(recv_pkts[window_start][0])) / \
                               (1.0 * (j - window_start))
                output[j][3] = len(loss_queue) / (1.0 * (len(loss_queue) + j - window_start))
            else:
                output[j][2] = 0
                output[j][3] = 0

        output_list.append(output)


    # Process pcap files from routers to determine queue occupency
    router_pkts = parse_packets(flp, packet_size_B)

    # Number of packets sent by the unfair flows within RTT window
    # Note that RTT window could be different for flows with different RTT
    window_pkt_count = [0] * unfair_flows
    # Start of the window for each flow (index in router_pkts)
    # Since different flows could have different RTTs, each flow
    # needs to keep track of their own window start
    router_window_start = [0] * unfair_flows
    # Index of the output array where the queue occupency result should be appended
    output_index = [0] * unfair_flows

    for i in range(len(router_pkts)):
        sender = int(router_pkts[i][1][scapy.layers.inet.IP].src.split(".")[2])

        if sender < unfair_flows:
            curr_time = parse_time_us(router_pkts[i][0])
            window_pkt_count[sender] += 1

            # Update window_start index for this flow, if the window is larger than
            # the expected window size. Also decrement packet count if the packet
            # belongs to the unfair flow
            window_start = router_window_start[sender]
            while (curr_time - parse_time_us(router_pkts[window_start][0]) >
                   rtt_list[sender] * rtt_window):
                prev_sender = int(router_pkts[window_start][1]
                                  [scapy.layers.inet.IP].src.split(".")[2])
                if prev_sender == sender:
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
    for i in range(unfair_flows):
        out_flp = path.join(out_dir, f"{path.basename(flp)[:-9]}-{rtt_window}rttW-{i+1}flowNum"
                            + "-dumbbell.npz")
        print("Saving " + out_flp)
        np.savez_compressed(out_flp, output_list[i])


def main():
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Parses the output of gen_training_data.py.")
    psr.add_argument(
        "--exp-dir",
        help=("The directory in which the experiment results are stored "
              "(required)."),
        required=True, type=str)
    psr.add_argument(
        "--out-dir", help=("The directory in which to store output files "
                           "(required)."),
        required=True, type=str)
    psr.add_argument('--rtt-window', type=int, default=RTT_WINDOW,
                     help='Size of the RTT window to calculate receiving \
                     rate and loss rate (default: 2 * RTT)')
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir
    rtt_window = args.rtt_window

    # Select only PCAP files (".pcap") that were captured at the router's input
    # ("-1-0") and were not the result of a failed experiment.
    pcaps = [(path.join(exp_dir, fln), out_dir, rtt_window)
             for fln in os.listdir(exp_dir)
             if fln.endswith("-1-0.pcap")]
    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    with multiprocessing.Pool() as pol:
        pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()