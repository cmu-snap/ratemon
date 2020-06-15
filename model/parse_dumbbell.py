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
         ("arrival time", "float"),
         ("inter-arrival time", "float"),
         ("RTT ratio", "float"),
         ("loss rate", "float"),
         ("queue occupancy", "float")]


def parse_pcap(sim_dir, out_dir, rtt_window):
    """Parse a PCAP file.

    Writes a npz file that contains the sequence number, RTT ratio,
    inter-arrival time, loss rate, and queue occupency percentage in
    the last n RTT (default set to be 2 RTT).
    """
    print(f"Parsing: {sim_dir}")
    sim = utils.Sim(sim_dir)
    assert sim.unfair_flws > 0, f"No unfair flows to analyze: {sim_dir}"

    # Construct the output filepaths.
    out_flp = path.join(out_dir, f"{sim.name}-{rtt_window}rttW.npz")
    # If the output file exists, then we do not need to parse this file.
    if path.exists(out_flp):
        print(f"    Already parsed: {sim_dir}")
        return

    # Process PCAP files from unfair senders and receivers

    unfair_flws = []
    rtts_us = []

    for unfair_idx in range(sim.unfair_flws):
        one_way_us = sim.btl_delay_us + 2 * sim.edge_delays[unfair_idx]
        rtt_us = one_way_us * 2
        rtts_us.append(rtt_us)

        send_pkts = utils.parse_packets(
            path.join(sim_dir, f"{sim.name}-{unfair_idx + 2}-0.pcap"),
            sim.payload_B)
        recv_pkts = utils.parse_packets(
            path.join(sim_dir,
                      f"{sim.name}-{unfair_idx + 2 + sim.unfair_flws + sim.other_flws}-0.pcap"),
            sim.payload_B)

        packet_loss = 0
        window_size = rtt_us * rtt_window
        window_start = 0
        loss_queue = deque()
        output = np.empty(len(recv_pkts), dtype=DTYPE)

        for j, recv_pkt in enumerate(recv_pkts):
            recv_pkt_seq = recv_pkt[1][scapy.layers.inet.TCP].seq
            send_pkt_seq = send_pkts[j + packet_loss][1][scapy.layers.inet.TCP].seq

            # Count the number of dropped packets by checking if the
            # sequence numbers at sender and receiver are the same. If
            # not, the packet is dropped, and the packet_loss counter
            # increases by one to keep the index offset at sender
            prev_loss = packet_loss
            while send_pkt_seq != recv_pkt_seq:
                # Packet loss
                packet_loss += 1
                send_pkt_seq = send_pkts[j + packet_loss][1][scapy.layers.inet.TCP].seq

            curr_loss = packet_loss - prev_loss
            curr_recv_time = utils.parse_time_us(recv_pkt[0])
            curr_send_time = utils.parse_time_us(send_pkts[j + packet_loss][0])

            # Process packet loss
            if (curr_loss > 0 and j > 0):
                prev_recv_time = utils.parse_time_us(recv_pkts[j-1][0])
                loss_interval = (curr_recv_time - prev_recv_time) / (curr_loss + 1.0)
                for k in range(0, curr_loss):
                    loss_queue.append(prev_recv_time + (k + 1) * loss_interval)

            # Pop out earlier loss
            while loss_queue and loss_queue[0] < curr_recv_time - (window_size):
                loss_queue.popleft()

            # Update window_start
            while (curr_recv_time - utils.parse_time_us(recv_pkts[window_start][0]) >
                   window_size):
                window_start += 1

            output[j]["seq"] = recv_pkt_seq
            output[j]["arrival time"] = curr_recv_time
            if j > 0:
                interarrival_time = curr_recv_time - utils.parse_time_us(
                    recv_pkts[j - 1][0])
            else:
                interarrival_time = 0
            output[j]["inter-arrival time"] = interarrival_time
            output[j]["RTT ratio"] = ((curr_recv_time - curr_send_time) /
                                      one_way_us)
            # If it's processing the first packet (j == window_start == 0) or no
            # other packets received within the RTT window, then output 0 for
            # loss rate.
            if j - window_start > 0:
                loss_rate = (len(loss_queue) /
                             (1.0 * (len(loss_queue) + j - window_start)))
            else:
                loss_rate = 0
            output[j]["loss rate"] = loss_rate

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
    router_pkts = utils.parse_packets(
        path.join(sim_dir, f"{sim.name}-1-0.pcap"), sim.payload_B)
    # Number of packets sent by the unfair flows within RTT window
    # Note that RTT window could be different for flows with different RTT
    window_pkt_count = [0] * sim.unfair_flws
    # Start of the window for each flow (index in router_pkts)
    # Since different flows could have different RTTs, each flow
    # needs to keep track of their own window start
    router_window_start = [0] * sim.unfair_flws
    # Index of the output array where the queue occupency result should be appended
    output_index = [0] * sim.unfair_flws

    for i, router_pkt in enumerate(router_pkts):
        sender = int(router_pkt[1][scapy.layers.inet.IP].src.split(".")[2])

        if sender < sim.unfair_flws:
            curr_time = utils.parse_time_us(router_pkt[0])
            window_pkt_count[sender] += 1

            # Update window_start index for this flow, if the window is larger than
            # the expected window size. Also decrement packet count if the packet
            # belongs to the unfair flow
            window_start = router_window_start[sender]
            while (curr_time - utils.parse_time_us(router_pkts[window_start][0]) >
                   rtts_us[sender] * rtt_window):
                prev_sender = int(router_pkts[window_start][1]
                                  [scapy.layers.inet.IP].src.split(".")[2])
                if prev_sender == sender:
                    window_pkt_count[sender] -= 1
                window_start += 1

            router_window_start[sender] = window_start
            # Get queue occupency
            queue_occupency = window_pkt_count[sender] / float(i - window_start + 1)

            # Add this record to output array
            if output_index[sender] < len(unfair_flws[sender]):
                unfair_flws[sender][output_index[sender]]["queue occupancy"] = (
                    queue_occupency)
                output_index[sender] = output_index[sender] + 1

    # Write to output
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
        # Use 20 workers only to reduce the likelihood of running out
        # of memory.
        with multiprocessing.Pool(processes=20) as pol:
            pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
