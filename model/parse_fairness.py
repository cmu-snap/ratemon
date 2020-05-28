#! /usr/bin/env python3
"""Parses the pcap file of router's output from trainning data. """

import argparse
import json
import multiprocessing
import os
from os import path
import time

import numpy as np
import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp


# The IP address of the sender.
SRC_IP = "10.1.1.1"
# RTT window to consider queue occupency
RTT_WINDOW = 2
# Source port offset
SPORT_OFFSET = 49153


def failed(fln, fals):
    # TODO: This does not work.
    for cnf in fals:
        if (f"{cnf['bandwidth_Mbps']}Mbps-"
            f"{cnf['delay_us'] * 4}us-"
            f"{cnf['experiment_duration_s']}s"
            f"{cnf['queue_capacity_p']}p"
            f"{cnf['other_flows']}") in fln:
            print(f"Discarding: {fln}")
            return True
    return False


def parse_time_us(pkt_mdat):
    """
    Returns the timestamp, in microseconds, of the packet associated with this
    PacketMetadata object.
    """
    return pkt_mdat.sec * 1e6 + pkt_mdat.usec


def parse_pcap(flp, out_dir, rtt_window):
    """Parse a PCAP file.

    Writes to a npz file that contains the sequence number and queue occupency
    percentage in the last RTT
    """
    print(f"Parsing: {flp}")
    # E.g., 64Mbps-40000us-100p-1unfair-8other-1380B-2s-1-1.pcap
    bw_Mbps, rtt_us, queue_p, unfair_flows, other_flows, packet_size_B, dur_s, _, _, = path.basename(flp).split("-")
    # Link bandwidth (Mbps).
    bw_Mbps = float(bw_Mbps[:-4])
    # RTT (us).
    rtt_us = float(rtt_us[:-2])
    # Queue size (packets).
    queue_p = float(queue_p[:-1])
    # Experiment duration (s).
    dur_s = float(dur_s[:-1])
    # Packet size (bytes)
    packet_size_B = float(packet_size_B[:-1])
    # Number of unfair flows
    unfair_flows = int(unfair_flows[:-6])
    # Number of other flows
    other_flows = int(other_flows[:-5])

    flow_packet_count = dict()
    output_array = dict()

    for i in range(unfair_flows):
        flow_packet_count[SPORT_OFFSET + i] = 0

    for i in range(unfair_flows):
        output_array[SPORT_OFFSET + i] = np.empty([0, 2], dtype=float)


    pkts = [
        (pkt_mdat, pkt) for pkt_mdat, pkt in [
            # Parse each packet as a PPP packet.
            (pkt_mdat, scapy.layers.ppp.PPP(pkt_dat))
            for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)]
        # Select only IP/TCP packets sent from SRC_IP.
        if (scapy.layers.inet.IP in pkt and
            scapy.layers.inet.TCP in pkt and
            pkt[scapy.layers.inet.IP].src == SRC_IP and
            pkt_mdat.wirelen >= packet_size_B) # Ignore non-data packets
    ]

    window_start = 0

    for i in range(len(pkts)):
        port = pkts[i][1][scapy.layers.inet.TCP].sport

        if (port - SPORT_OFFSET < unfair_flows):
            cur_time_us = parse_time_us(pkts[i][0])
            flow_packet_count[port] += 1

            # Move window_start to less than rtt_window
            while (cur_time_us - parse_time_us(pkts[window_start][0]) > rtt_window * rtt_us):
                packet_port = pkts[window_start][1][scapy.layers.inet.TCP].sport
                if (packet_port - SPORT_OFFSET < unfair_flows):
                    flow_packet_count[packet_port] -= 1
                window_start += 1

            # Get queue occupency
            queue_occupency = flow_packet_count[port] / float(i - window_start + 1)

            # Append this record to numpy array
            output_array[port] = np.vstack([output_array[port], [pkts[i][1][scapy.layers.inet.TCP].seq, queue_occupency]])

    for i in range(unfair_flows):
        print("Saving " + out_dir + "/" + path.basename(flp)[:-9] + "-" + 
            str(rtt_window) + "rttW-" + str(i+1) + "flowNum-fairness.npz")
        np.savez_compressed(out_dir + "/" + path.basename(flp)[:-9] + "-" + 
            str(rtt_window) + "rttW-" + str(i+1) + "flowNum-fairness.npz", 
            output_array[SPORT_OFFSET + i])


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
        help='Size of the RTT window to calculate receiving rate and loss rate (default: 2 * RTT)')
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir
    rtt_window = args.rtt_window

    # Determine which configurations failed.
    fals = []
    fals_flp = path.join(exp_dir, "failed.json")
    if path.exists(fals_flp):
        with open(fals_flp, "r") as fil:
            fals = json.load(fil)

    # Select only PCAP files (".pcap") that were captured at the router's output
    # ("-1-1") and were not the result of a failed experiment.
    pcaps = [(path.join(exp_dir, fln), out_dir, rtt_window)
             for fln in os.listdir(exp_dir)
             if (fln.endswith("-1-1.pcap") and not failed(fln, fals))]
    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    with multiprocessing.Pool() as pol:
        pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()

