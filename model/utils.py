#! /usr/bin/env python3
"""Utility functions for parse_*.py """

from os import path

def parse_filename(flp):
    bw_Mbps, btl_delay_us, queue_p, unfair_flows, other_flows, edge_delays, \
      packet_size_B, dur_s, _, _, = path.basename(flp).split("-")
    # Link bandwidth (Mbps).
    bw_Mbps = float(bw_Mbps[:-4])
    # Bottleneck router delay (us).
    btl_delay_us = float(btl_delay_us[:-2])
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
    # Edge delays
    edge_delays = list(map(int, edge_delays[:-2].split(",")))

    return (bw_Mbps, btl_delay_us, queue_p, dur_s, packet_size_B, unfair_flows, 
             other_flows, edge_delays)