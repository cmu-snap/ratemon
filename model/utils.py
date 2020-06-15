#! /usr/bin/env python3
"""Utility functions for parse_*.py """

from os import path

import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp

def parse_filename(flp):
    """
    Returns experiment setup in filename
    """
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

def parse_packets_endpoint(flp, packet_size_B):
    """
    Takes in a file path and return (seq, timestamp)
    """
    # Not using parse_time_us for efficiency purpose
    return [
        (scapy.layers.ppp.PPP(pkt_dat)[scapy.layers.inet.TCP].seq, pkt_mdat.sec * 1e6 + pkt_mdat.usec)
            for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Select only IP/TCP packets larger than or equal to packet size.
        if (pkt_mdat.wirelen >= packet_size_B) # Ignore non-data packets
        ]

def parse_packets_router(flp, packet_size_B):
    """
    Takes in a file path and return (sender, timestamp)
    """
    # Not using parse_time_us for efficiency purpose
    return [
        # Parse each packet as a PPP packet.
        (int(scapy.layers.ppp.PPP(pkt_dat)[scapy.layers.inet.IP].src.split(".")[2]), pkt_mdat.sec * 1e6 + pkt_mdat.usec)
            for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Select only IP/TCP packets larger than or equal to packet size.
        if (pkt_mdat.wirelen >= packet_size_B) # Ignore non-data packets
        ]


def scale(x, min_in, max_in, min_out, max_out):
    """
    Scales x, which is from the range [min_in, max_in], to the range
    [min_out, max_out].
    """
    assert min_in != max_in, "Divide by zero!"
    return min_out + (x - min_in) * (max_out - min_out) / (max_in - min_in)
