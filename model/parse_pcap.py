#! /usr/bin/env python3
"""Parses the output of gen_training_data.py. """

import argparse
import json
import multiprocessing
import os
from os import path
import time

from matplotlib import pyplot as plt
import numpy as np
import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp


# Whether to plot per-experiment average throughput.
DO_PLT = False
# The number of link segments on each round trip path.
LINKS_PER_RTT = 4
# The IP address of the sender.
SRC_IP = "10.1.1.1"
# The number of packets to consider when computing instantaneous bandwidth.
TPT_WINDOW = 5


def failed(fln, fals):
    # TODO: This does not work.
    for cnf in fals:
        if (f"{cnf['ack_period_us']}us-"
            f"{cnf['warmup_s']}s-"
            f"{cnf['bandwidth_Mbps']}Mbps-"
            f"{cnf['delay_us'] * 4}us-"
            f"{cnf['experiment_duration_s']}s") in fln:
            print(f"Discarding: {fln}")
            return True
    return False


def parse_time_s(pkt_mdat):
    """
    Returns the timestamp, in seconds, of the packet associated with this
    PacketMetadata object.
    """
    return pkt_mdat.sec + (pkt_mdat.usec / 1e6)


def avg_tpt_bps(pkts):
    """Calculates the average throughput over a list of packets.

    Each item in "pkts" should be a pair of: (PacketMetadata, packet). The
    return value is a pair of: (timestamp (s), throughput (bps)). The timestamp
    corresponds to the last packet in pkts.

    The temporal window is defined by the difference in time between the first
    and last packets. However, when computing the number of bytes of data, the
    first packet is discarded. This is because I believe that the timestamps are
    end times, meaning that the timestamp of the first packet is best thought of
    as the time at which the *second* packet *began* arriving.
    """
    tim_end_s = parse_time_s(pkts[-1][0])
    tim_srt_s = parse_time_s(pkts[0][0])
    return (
        # Use the time of the last packet as the time for this window.
        tim_end_s,
        # Sum the lengths of the TCP payloads, multiply by 8 to convert to bits,
        # and divide by the elapsed time. The TCP payload length is the total
        # length of the IP packet minus the lengths of the IP and TCP headers.
        (sum([(pkt[scapy.layers.inet.IP].len -
               pkt[scapy.layers.inet.IP].ihl * 4 -
               # When summing the number of bytes of data, discard the first
               # packet.
               pkt[scapy.layers.inet.TCP].dataofs * 4) for _, pkt in pkts[1:]])
         * 8 / (tim_end_s - tim_srt_s))
    )


def parse_pcap(flp, out_dir, do_plt=True):
    """Parse a PCAP file.

    Returns the parameters of the experiment that generated this trace as well
    as the average throughput before and after enabling ACK pacing. If do_plt is
    True, then also generates a graph of throughput over time.
    """
    print(f"Parsing: {flp}")
    # E.g., shark-5000us-5s-64Mbps-262144us-20s-2-0.pcap
    ack_per_us, wrm_s, bw_Mbps, dly_us, dur_s, _, _ = path.basename(
        flp).split("-")
    # Inter-ACK period (us).
    ack_per_us = float(ack_per_us[:-2])
    # Warmup duration before ACK pacing is enable (s).
    wrm_s = float(wrm_s[:-1])
    # Link bandwidth (Mbps).
    bw_Mbps = float(bw_Mbps[:-4])
    # Link delay (i.e., RTT / 4) (us).
    dly_us = float(dly_us[:-2])
    # Experiment duration (s).
    dur_s = float(dur_s[:-1])

    pkts = [
        (pkt_mdat, pkt) for pkt_mdat, pkt in [
            # Parse each packet as a PPP packet.
            (pkt_mdat, scapy.layers.ppp.PPP(pkt_dat))
            for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)]
        # Select only IP/TCP packets sent from SRC_IP.
        if (scapy.layers.inet.IP in pkt and
            scapy.layers.inet.TCP in pkt and
            pkt[scapy.layers.inet.IP].src == SRC_IP)
    ]

    if do_plt:
        # Slide a window across the packet stream and compute an average
        # throughput over each window. Each item is a pair of: (time (s),
        # throughput (bps)). Unzip these to create a list of xs and a list of
        # ys.
        xs, ys = zip(*[
            # Grow the window by one because we want the start time to be the
            # end time of the previous packet. See the docstring for
            # avg_tpt_bps().
            avg_tpt_bps(pkts[idx - ((TPT_WINDOW + 1) - 1):idx + 1])
            for idx in range((TPT_WINDOW + 1) - 1, len(pkts))])
        # Convert from bps to Mbps.
        ys = np.array(ys) / 1e6
        plt.plot(xs, ys, marker="o", fillstyle="none")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (Mbps)")
        plt.xlim(0, dur_s)
        plt.ylim(0, bw_Mbps)
        plt.savefig(path.join(
            out_dir,
            f"{ack_per_us}us-{wrm_s}s-{bw_Mbps}Mbps-{dly_us}us-{dur_s}s.pdf"))
        plt.close()

    return {
        "ack_period_us": ack_per_us,
        "warmup_s": wrm_s,
        "bandwidth_Mbps": bw_Mbps,
        "delay_us": dly_us,
        "rtt_us": dly_us * LINKS_PER_RTT,
        "experimant_duration_s": dur_s,
        # The average throughput (bps) while ACK pacing is disabled. Select only
        # the second item because we do not need first item, which is a
        # timestamp.
        "average_throughput_before_bps": avg_tpt_bps(
            [(pkt_mdat, pkt) for pkt_mdat, pkt in pkts
             # Select only packets from time [50% of warmup time, 95% of warmup
             # time). Start at 50% of the warmup time to allow the flow to ramp
             # up.
             if (wrm_s * 0.5) < parse_time_s(pkt_mdat) < (wrm_s * 0.95)])[1],
        # The average throughput (bps) while ACK pacing is disabled. Select only
        # the second item because we do not need first item, which is a
        # timestamp.
        "average_throughput_after_bps": avg_tpt_bps(
            [(pkt_mdat, pkt) for pkt_mdat, pkt in pkts
             # Select only packets from time (105% of warmup time, end).
             if parse_time_s(pkt_mdat) > (wrm_s * 1.05)])[1]
        }


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
    psr.add_argument(
        "--out-file", dest="out_flp",
        help="The path to the output file (required).", required=True, type=str)
    args = psr.parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir
    out_flp = args.out_flp

    # Determine which configurations failed.
    fals = []
    fals_flp = path.join(exp_dir, "failed.json")
    if path.exists(fals_flp):
        with open(fals_flp, "r") as fil:
            fals = json.load(fil)

    # Select only PCAP files (".pcap") that were captured at the receiver
    # ("-2-0") and were not the result of a failed experiment.
    pcaps = [(path.join(exp_dir, fln), out_dir, DO_PLT)
             for fln in os.listdir(exp_dir)
             if (fln.endswith("-2-0.pcap") and not failed(fln, fals))]
    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    with multiprocessing.Pool() as pol:
        ress = pol.starmap(parse_pcap, pcaps)
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")
    # Save results.
    with open(out_flp, "w") as fil:
        fil.write(json.dumps(ress, indent=4))
    print(f"Results in: {out_flp}")

    # Graphs results. Generate a converged throughput vs. ack period graph for
    # each configuration of bandwidth and delay.
    #
    # Returns a sorted list of all unique values for a particular key.
    get_all = lambda key: sorted(list({res[key] for res in ress}))
    for bw_Mbps in get_all("bandwidth_Mbps"):
        for dly_us in get_all("delay_us"):
            # Select only the results with this bandwidth and delay, and pick
            # the value of the ACK period as the x value. The output is a list
            # of (x, y) pairs. Finally, sort the results by x value.
            xys = sorted(
                [(res["ack_period_us"], res["average_throughput_after_bps"])
                 for res in ress
                 if (res["bandwidth_Mbps"] == bw_Mbps and
                     res["delay_us"] == dly_us)],
                key=lambda xy: xy[0])
            # If there are no results for this combination of bandwidth and
            # delay, then skip it.
            if not xys:
                continue
            # Split the (x, y) pairs into a list of xs and a list of ys.
            xs, ys = zip(*xys)
            # Convert from us to ms.
            xs = np.array(xs) / 1e3
            # Convert from bps to Mbps.
            ys = np.array(ys) / 1e6
            plt.plot(xs, ys)
            plt.xlabel("ACK delay (ms)")
            plt.ylabel("Converged throughput (Mb/s)")
            plt.ylim(bottom=0, top=bw_Mbps)
            plt.savefig(path.join(out_dir, f"{bw_Mbps}Mbps-{dly_us}us.pdf"))
            plt.close()


if __name__ == "__main__":
    main()
