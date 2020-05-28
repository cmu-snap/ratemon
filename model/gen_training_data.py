#! /usr/bin/env python3
"""Runs experiments to generate training data. """

import argparse
import itertools
import logging
import os
from os import path
import time

import sim


# Bandwidth (Mbps).
BW_MIN_Mbps = 4
BW_MAX_Mbps = 50
BW_DELTA_Mbps = 2
BWS_Mbps = range(BW_MIN_Mbps, BW_MAX_Mbps + 1, BW_DELTA_Mbps)
# Link delay (us).
DELAY_MIN_us = 1000
DELAY_MAX_us = 20000
DELAY_DELTA_us = 1000
DELAYS_us = range(DELAY_MIN_us, DELAY_MAX_us + 1, DELAY_DELTA_us)
# Router queue size (BDP).
QUEUE_p = range(1, 32, 2)  # 1 to 32 BDP
# Number of other flows
OTHER_FLOWS = range(1, 11)  # 1 to 10 non-BBR flows
# Packet size (bytes)
PACKET_SIZE_B = 1380
# Simulation duration (s).
DUR_s = 80
# Delay until ACK pacing begins.
WARMUP_s = 60
# Whether to enable unfairness mitigation.
ENABLE = False
# Whether to capture pcap traces.
PCAP = True
# Whether to capture csv files.
CSV = True
# The number of BBR flows.
UNFAIR_FLOWS = 1
# The protocol to use for the non-BBR flows.
OTHER_PROTO = "ns3::TcpNewReno"
# Whether to return before running experiments.
DRY_RUN = False
# Whether to run the simulations synchronously or in parallel.
SYNC = False
# Default destination for email updates.
EMAIL_DST = "c.canel@icloud.com"
# Log level.
LOG_LVL = "INFO"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]


def bdp_bps(bw_Mbps, one_way_dly_us):
    """ Calculates the BDP in bits per second. """
    return (bw_Mbps / 8. * 1e6) * (one_way_dly_us / 1e6)


def main():
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Generates training data.")
    psr.add_argument(
        "--out-dir",
        help="The directory in which to store output files (required).",
        required=True, type=str)
    psr.add_argument(
        "--log-dst", default=EMAIL_DST,
        help="The email address to which updates will be sent.", type=str)
    args = psr.parse_args()
    # The ID of the experiment.
    eid = str(round(time.time()))
    # Create a new output directory based on the current time.
    out_dir = path.join(args.out_dir, eid)
    # For organization purposes, store the pcap files in a subdirectory.
    sim_dir = path.join(out_dir, "sim")
    # This also creates out_dir.
    os.makedirs(sim_dir)

    # Set up logging.
    numeric_level = getattr(logging, LOG_LVL.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {LOG_LVL}")
    logging.basicConfig(level=numeric_level)
    log = logging.getLogger(LOGGER)

    # Assemble the configurations.
    cnfs = [{"bandwidth_Mbps": bw_Mbps,
             "delay_us": dly_us,
             # Calculate queue capacity as a multiple of the BDP. If the BDP is
             # less than a single packet, then use 1 packet as the BDP anyway.
             "queue_capacity_p": int(round(
                 que_p *
                 max(1, bdp_bps(bw_Mbps, dly_us * 2) / float(PACKET_SIZE_B)))),
             "experiment_duration_s": DUR_s,
             "unfair_flows": UNFAIR_FLOWS,
             "other_flows": flws,
             "other_proto": OTHER_PROTO,
             "enable" : "true" if ENABLE else "false",
             "pcap": "true" if PCAP else "false",
             "csv": "true" if CSV else "false",
             "packet_size": PACKET_SIZE_B,
             "out_dir": sim_dir}
            for bw_Mbps, dly_us, que_p, flws in itertools.product(
                BWS_Mbps, DELAYS_us, QUEUE_p, OTHER_FLOWS)]
    sim.sim(eid, cnfs, out_dir, log_par=LOGGER, log_dst=args.log_dst,
            dry_run=DRY_RUN, sync=SYNC)

    log.info(f"Results in: {out_dir}")
    log.critical("Finished.")


if __name__ == "__main__":
    main()
