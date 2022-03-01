#! /usr/bin/env python3
"""Runs experiments to generate training data. """

import argparse
import itertools
import logging
import os
from os import path
import time

import cl_args
import defaults
import sim
import utils


# Bandwidth (Mbps).
BW_MIN_Mbps = 10
BW_MAX_Mbps = 10
BW_DELTA_Mbps = 8
BWS_Mbps = list(range(BW_MIN_Mbps, BW_MAX_Mbps + 1, BW_DELTA_Mbps))
# Link delay (us).
DELAY_MIN_us = 6500
DELAY_MAX_us = 6500
DELAY_DELTA_us = 2000
DELAYS_us = list(range(DELAY_MIN_us, DELAY_MAX_us + 1, DELAY_DELTA_us))
# Router queue size (multiples of the BDP).
QUEUE_MULTS = [2**i / 4 for i in range(10)]
# The number of "unfair" flows.
UNFAIR_FLOWS = range(1, 2)
# Number of "fair" flows
FAIR_FLOWS = range(1, 2)
# Packet size (bytes)
PACKET_SIZE_B = 1380
# Simulation duration (s).
DUR_s = 240
# Delay until ACK pacing begins.
WARMUP_s = 60
# Whether to enable unfairness mitigation.
ENABLE_MITIGATION = False
# Whether to capture pcap traces.
PCAP = True
# Whether to capture csv files.
CSV = True
# The protool to use for the "unfair" flows.
UNFAIR_PROTO = "ns3::TcpNewReno"
# The protocol to use for the "fair" flows.
FAIR_PROTO = "ns3::TcpBbr"
# Whether to return before running experiments.
DRY_RUN = False
# Default destination for email updates.
EMAIL_DST = "c.canel@icloud.com"
# Log level.
LOG_LVL = "INFO"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Generates training data.")
    psr.add_argument(
        "--log-dst", default=EMAIL_DST,
        help="The email address to which updates will be sent.", type=str)
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
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

    log.info("Bandwidths (Mbps): %s", list(BWS_Mbps))
    log.info("Link delays (us): %s", list(DELAYS_us))
    log.info("Queue size (x BDP): %s", list(QUEUE_MULTS))
    log.info("Unfair proto: %s", UNFAIR_PROTO)
    log.info("Unfair flows: %s", list(UNFAIR_FLOWS))
    log.info("Fair proto: %s", FAIR_PROTO)
    log.info("Fair flows: %s", list(FAIR_FLOWS))
    log.info("Duration (s): %s", DUR_s)

    # Assemble the configurations.
    cnfs = [{"bottleneck_bandwidth_Mbps": bw_Mbps,
             "bottleneck_delay_us": dly_us,
             # Calculate queue capacity as a multiple of the BDP. If the BDP is
             # less than a single packet, then use 1 packet as the BDP anyway.
             "bottleneck_queue_p": int(round(
                 que_mult *
                 max(1,
                     utils.bdp_B(bw_Mbps * 1e6, dly_us / 1e6 * 6) / float(PACKET_SIZE_B)))),
             "unfair_flows": unfair_flws,
             "unfair_proto": UNFAIR_PROTO,
             "fair_flows": fair_flws,
             "fair_proto": FAIR_PROTO,
             "unfair_edge_delays_us": f"[{dly_us}]",
             "fair_edge_delays_us": f"[{dly_us}]",
             "payload_B": PACKET_SIZE_B,
             "enable_mitigation": "false",
             "duration_s": DUR_s,
             "pcap": "true" if PCAP else "false",
             "out_dir": sim_dir}
            for (bw_Mbps, dly_us, que_mult, unfair_flws,
                 fair_flws) in itertools.product(
                     BWS_Mbps, DELAYS_us, QUEUE_MULTS, UNFAIR_FLOWS,
                     FAIR_FLOWS)]
    sim.sim(eid, cnfs, out_dir, log_par=LOGGER, log_dst=args.log_dst,
            dry_run=DRY_RUN, sync=defaults.SYNC)

    log.info("Results in: %s", out_dir)
    log.critical("Finished.")


if __name__ == "__main__":
    main()
