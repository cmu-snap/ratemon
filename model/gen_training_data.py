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
BWS_Mbps = list(range(BW_MIN_Mbps, BW_MAX_Mbps + 1, BW_DELTA_Mbps))
# Link delay (us).
DELAY_MIN_us = 100
DELAY_MAX_us = 1000
DELAY_DELTA_us = 50
DELAYS_us = list(range(DELAY_MIN_us, DELAY_MAX_us + 1, DELAY_DELTA_us))
# Router queue size (packets).
QUEUE = 100
# Simulation duration (s).
DUR_s = 20
# Delay until ACK pacing begins.
WARMUP_s = 5
# Whether to return before running experiments.
DRY_RUN = False
# Whether to capture pcap traces.
PCAP = True
# Whether to run the simulations synchronously or in parallel.
SYNC = False
# Default destination for email updates.
EMAIL_DST = "c.canel@icloud.com"
# Log level.
LOG_LVL = "INFO"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]


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
    cnfs = [{"bandwidth_Mbps": bw_Mbps, "delay_us": dly_us,
             "queue_capacity_p": QUEUE, "experiment_duration_s": DUR_s,
             "warmup_s": WARMUP_s, "pcap": "true" if PCAP else "false",
             "out_dir": sim_dir}
            for bw_Mbps, dly_us in itertools.product(BWS_Mbps, DELAYS_us)][:10]
    sim.sim(eid, cnfs, out_dir, log_par=LOGGER, log_dst=args.log_dst,
            dry_run=DRY_RUN, sync=SYNC)

    log.info(f"Results in: {out_dir}")
    log.critical("Finished.")


if __name__ == "__main__":
    main()
