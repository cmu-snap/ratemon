#! /usr/bin/env python3
"""Runs experiments to generate training data. """

import argparse
import itertools
import json
import logging
import os
from os import path
import re
import time

import matplotlib.pyplot as plt

import sim

# Bandwidth (Mbps).
BWS_Mbps = 10
# Link delay (us).
DELAYS_us = 20_000
# Ack period (us).
# Router queue size (packets).
QUEUE = 1000
# Simulation duration (s).
DUR_s = 300
# Delay until ACK pacing begins.
WARMUP_s = [100, DUR_s]

MODEL = "/home/ccanel/BBR-Receiver/model/780_final_submission/net.pth"
RECALC_us = 1000000
NUM_OTHER_FLOWS = list(range(1, 33))
USE_RENO = [True, False]
# Whether to return before running experiments.
DRY_RUN = False
# Whether to capture pcap traces.
PCAP = False
# Whether to run the simulations synchronously or in parallel.
SYNC = False
# Default destination for email updates.
EMAIL_DST = "c.canel@icloud.com"
# Log level.
LOG_LVL = "INFO"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]


def res_fnc(out):
    # Build the arguments array, run the simulation, and iterate over each line
    # in its output.
    for line in out:
        if line.startswith("Fairness"):
            match = re.search(r"Fairness: ([\d.]+)", line)
            assert match is not None, f"Improperly formed output line: {line}"
            return float(match.group(1))
    return None


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
    # The ID of this experiment.
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
    cnfs = [{
        "bandwidth_Mbps": BWS_Mbps,
        "delay_us": DELAYS_us,
        "queue_capacity_p": QUEUE,
        "experiment_duration_s": DUR_s,
        "warmup_s": warm,
        "pcap": "true" if PCAP else "false",
        "out_dir": sim_dir,
        "other_flows": num,
        "model": MODEL,
        "recalc_us": RECALC_us,
        "use_reno": "true" if reno else "false",
    } for num, warm, reno in itertools.product(NUM_OTHER_FLOWS, WARMUP_s, USE_RENO)][:1]
    data = sim.sim(eid, cnfs, out_dir, log_par=LOGGER, log_dst=args.log_dst,
                   dry_run=DRY_RUN, sync=SYNC)

    with open("data.json", "w") as f:
        json.dump(data, f)

    for reno in USE_RENO:
        _, ax = plt.subplots()
        str_reno = "true" if reno else "false"
        for warm in WARMUP_s:
            selected = [(k['other_flows'], v) for k, v in data
                        if k['warmup_s'] == warm and k['use_reno'] == str_reno]
            selected.sort(key=lambda x: x[0])
            xs, ys = zip(*selected)
            ax.plot(list(xs), list(ys), label=(
                "With" if warm < DUR_s else "Without") + " ACK pacing")
        flow = 'Reno' if reno else 'Cubic'
        ax.set_xlabel(f"Number of {flow} Flows")
        ax.set_ylabel("Jain's Fairness Index")
        ax.legend()
    plt.savefig(f"{flow}.pdf")


if __name__ == "__main__":
    main()
