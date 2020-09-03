#! /usr/bin/env python3
""" Graph all of the features from a single simulation. """

import argparse
import os
from os import path

from matplotlib import pyplot
import numpy as np

import cl_args
import utils


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Visualize a simulation's features.")
    psr.add_argument(
        "--parsed-data", help="The path to the parsed simulation data.",
        required=True, type=str)
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    dat_flp = args.parsed_data
    out_dir = args.out_dir
    assert path.exists(dat_flp), f"File does not exist: {dat_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    with np.load(dat_flp) as fil:
        num_unfair = len(fil.files)
        assert num_unfair == 1, \
            ("This script supports simulations with a single unfair flow only, "
             f"but the provided simulation contains {num_unfair} unfair flows!")
        dat = fil[fil.files[0]]

    sim = utils.Sim(dat_flp)
    queue_fair_occupancy = 1 / (sim.unfair_flws + sim.other_flws)

    for fet in dat.dtype.names:
        if fet == "arrival time us":
            continue
        print(f"Plotting feature: {fet}")
        pyplot.plot(
            dat["arrival time us"],
            np.where(dat[fet] == -1, np.nan, dat[fet]))
        pyplot.xlabel("arrival time (us)")
        pyplot.ylabel(fet)

        # Adjust plot limits.
        pyplot.ylim(bottom=0)
        if "queue" in fet:
            pyplot.ylim(top=1.1)
            pyplot.hlines(
                queue_fair_occupancy, 0, dat["arrival time us"][-1],
                colors="k", linestyles="dashdot")
        if ("mathis model label" in fet or "loss" in fet) and "sqrt" not in fet:
            pyplot.ylim(top=1.1)

        pyplot.tight_layout()
        pyplot.savefig(path.join(
            out_dir,
            ("arrival_time_us_vs_"
             f"{fet.replace(' ', '_').replace('/', '-')}.pdf")))
        pyplot.close()


if __name__ == "__main__":
    main()
