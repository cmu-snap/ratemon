#! /usr/bin/env python3

from os import path
import argparse
import numpy as np
from matplotlib import pyplot

def main():
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Visualize a simulation's features.")
    psr.add_argument(
        "--parsed-data", help="The path to the parsed simulation data.",
        required=True, type=str)
    psr.add_argument(
        "--out-dir", default=".",
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()
    dat_flp = args.parsed_data
    out_dir = args.out_dir
    assert path.exists(dat_flp), f"File does not exist: {dat_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    dat = np.load(dat_flp)
    num_unfair = len(dat.files)
    assert num_unfair == 1, \
        ("This script supports simulations with a single unfair flow only, but "
         f"the provided simulation contains {num_unfair} unfair flows!")
    dat = dat[dat.files[0]]

    # determine fair queue occupancy
    toks = dat_flp.split("-")
    if dat_flp.endswith(".npz"):
        toks[-1] = toks[-1][:-4]
    (bw_Mbps, btl_delay_us, queue_p, unfair_flws, other_flws, edge_delays,
     payload_B, dur_s) = toks

    queue_fair_occupancy = 1 / (1.0 * (int(unfair_flws[:-6]) + int(other_flws[:-5])))

    for fet in dat.dtype.names:
        if fet != "arrival time us":
            pyplot.plot(dat["arrival time us"], np.where(dat[fet] ==-1, np.nan, dat[fet]))
            pyplot.xlabel("Arrival time")
            pyplot.ylabel(fet)
            pyplot.tight_layout()

            pyplot.ylim(bottom=-0.1)

            if "queue" in fet:
                pyplot.ylim(top=1.1)
                pyplot.hlines(queue_fair_occupancy, 0, dat["arrival time us"][-1], colors='k', linestyles='dashdot')

            if "mathis model label" in fet:
                pyplot.ylim(top=1.1)

            # Replace name
            fet = fet.replace(" ", "_")
            fet = fet.replace("/", "-")
            pyplot.savefig(path.join(out_dir, f"Arrival_time_vs_{fet}.pdf"))
            pyplot.close()


if __name__ == "__main__":
    main()