#! /usr/bin/env python3
"""Graph all of the features from a single experiment."""

import argparse
import multiprocessing
import os
from os import path

import cl_args
import features
import numpy as np
import utils
from matplotlib import pyplot as plt


def graph_fet(out_dir, dat, fet, bw_share_fair, bw_fair, x_min, x_max, labels):
    """Graphs a single feature."""
    if features.ARRIVAL_TIME_FET in fet:
        return

    print(f"Plotting feature: {fet}")
    for flw_dat in dat:
        plt.plot(
            flw_dat[features.ARRIVAL_TIME_FET],
            np.where(flw_dat[fet] == -1, np.nan, flw_dat[fet]),
            ".",
            markersize=0.5,
        )
        plt.xlabel(features.ARRIVAL_TIME_FET)
        plt.ylabel(fet)

    if features.TPUT_FET in fet and features.TOTAL_TPUT_FET not in fet:
        plt.hlines(bw_fair, 0, x_max, colors="k", linestyles="dashdot")
    elif features.TPUT_SHARE_FRAC_FET in fet:
        plt.hlines(bw_share_fair, 0, x_max, colors="k", linestyles="dashdot")

    plt.legend(labels, loc="upper left")
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(
        path.join(
            out_dir,
            (features.ARRIVAL_TIME_FET + "_vs_" + fet + ".pdf")
            .replace(" ", "_")
            .replace("/", "-"),
        )
    )
    plt.close()


def main():
    """This program's entrypoint."""
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Visualize a experiment's features.")
    psr.add_argument(
        "--parsed-data",
        help=(
            "The path to the parsed experiment data generated by " "gen_features.py."
        ),
        required=True,
        type=str,
    )
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    dat_flp = args.parsed_data
    out_dir = args.out_dir
    assert path.exists(dat_flp), f"File does not exist: {dat_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    with np.load(dat_flp) as fil:
        dat = [fil[flw] for flw in sorted(fil.files, key=int)]

    exp = utils.Exp(dat_flp)
    num_flws = exp.tot_flws
    found_flws = len(dat)
    assert num_flws == found_flws, (
        f"Experiment has {num_flws} flows, parsed data has {found_flws} "
        f"flows: {dat_flp}"
    )
    if found_flws == 0:
        print("No flows to graph.")
        return

    bw_share_fair = 1 / num_flws
    bw_fair = exp.bw_bps * bw_share_fair
    labels = [f"Flow {flw}" for flw in range(num_flws)]
    x_min = min(flw_dat[0][features.ARRIVAL_TIME_FET] for flw_dat in dat)
    x_max = max(flw_dat[-1][features.ARRIVAL_TIME_FET] for flw_dat in dat)
    fets = dat[0].dtype.names

    print(f"Plotting {len(fets)} features...")
    with multiprocessing.Pool() as pol:
        pol.starmap(
            graph_fet,
            (
                (out_dir, dat, fet, bw_share_fair, bw_fair, x_min, x_max, labels)
                for fet in fets
            ),
        )


if __name__ == "__main__":
    main()
