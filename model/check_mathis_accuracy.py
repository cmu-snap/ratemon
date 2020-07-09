#! /usr/bin/env python3
"""
Tests the accuracy of the Mathis Model in determining whether a flow is behaving
 fairly.
"""

import argparse
import os
from os import path

import numpy as np

import utils


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        ("Checks the accuracy of the Mathis Model in predicting a flow's "
         "fairness. Uses the output of parse_dumbbell.py."))
    psr.add_argument(
        "--exp-dir",
        help=("The directory in which the experiment results are stored "
              "(required)."), required=True, type=str)
    args = psr.parse_args()
    exp_dir = args.exp_dir

    correct = 0
    total = 0
    for flp in os.listdir(exp_dir):
        dat = np.load(path.join(exp_dir, flp))
        dat = dat[dat.files[0]]
        sim = utils.Sim(flp)
        correct += (
            (dat["mathis model label"] ==
             # To determine the ground truth, compare the actual queue
             # occupancy to the fair queue occupancy (convert to ints to
             # generate class labels).
             (dat["queue occupancy ewma-alpha0.5"] > (
                 1 / (sim.unfair_flws + sim.other_flws))).astype(int)
             # Count the number of correct predictions by converting
             # from bools to ints and summing them up.
            ).astype(int).sum())
        total += dat.shape[0]
    print(f"Mathis Model accuracy: {(correct / total) * 100:.2f}%")


if __name__ == "__main__":
    main()
