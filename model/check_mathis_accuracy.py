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
    sims = os.listdir(exp_dir)
    print(f"Found {len(sims)} simulations.")

    correct = 0
    total = 0
    total_all = 0
    for flp in sims:
        dat = np.load(path.join(exp_dir, flp))
        dat = dat[dat.files[0]]
        sim = utils.Sim(flp)
        labels = dat["mathis model label"]
        valid = np.where(labels != -1)
        correct += (
            (labels[valid] ==
             # To determine the ground truth, compare the actual queue
             # occupancy to the fair queue occupancy (convert to ints to
             # generate class labels).
             (dat["queue occupancy ewma-alpha0.5"][valid] > (
                 1 / (sim.unfair_flws + sim.other_flws))).astype(int)
             # Count the number of correct predictions by converting
             # from bools to ints and summing them up.
            ).astype(int).sum())
        total += len(valid[0])
        total_all += dat.shape[0]
    print(f"Mathis Model accuracy: {(correct / total) * 100:.2f}%")
    print(f"Total: {total} packets")
    print(f"Discarded {total_all - total} packets.")
    print(f"Accuracy without discarding packets: {(correct / total_all) * 100:.2f}%")


if __name__ == "__main__":
    main()
