#! /usr/bin/env python3
"""
Tests the accuracy of the Mathis Model in determining whether a flow is behaving
 fairly.
"""

import argparse
import multiprocessing
import os
from os import path

import numpy as np

import utils


def process_one(flp):
    """ Process a single simulation. """
    dat = np.load(flp)
    dat = dat[dat.files[0]]
    labels = dat["mathis model label"]
    valid = np.where(labels != -1)
    sim = utils.Sim(flp)
    return (
        (
            (labels[valid] ==
             # To determine the ground truth, compare the actual queue
             # occupancy to the fair queue occupancy (convert to ints to
             # generate class labels).
             (dat["queue occupancy ewma-alpha0.5"][valid] > (
                 1 / (sim.unfair_flws + sim.fair_flws))).astype(int)
             # Count the number of correct predictions by converting
             # from bools to ints and summing them up.
            ).astype(int).sum()),
        len(valid[0]),
        dat.shape[0])


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
    sims = [path.join(exp_dir, sim) for sim in os.listdir(exp_dir)]
    print(f"Found {len(sims)} simulations.")

    with multiprocessing.Pool() as pol:
        res = pol.map(process_one, sims)

    correct, total, total_all = zip(*res)
    correct = sum(correct)
    total = sum(total)
    total_all = sum(total_all)
    print(f"Mathis Model accuracy: {(correct / total) * 100:.2f}%")
    print(f"Total: {total} packets")
    print(f"Discarded {total_all - total} packets.")
    print(
        "Accuracy without discarding packets: "
        f"{(correct / total_all) * 100:.2f}%")


if __name__ == "__main__":
    main()
