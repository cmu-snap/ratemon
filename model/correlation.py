#! /usr/bin/env python3
""" Evaluates feature correlation. """

import argparse
import itertools
import multiprocessing
from os import path

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import models
import train


def run_cnfs(cnfs):
    """ Trains a model for each provided configuration. """
    with multiprocessing.Pool() as pol:
        # Note that accuracy = 1 - loss.
        return dict(zip(
            (cnf["features"] for cnf in cnfs),
            1 - np.array(pol.starmap(train.run_many, cnfs))))


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Evaluates feature correlation.")
    psr.add_argument(
        "--data-dir",
        help=("The path to a directory containing the"
              "training/validation/testing data (required)."),
        required=True, type=str)
    psr.add_argument(
        "--warmup", default=train.DEFAULTS["warmup"],
        help=("The number of packets to drop from the beginning of each "
              "simulation."),
        type=int)
    psr.add_argument(
        "--num-sims", default=train.DEFAULTS["num_sims"],
        help="The number of simulations to consider.", type=int)
    psr.add_argument(
        "--model", choices=models.MODEL_NAMES, default=train.DEFAULTS["model"],
        help="The model to use.", type=str)
    psr.add_argument(
        "--kernel", default=train.DEFAULTS["kernel"],
        choices=["linear", "poly", "rbf", "sigmoid"],
        help=("If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type kernel. Ignored otherwise."),
        type=str)
    psr.add_argument(
        "--degree", default=train.DEFAULTS["degree"],
        help=("If the model is of type \"{models.SvmSklearnWrapper().name()}\" "
              "and \"--kernel=poly\", then this is the degree of the polynomial "
              "that will be fit. Ignored otherwise."),
        type=int)
    psr.add_argument(
        "--penalty", default=train.DEFAULTS["penalty"], choices=["l1", "l2"],
        help=(f"If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type of regularization. Ignored otherwise."))
    psr.add_argument(
        "--max-iter", default=train.DEFAULTS["max_iter"],
        help=("If the model is an sklearn model, then this is the maximum "
              "number of iterations to use during the fitting process. Ignored "
              "otherwise."),
        type=int)
    psr.add_argument(
        "--standardize", action="store_true",
        help=("Standardize the data so that it has a mean of 0 and a variance "
              "of 1. Otherwise, data will be rescaled to the range [0, 1]."))
    psr.add_argument(
        "--val-improvement-thresh", default=train.DEFAULTS["val_improvement_thresh"],
        help="Threshold for percept improvement in validation loss.",
        type=float)
    psr.add_argument(
        "--no-rand", action="store_true", help="Use a fixed random seed.")
    psr.add_argument(
        "--out-dir", default=train.DEFAULTS["out_dir"],
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()

    # Train models.
    all_fets = sorted(models.MODELS[args.model].in_spc)
    accs_single = run_cnfs(
        [{"features": fet, **vars(args)} for fet in all_fets])
    # To remove duplicates first create a set of all feature pairs,
    # where each pair is sorted.
    accs_pairs = run_cnfs([
        {"features": fets, **vars(args)} for fets in
        {tuple(sorted(fets))
         for fets in itertools.product(all_fets, all_fets)}])

    # Calculate the accuracy ratios.
    lbls_x = list(reversed(all_fets))
    lbls_y = all_fets
    # For a pair of features (fet1, fet2), only the pair sorted((fet1, fet2))
    # will be present in accs_pairs. For the missing "backwards" pair, record a
    # 0. These 0s will be masked out, below.
    accs_ratios = np.array(
        [[accs_pairs[(fet1, fet2)] / accs_single[fet1]
          if (fet1, fet2) in accs_pairs else 0
          for fet2 in lbls_x]
         for fet1 in lbls_y])

    # Graph results.
    mask = np.zeros_like(accs_ratios)
    # Mask out feature pairs that were duplicates.
    mask[np.triu_indices_from(mask, k=1)] = True
    f, ax = plt.subplots(figsize=(25, 20))
    with sns.axes_style("white"):
        ax = sns.heatmap(
            accs_ratios, mask=mask, linewidth=0.5, center=1, xticklabels=lbls_x,
            yticklabels=lbls_y, square=True, annot=True, fmt=".2f",
            annot_kws={"fontsize":5})
    plt.tight_layout()
    out_flp = path.join(args.out_dir, "out.pdf")
    plt.savefig(out_flp)
    print(f"Saved: {out_flp}")


if __name__ == "__main__":
    main()
