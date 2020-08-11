#! /usr/bin/env python3
""" Evaluates feature correlation. """

import argparse
import multiprocessing
import os
from os import path
import random
import shutil

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import models
import train
import utils

# Whether to parse simulation files synchronously or in parallel.
SYNC = False
# Features to analyze.
ALL_FETS = sorted([
    # "1/sqrt loss event rate-windowed-minRtt1",
    # "1/sqrt loss event rate-windowed-minRtt1024",
    # "1/sqrt loss event rate-windowed-minRtt128",
    # "1/sqrt loss event rate-windowed-minRtt16",
    # "1/sqrt loss event rate-windowed-minRtt2",
    # "1/sqrt loss event rate-windowed-minRtt256",
    # "1/sqrt loss event rate-windowed-minRtt32",
    # "1/sqrt loss event rate-windowed-minRtt4",
    # "1/sqrt loss event rate-windowed-minRtt512",
    # "1/sqrt loss event rate-windowed-minRtt64",
    # "1/sqrt loss event rate-windowed-minRtt8",
    # "RTT estimate ratio-ewma-alpha0.001",
    # "RTT estimate ratio-ewma-alpha0.002",
    # "RTT estimate ratio-ewma-alpha0.003",
    # "RTT estimate ratio-ewma-alpha0.004",
    # "RTT estimate ratio-ewma-alpha0.005",
    # "RTT estimate ratio-ewma-alpha0.006",
    # "RTT estimate ratio-ewma-alpha0.007",
    # "RTT estimate ratio-ewma-alpha0.008",
    # "RTT estimate ratio-ewma-alpha0.009",
    # "RTT estimate ratio-ewma-alpha0.01",
    # "RTT estimate ratio-ewma-alpha0.1",
    # "RTT estimate ratio-ewma-alpha0.2",
    # "RTT estimate ratio-ewma-alpha0.3",
    # "RTT estimate ratio-ewma-alpha0.4",
    "RTT estimate ratio-ewma-alpha0.5",
    # "RTT estimate ratio-ewma-alpha0.6",
    # "RTT estimate ratio-ewma-alpha0.7",
    # "RTT estimate ratio-ewma-alpha0.8",
    # "RTT estimate ratio-ewma-alpha0.9",
    # "RTT estimate ratio-ewma-alpha1.0",
    # "RTT estimate us-ewma-alpha0.001",
    # "RTT estimate us-ewma-alpha0.002",
    # "RTT estimate us-ewma-alpha0.003",
    # "RTT estimate us-ewma-alpha0.004",
    # "RTT estimate us-ewma-alpha0.005",
    # "RTT estimate us-ewma-alpha0.006",
    # "RTT estimate us-ewma-alpha0.007",
    # "RTT estimate us-ewma-alpha0.008",
    # "RTT estimate us-ewma-alpha0.009",
    # "RTT estimate us-ewma-alpha0.01",
    # "RTT estimate us-ewma-alpha0.1",
    # "RTT estimate us-ewma-alpha0.2",
    # "RTT estimate us-ewma-alpha0.3",
    # "RTT estimate us-ewma-alpha0.4",
    "RTT estimate us-ewma-alpha0.5",
    # "RTT estimate us-ewma-alpha0.6",
    # "RTT estimate us-ewma-alpha0.7",
    # "RTT estimate us-ewma-alpha0.8",
    # "RTT estimate us-ewma-alpha0.9",
    # "RTT estimate us-ewma-alpha1.0",
    # "arrival time us",
    # "average RTT estimate ratio-windowed-minRtt1",
    # "average RTT estimate ratio-windowed-minRtt1024",
    # "average RTT estimate ratio-windowed-minRtt128",
    # "average RTT estimate ratio-windowed-minRtt16",
    # "average RTT estimate ratio-windowed-minRtt2",
    # "average RTT estimate ratio-windowed-minRtt256",
    # "average RTT estimate ratio-windowed-minRtt32",
    # "average RTT estimate ratio-windowed-minRtt4",
    # "average RTT estimate ratio-windowed-minRtt512",
    # "average RTT estimate ratio-windowed-minRtt64",
    # "average RTT estimate ratio-windowed-minRtt8",
    # "average RTT estimate us-windowed-minRtt1",
    # "average RTT estimate us-windowed-minRtt1024",
    # "average RTT estimate us-windowed-minRtt128",
    # "average RTT estimate us-windowed-minRtt16",
    # "average RTT estimate us-windowed-minRtt2",
    # "average RTT estimate us-windowed-minRtt256",
    # "average RTT estimate us-windowed-minRtt32",
    # "average RTT estimate us-windowed-minRtt4",
    # "average RTT estimate us-windowed-minRtt512",
    # "average RTT estimate us-windowed-minRtt64",
    # "average RTT estimate us-windowed-minRtt8",
    # "average interarrival time us-windowed-minRtt1",
    # "average interarrival time us-windowed-minRtt1024",
    # "average interarrival time us-windowed-minRtt128",
    # "average interarrival time us-windowed-minRtt16",
    # "average interarrival time us-windowed-minRtt2",
    # "average interarrival time us-windowed-minRtt256",
    # "average interarrival time us-windowed-minRtt32",
    # "average interarrival time us-windowed-minRtt4",
    # "average interarrival time us-windowed-minRtt512",
    # "average interarrival time us-windowed-minRtt64",
    # "average interarrival time us-windowed-minRtt8",
    # "average throughput p/s-windowed-minRtt1",
    # "average throughput p/s-windowed-minRtt1024",
    # "average throughput p/s-windowed-minRtt128",
    # "average throughput p/s-windowed-minRtt16",
    # "average throughput p/s-windowed-minRtt2",
    # "average throughput p/s-windowed-minRtt256",
    # "average throughput p/s-windowed-minRtt32",
    # "average throughput p/s-windowed-minRtt4",
    # "average throughput p/s-windowed-minRtt512",
    # "average throughput p/s-windowed-minRtt64",
    # "average throughput p/s-windowed-minRtt8",
    # "interarrival time us-ewma-alpha0.001",
    # "interarrival time us-ewma-alpha0.002",
    # "interarrival time us-ewma-alpha0.003",
    # "interarrival time us-ewma-alpha0.004",
    # "interarrival time us-ewma-alpha0.005",
    # "interarrival time us-ewma-alpha0.006",
    # "interarrival time us-ewma-alpha0.007",
    # "interarrival time us-ewma-alpha0.008",
    # "interarrival time us-ewma-alpha0.009",
    # "interarrival time us-ewma-alpha0.01",
    # "interarrival time us-ewma-alpha0.1",
    # "interarrival time us-ewma-alpha0.2",
    # "interarrival time us-ewma-alpha0.3",
    # "interarrival time us-ewma-alpha0.4",
    "interarrival time us-ewma-alpha0.5",
    # "interarrival time us-ewma-alpha0.6",
    # "interarrival time us-ewma-alpha0.7",
    # "interarrival time us-ewma-alpha0.8",
    # "interarrival time us-ewma-alpha0.9",
    # "interarrival time us-ewma-alpha1.0",
    # "loss event rate-windowed-minRtt1",
    # "loss event rate-windowed-minRtt1024",
    # "loss event rate-windowed-minRtt128",
    # "loss event rate-windowed-minRtt16",
    # "loss event rate-windowed-minRtt2",
    # "loss event rate-windowed-minRtt256",
    # "loss event rate-windowed-minRtt32",
    # "loss event rate-windowed-minRtt4",
    # "loss event rate-windowed-minRtt512",
    # "loss event rate-windowed-minRtt64",
    # "loss event rate-windowed-minRtt8",
    # "loss rate estimate-ewma-alpha0.001",
    # "loss rate estimate-ewma-alpha0.002",
    # "loss rate estimate-ewma-alpha0.003",
    # "loss rate estimate-ewma-alpha0.004",
    # "loss rate estimate-ewma-alpha0.005",
    # "loss rate estimate-ewma-alpha0.006",
    # "loss rate estimate-ewma-alpha0.007",
    # "loss rate estimate-ewma-alpha0.008",
    # "loss rate estimate-ewma-alpha0.009",
    # "loss rate estimate-ewma-alpha0.01",
    # "loss rate estimate-ewma-alpha0.1",
    # "loss rate estimate-ewma-alpha0.2",
    # "loss rate estimate-ewma-alpha0.3",
    # "loss rate estimate-ewma-alpha0.4",
    "loss rate estimate-ewma-alpha0.5",
    # "loss rate estimate-ewma-alpha0.6",
    # "loss rate estimate-ewma-alpha0.7",
    # "loss rate estimate-ewma-alpha0.8",
    # "loss rate estimate-ewma-alpha0.9",
    # "loss rate estimate-ewma-alpha1.0",
    # "loss rate estimate-windowed-minRtt1",
    # "loss rate estimate-windowed-minRtt1024",
    # "loss rate estimate-windowed-minRtt128",
    # "loss rate estimate-windowed-minRtt16",
    # "loss rate estimate-windowed-minRtt2",
    # "loss rate estimate-windowed-minRtt256",
    # "loss rate estimate-windowed-minRtt32",
    # "loss rate estimate-windowed-minRtt4",
    # "loss rate estimate-windowed-minRtt512",
    # "loss rate estimate-windowed-minRtt64",
    # "loss rate estimate-windowed-minRtt8",
    # "mathis model label-ewma-alpha0.001",
    # "mathis model label-ewma-alpha0.002",
    # "mathis model label-ewma-alpha0.003",
    # "mathis model label-ewma-alpha0.004",
    # "mathis model label-ewma-alpha0.005",
    # "mathis model label-ewma-alpha0.006",
    # "mathis model label-ewma-alpha0.007",
    # "mathis model label-ewma-alpha0.008",
    # "mathis model label-ewma-alpha0.009",
    # "mathis model label-ewma-alpha0.01",
    # "mathis model label-ewma-alpha0.1",
    # "mathis model label-ewma-alpha0.2",
    # "mathis model label-ewma-alpha0.3",
    # "mathis model label-ewma-alpha0.4",
    "mathis model label-ewma-alpha0.5",
    # "mathis model label-ewma-alpha0.6",
    # "mathis model label-ewma-alpha0.7",
    # "mathis model label-ewma-alpha0.8",
    # "mathis model label-ewma-alpha0.9",
    # "mathis model label-ewma-alpha1.0",
    # "mathis model label-windowed-minRtt1",
    # "mathis model label-windowed-minRtt1024",
    # "mathis model label-windowed-minRtt128",
    # "mathis model label-windowed-minRtt16",
    # "mathis model label-windowed-minRtt2",
    # "mathis model label-windowed-minRtt256",
    # "mathis model label-windowed-minRtt32",
    # "mathis model label-windowed-minRtt4",
    # "mathis model label-windowed-minRtt512",
    # "mathis model label-windowed-minRtt64",
    # "mathis model label-windowed-minRtt8",
    # "mathis model throughput p/s-ewma-alpha0.001",
    # "mathis model throughput p/s-ewma-alpha0.002",
    # "mathis model throughput p/s-ewma-alpha0.003",
    # "mathis model throughput p/s-ewma-alpha0.004",
    # "mathis model throughput p/s-ewma-alpha0.005",
    # "mathis model throughput p/s-ewma-alpha0.006",
    # "mathis model throughput p/s-ewma-alpha0.007",
    # "mathis model throughput p/s-ewma-alpha0.008",
    # "mathis model throughput p/s-ewma-alpha0.009",
    # "mathis model throughput p/s-ewma-alpha0.01",
    # "mathis model throughput p/s-ewma-alpha0.1",
    # "mathis model throughput p/s-ewma-alpha0.2",
    # "mathis model throughput p/s-ewma-alpha0.3",
    # "mathis model throughput p/s-ewma-alpha0.4",
    "mathis model throughput p/s-ewma-alpha0.5",
    # "mathis model throughput p/s-ewma-alpha0.6",
    # "mathis model throughput p/s-ewma-alpha0.7",
    # "mathis model throughput p/s-ewma-alpha0.8",
    # "mathis model throughput p/s-ewma-alpha0.9",
    # "mathis model throughput p/s-ewma-alpha1.0",
    # "mathis model throughput p/s-windowed-minRtt1",
    # "mathis model throughput p/s-windowed-minRtt1024",
    # "mathis model throughput p/s-windowed-minRtt128",
    # "mathis model throughput p/s-windowed-minRtt16",
    # "mathis model throughput p/s-windowed-minRtt2",
    # "mathis model throughput p/s-windowed-minRtt256",
    # "mathis model throughput p/s-windowed-minRtt32",
    # "mathis model throughput p/s-windowed-minRtt4",
    # "mathis model throughput p/s-windowed-minRtt512",
    # "mathis model throughput p/s-windowed-minRtt64",
    # "mathis model throughput p/s-windowed-minRtt8",
    "min RTT us",
    # "seq",
    # "throughput p/s-ewma-alpha0.001",
    # "throughput p/s-ewma-alpha0.002",
    # "throughput p/s-ewma-alpha0.003",
    # "throughput p/s-ewma-alpha0.004",
    # "throughput p/s-ewma-alpha0.005",
    # "throughput p/s-ewma-alpha0.006",
    # "throughput p/s-ewma-alpha0.007",
    # "throughput p/s-ewma-alpha0.008",
    # "throughput p/s-ewma-alpha0.009",
    # "throughput p/s-ewma-alpha0.01",
    # "throughput p/s-ewma-alpha0.1",
    # "throughput p/s-ewma-alpha0.2",
    # "throughput p/s-ewma-alpha0.3",
    # "throughput p/s-ewma-alpha0.4",
    "throughput p/s-ewma-alpha0.5",
    # "throughput p/s-ewma-alpha0.6",
    # "throughput p/s-ewma-alpha0.7",
    # "throughput p/s-ewma-alpha0.8",
    # "throughput p/s-ewma-alpha0.9",
    # "throughput p/s-ewma-alpha1.0"
])


def run_cnfs(fets, args, sims):
    """ Trains a model for each provided configuration. """
    # Assemble configurations.
    cnfs = [
        {**vars(args), "features": fets_, "sims": sims, "sync": True,
         "out_dir": path.join(args.out_dir, subdir),
         "tmp_dir": path.join("/tmp", subdir)}
        for fets_, subdir in zip(
            fets,
            # Create a subdirectory name for each list of features.
            [",".join([
                str(fet).replace(" ", "_").replace("/", "p")
                for fet in fets_])
             for fets_ in fets])]
    # Train configurations.
    if SYNC:
        res = [train.run_trials(cnf) for cnf in cnfs]
    else:
        with multiprocessing.Pool(processes=4) as pol:
            res = pol.map(train.run_trials, cnfs)
    # Remove temporary subdirs.
    for cnf in cnfs:
        try:
            shutil.rmtree(cnf["tmp_dir"])
        except FileNotFoundError:
            pass
    # Note that accuracy = 1 - loss.
    return dict(zip(
        [tuple(cnf["features"]) for cnf in cnfs], 1 - np.array(res)))


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
        "--warmup-percent", default=train.DEFAULTS["warmup_percent"],
        help=("The percent of each simulation's datapoint to drop from the "
              "beginning."),
        type=float)
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
    # all_fets = sorted(models.MODELS[args.model].in_spc)
    all_fets = ALL_FETS

    # x-axis features.
    fets_x = list(reversed(all_fets))
    # y-axis features.
    fets_y = all_fets

    out_dir = args.out_dir
    dat_flp = path.join(out_dir, "correlation.npz")
    if path.exists(dat_flp):
        # Load existing results.
        print(f"Found existing data: {dat_flp}")
        with np.load(dat_flp) as fil:
            accs_ratios = fil[fil.files[0]]
    else:
        # Create the list of simulations here instead of letting the
        # training script do it so that all runs use the same
        # simulations.
        dat_dir = args.data_dir
        sims = [path.join(dat_dir, sim) for sim in os.listdir(dat_dir)]
        if train.SHUFFLE:
            # Set the random seed so that multiple instances of this
            # script see the same random order.
            random.seed(utils.SEED)
            random.shuffle(sims)
        num_sims = args.num_sims
        if num_sims is not None:
            num_sims_actual = len(sims)
            assert num_sims_actual >= num_sims, \
                (f"Insufficient simulations. Requested {num_sims}, but only "
                 f"{num_sims_actual} availabled.")
            sims = sims[:num_sims]

        # Train models.
        accs_single = run_cnfs([[fet] for fet in all_fets], args, sims)
        accs_pairs = run_cnfs(
            [[fet1, fet2]
             for i, fet1 in enumerate(all_fets)
             for j, fet2 in enumerate(all_fets)
             # Do not consider pairs of the same feature.
             if i != j],
            args, sims)
        # Calculate the accuracy ratios.
        accs_ratios = np.array(
            [[(accs_pairs[(fet1, fet2)] / accs_single[(fet1,)]
               if (fet1, fet2) in accs_pairs else 0)
              for fet2 in fets_x]
             for fet1 in fets_y])
        # Save results.
        np.savez_compressed(dat_flp, accs_ratios=accs_ratios)
        print(f"Saving results: {dat_flp}")

    # Graph results.
    plt.subplots(figsize=(8, 7))
    with sns.axes_style("white"):
        sns.heatmap(
            accs_ratios, linewidth=0.5, center=1, xticklabels=fets_x,
            yticklabels=fets_y, square=True, annot=True, fmt=".2f",
            annot_kws={"fontsize":8})
    plt.tight_layout()
    out_flp = path.join(out_dir, "correlation.pdf")
    print(f"Saving graph: {out_flp}")
    plt.savefig(out_flp)


if __name__ == "__main__":
    main()
