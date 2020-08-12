#!/usr/bin/env python3
""" Visualizes sklearn training parameters. """

import argparse
import itertools
import multiprocessing
import os
from os import path
import random
import shutil
import time

import numpy as np
from matplotlib import pyplot

import models
import train
import utils


# Whether to parse simulation files synchronously or in parallel.
SYNC = False
# Experiment parameters.
#
# Number of simulations.
NUMS_SIMS = [1]
# NUMS_SIMS = [1, 10, 100, 1000, 10000]
# Percent of each simulation.
PRC_MIN = 100
PRC_MAX = 100
PRC_DELTA = 10
# PRC_MIN = 10
# PRC_MAX = 100
# PRC_DELTA = 10
PRCS = [100]
# PRCS = [10, 30, 60, 100]
# Number of iterations.
NUM_ITERS_MIN = 5
NUM_ITERS_MAX = 5
NUM_ITERS_DELTA = 15
# NUM_ITERS_MIN = 5
# NUM_ITERS_MAX = 150
# NUM_ITERS_DELTA = 15
NUMS_ITERS = [5]
# NUMS_ITERS = [5, 20, 65, 155]


def main():
    """ This program's entrypoint. """
    psr = argparse.ArgumentParser(
        description="Visualize sklearn training parameters.")
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
        "--standardize", action="store_true",
        help=("Standardize the data so that it has a mean of 0 and a variance "
              "of 1. Otherwise, data will be rescaled to the range [0, 1]."))
    psr.add_argument(
        "--no-rand", action="store_true", help="Use a fixed random seed.")
    psr.add_argument(
        "--out-dir", default=train.DEFAULTS["out_dir"],
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()
    args = train.prepare_args(vars(args))

    tim_srt_s = time.time()
    out_dir = args["out_dir"]
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    dat_flp = path.join(out_dir, "training_param_sweep.npz")
    if path.exists(dat_flp):
        # Load existing results.
        print(f"Found existing data: {dat_flp}")
        with np.load(dat_flp) as fil:
            ress = fil["results"]
    else:
        # The data processing here is tricky. Our goal is to have the
        # configurations all process the same data to make the
        # comparison as "apples-to-apples" as possible. To that end,
        # all of the configurations will process subsets of the same
        # set of simulations. The high-level strategy is to manually load and
        # process the data and then place it strategically so that the training
        # process reads it in automatically. The process is as follows:
        #    1) Determine the greatest number of simulations that any
        #       configuration will require and parse them.
        #    2) For each configuration, pick a subset of these simulations. When
        #       picking which simulations to use, always pick from the front.
        #       This means that configurations with more simulations will
        #       process the same simulations as those with fewer simulations,
        #       plus some extra. When picking which datapoints to use from
        #       within a simulation, choose them randomly.
        #    3) For each configuration, copy its data into its output directory
        #       so that it will be automatically detected and read in.

        # Load the required number of simulations.
        dat_dir = args["data_dir"]
        sims = [path.join(dat_dir, sim) for sim in os.listdir(dat_dir)]
        if train.SHUFFLE:
            # Set the random seed so that multiple instances of this
            # script see the same random order.
            random.seed(utils.SEED)
            random.shuffle(sims)
        num_sims_actual = len(sims)
        max_sims = max(NUMS_SIMS)
        assert num_sims_actual >= max_sims, \
            (f"Insufficient simulations. Requested {max_sims}, but only "
             f"{num_sims_actual} available.")
        sims = sims[:max_sims]
        net = models.MODELS[args["model"]]()
        sim_args = [
            (idx, max_sims, net, sim_flp, out_dir, args["warmup_percent"], 100)
            for idx, sim_flp in enumerate(sims)]
        if SYNC:
            dat_all = [train.process_sim(*sim_args_) for sim_args_ in sim_args]
        else:
            with multiprocessing.Pool() as pol:
                dat_all = pol.starmap(train.process_sim, sim_args)
        # Verify that we were able to parse all simulations. Normally,
        # we would allow the training process to proceed even if some
        # simulations failed to parse, but since in this case we are
        # looking at specific trends, we need to make sure that we are
        # training on the number of simulations that we intend.
        for dat in dat_all:
            assert dat is not None, \
                "Error processing at least one simulation. Check logs (above)."
        # Unpack the data.
        dat_all, sims = zip(*dat_all)
        dat_all = [utils.load_tmp_file(flp) for flp in dat_all]
        dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps = zip(*dat_all)
        dat_in = list(dat_in)
        dat_out = list(dat_out)
        dat_out_raw = list(dat_out_raw)
        dat_out_oracle = list(dat_out_oracle)
        scl_grps = list(scl_grps)

        # For each possible configuration of number of simulations and
        # percentage of each simulation, create a temporary file
        # containing the parsed data for that configuration.
        dat = {}
        all_prcs = list(set(
            PRCS + list(range(PRC_MIN, PRC_MAX + 1, PRC_DELTA))))
        for num_sims, prc in itertools.product(NUMS_SIMS, all_prcs):
            # Select the data corresponding to this number of
            # simulations and percent of each simulation.
            dat_all = list(zip(*utils.filt(
                dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps,
                num_sims, prc)))
            base_dir = path.join(out_dir, f"{num_sims}_{prc}")
            if not path.exists(base_dir):
                os.makedirs(base_dir)
            tmp_dat_flp = path.join(base_dir, "data.npz")
            tmp_scl_prms_flp = path.join(base_dir, "scale_params.json")
            # Finish processesing the data and save it in a form that
            # can be read by the training process.
            ignore = train.gen_data(
                net, args, dat_flp=tmp_dat_flp, scl_prms_flp=tmp_scl_prms_flp,
                dat=(dat_all, sims), save_data=True)
            del ignore
            # Record the paths to the data and scale parameters files.
            dat[(num_sims, prc)] = (tmp_dat_flp, tmp_scl_prms_flp)

        # Assembles the configurations to test.
        cnfs = [
            train.prepare_args({
                "warmup_percent": args["warmup_percent"],
                "model": args["model"],
                "kernel": args["kernel"],
                "degree": args["degree"],
                "penalty": args["penalty"],
                "standardize": args["standardize"],
                "no_rand": args["no_rand"],
                "max_iter": max_iter,
                "keep_percent": prc,
                "num_sims": num_sims,
                "out_dir": path.join(out_dir, f"{num_sims}_{prc}_{max_iter}")
            }) for num_sims, prc, max_iter in set(
                # Fix number of iterations and number of
                # simulations. Vary percent of each simulation.
                list(itertools.product(
                    NUMS_SIMS,
                    range(PRC_MIN, PRC_MAX + 1, PRC_DELTA), NUMS_ITERS)) +
                # Fix percent of each simulation and number of
                # simulations. Vary number of iterations.
                list(itertools.product(
                    NUMS_SIMS, PRCS,
                    range(NUM_ITERS_MIN, NUM_ITERS_MAX + 1, NUM_ITERS_DELTA))))
        ]

        # Look up to the location of the data corresponding to each
        # configuration and copy it to the configuration's output
        # directory, where it will be read automatically.
        for cnf in cnfs:
            cnf_out_dir = cnf["out_dir"]
            if not path.exists(cnf_out_dir):
                os.makedirs(cnf_out_dir)
            tmp_dat_flp, tmp_scl_prms_flp = dat[
                (cnf["num_sims"], cnf["keep_percent"])]
            shutil.copyfile(
                src=tmp_dat_flp,
                dst=path.join(cnf_out_dir, path.basename(tmp_dat_flp)))
            shutil.copyfile(
                src=tmp_scl_prms_flp,
                dst=path.join(cnf_out_dir, path.basename(tmp_scl_prms_flp)))
        # Remove temporary data files.
        for num_sims, prc in itertools.product(NUMS_SIMS, all_prcs):
            tmp_dir = path.join(out_dir, f"{num_sims}_{prc}")
            print(f"Removing: {tmp_dir}")
            shutil.rmtree(tmp_dir)

        cnfs = cnfs[:10]

        # Train models. The result is a list of tuples of the form:
        #     (test loss (in range [0, 1]), training time (seconds))
        # Zip the results list with the list of configurations, then extract the
        # experiment parameters from each configuration and package them and the
        # results into a numpy array.
        ress = np.array([
            (cnf["num_sims"], cnf["keep_percent"], cnf["max_iter"],
             los_tst, tim_trn_s)
            for cnf, (los_tst, tim_trn_s) in zip(
                cnfs, train.run_cnfs(cnfs, SYNC))])

        # Remove real data files.
        for cnf in cnfs:
            print(f"Removing: {cnf['out_dir']}")
            shutil.rmtree(cnf["out_dir"])

        # Save results.
        np.savez_compressed(dat_flp, results=ress)
        print(f"Saving results: {dat_flp}")

    # # Graph results.
    # plt.subplots(figsize=(8, 7))
    # with sns.axes_style("white"):
    #     sns.heatmap(
    #         accs_ratios, linewidth=0.5, center=1, xticklabels=fets_x,
    #         yticklabels=fets_y, square=True, annot=True, fmt=".2f",
    #         annot_kws={"fontsize":8})
    # plt.tight_layout()
    # out_flp = path.join(out_dir, "correlation.pdf")
    # print(f"Saving graph: {out_flp}")
    # plt.savefig(out_flp)

    print(f"Total time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
