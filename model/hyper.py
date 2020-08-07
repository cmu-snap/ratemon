#! /usr/bin/env python3
"""
Hyper-parameter optimization for train.py.
"""

import argparse
import itertools
import multiprocessing
import sys
import time

import ax
import numpy as np

import models
import train
import utils


DEFAULT_TLS_OPT = 40
# When using exhaustive mode, whether to run configurations
# synchronously or in parallel. Ignored otherwise.
SYNC = False


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Hyper-parameter optimizer for train.py.")
    model_opts = sorted(models.MODELS.keys())
    psr.add_argument(
        "--model", default=model_opts[0], help="The model to use.",
        choices=model_opts, type=str)
    psr.add_argument(
        "--data-dir",
        help="The path to the training/validation/testing data (required).",
        required=True, type=str)
    psr.add_argument(
        "--num-sims", default=sys.maxsize,
        help="The number of simulations to consider.", type=int)
    psr.add_argument(
        "--opt-trials", default=DEFAULT_TLS_OPT,
        help="The number of optimization trials to run.", type=int)
    psr.add_argument(
        "--conf-trials", default=train.DEFAULTS["conf_trials"],
        help="The number of trials of each configuration to run.", type=int)
    psr.add_argument(
        "--early-stop", action="store_true",
        help="Enable early stopping as an optimization criterion.")
    psr.add_argument(
        "--max-attempts", default=train.DEFAULTS["max_attempts"],
        help=("The maximum number of failed training attempts to survive, per "
              "configuration."), type=int)
    psr.add_argument(
        "--exhaustive", action="store_true",
        help=("Try all combinations of parameters. Incompatible with parameters "
              "of type \"range\"."))
    psr.add_argument(
        "--no-rand", action="store_true", help="Use a fixed random seed.")
    psr.add_argument(
        "--timeout-s", default=train.DEFAULTS["timeout_s"],
        help="Automatically stop training after this amount of time (seconds).",
        type=float)
    psr.add_argument(
        "--out-dir", default=".",
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()
    tls_opt = args.opt_trials
    tls_cnf = args.conf_trials
    no_rand = args.no_rand

    # Define the optimization parameters.
    params = [
        {
            "name": "data_dir",
            "type": "fixed",
            "value": args.data_dir
        },
        {
            "name": "model",
            "type": "choice",
            "values": ["SvmSklearn", "LrSklearn"]
        },
        {
            "name": "conf_trials",
            "type": "fixed",
            "value": tls_cnf
        },
        {
            "name": "max_attempts",
            "type": "fixed",
            "value": args.max_attempts
        },
        {
            "name": "num_gpus",
            "type": "fixed",
            "value": 0
        },
        {
            "name": "warmup",
            "type": "fixed",
            "value": 500
        },
        {
            "name": "num_sims",
            "type": "fixed",
            "value": args.num_sims
        },
        # {
        #     "name": "train_batch",
        #     "type": "fixed",
        #     "value": 10
        # },
        # {
        #     "name": "learning_rate",
        #     "type": "range",
        #     "bounds": [0.0001, 0.01]
        # },
        # {
        #     "name": "momentum",
        #     "type": "range",
        #     "bounds": [0.09, 0.99]
        # },
        {
            "name": "kernel",
            "type": "choice",
            "values": ["linear", "poly", "rbf", "sigmoid"]
        },
        # {
        #     "name": "degree",
        #     "type": "range",
        #     "bounds": [0, 20]
        # },
        # Represent degree as a choice parameter so that it is
        # compatible with exhaustive mode.
        {
            "name": "degree",
            "type": "choice",
            "values": list(range(0, 21))
        },
        {
            "name": "penalty",
            "type": "choice",
            "values": ["l1", "l2"]
        },
        {
            "name": "no_rand",
            "type": "fixed",
            "value": no_rand
        },
        {
            "name": "timeout_s",
            "type": "fixed",
            "value": args.timeout_s
        },
        {
            "name": "out_dir",
            "type": "fixed",
            "value": args.out_dir
        }
    ]
    # If we are using early stopping, then "epochs" is unnecessary but
    # "val_patience" and "val_improvement_thresh" are also candidates for
    # optimization.
    if args.early_stop:
        params.extend([
            {
                "name": "early_stop",
                "type": "fixed",
                "value": True
            },
            {
                "name": "val_patience",
                "type": "range",
                "bounds": [5, 20]
            },
            {
                "name": "val_improvement_thresh",
                "type": "range",
                "bounds": [0.01, 0.1]
            }
        ])
    else:
        params.extend([
            # {
            #     "name": "epochs",
            #     "type": "range",
            #     "bounds": [1, 100]
            # },
        ])

    tim_srt_s = time.time()
    if args.exhaustive:
        for param in params:
            assert param["type"] != "range", \
                f"Exhaustive mode does not support range parameters: {param}"
        fixed = {
            param["name"]: param["value"]
            for param in params if param["type"] == "fixed"}
        to_vary = [
            [(param["name"], value) for value in param["values"]]
            for param in params if param["type"] == "choice"]
        print(
            f"Varying these parameters, with {tls_cnf} sub-trials(s) for each "
            f"configuration: {[pairs[0][0] for pairs in to_vary]}")
        cnfs = [
            {**fixed, **dict(params)} for params in itertools.product(*to_vary)]
        print(f"Total trials: {len(cnfs) * tls_cnf}")
        if SYNC:
            res = [train.run_many(cnf) for cnf in cnfs]
        else:
            with multiprocessing.Pool() as pol:
                res = pol.map(train.run_many, cnfs)
        best_idx = np.argmin(np.array(res))
        best_params = cnfs[best_idx]
        best_err = res[best_idx]
    else:
        print((f"Running {tls_opt} optimization trial(s), with {tls_cnf} "
               "sub-trial(s) for each configuration."))
        best_params, best_vals, _, _ = ax.optimize(
            parameters=params,
            evaluation_function=train.run_many,
            minimize=True,
            total_trials=args.opt_trials,
            random_seed=utils.SEED if no_rand else None)
        best_err = best_vals[0]["objective"]
    print((f"Done with hyper-parameter optimization - "
           f"{time.time() - tim_srt_s:.2f} seconds"))
    print(f"\nBest params: {best_params}")
    print(f"Best error: {best_err:.4f}%")


if __name__ == "__main__":
    main()
