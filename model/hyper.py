#! /usr/bin/env python3
"""
Hyper-parameter optimization for train.py.
"""

import argparse
import time

import ax

import train
import models


DEFAULT_TLS_OPT = 40


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
        "--data",
        help="The path to the training/validation/testing data (required).",
        required=True, type=str)
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
            "name": "data",
            "type": "fixed",
            "value": args.data
        },
        {
            "name": "model",
            "type": "fixed",
            "value": args.model
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
            "value": 1
        },
        {
            "name": "warmup",
            "type": "fixed",
            "value": 1000
        },
        {
            "name": "num_sims",
            "type": "fixed",
            "value": None
        },
        {
            "name": "train_batch",
            "type": "fixed",
            "value": 10
        },
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [0.0001, 0.01]
        },
        {
            "name": "momentum",
            "type": "range",
            "bounds": [0.09, 0.99]
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
            {
                "name": "epochs",
                "type": "range",
                "bounds": [1, 100]
            },
        ])

    print((f"Running {tls_opt} optimization trial(s), with {tls_cnf} "
           "sub-trial(s) for each configuration."))
    tim_srt_s = time.time()
    best_params, best_vals, _, _ = ax.optimize(
        parameters=params,
        evaluation_function=train.run_many,
        minimize=True,
        total_trials=args.opt_trials,
        random_seed=train.SEED if no_rand else None)
    print((f"Done with hyper-parameter optimization - "
           f"{time.time() - tim_srt_s:.2f} seconds"))
    print(f"\nBest params: {best_params}")
    print(f"Best error: {best_vals[0]['objective']:.2f}%")


if __name__ == "__main__":
    main()
