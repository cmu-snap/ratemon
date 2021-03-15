#! /usr/bin/env python3
"""
Hyper-parameter optimization for train.py.
"""

import argparse
import itertools
import multiprocessing
import time

import ax
import numpy as np

import cl_args
import defaults
import train


DEFAULT_TLS_OPT = 40


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Hyper-parameter optimizer for train.py.")
    psr, psr_verify = cl_args.add_training(psr)
    psr.add_argument(
        "--opt-trials", default=DEFAULT_TLS_OPT,
        help="The number of optimization trials to run.", type=int)
    psr.add_argument(
        "--exhaustive", action="store_true",
        help=("Try all combinations of parameters. Incompatible with "
              "parameters of type \"range\"."))
    args = psr_verify(psr.parse_args())
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
            "name": "warmup_percent",
            "type": "fixed",
            "value": 10
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
        if defaults.SYNC:
            res = [train.run_trials(cnf)[0] for cnf in cnfs]
        else:
            with multiprocessing.Pool() as pol:
                res = pol.map(lambda cnf: train.run_trials(cnf)[0], cnfs)
        best_idx = np.argmin(np.array(res))
        best_params = cnfs[best_idx]
        best_err = res[best_idx]
    else:
        print((f"Running {tls_opt} optimization trial(s), with {tls_cnf} "
               "sub-trial(s) for each configuration."))
        best_params, best_vals, _, _ = ax.optimize(
            parameters=params,
            evaluation_function=lambda cnf: train.run_trials(cnf)[0],
            minimize=True,
            total_trials=args.opt_trials,
            random_seed=defaults.SEED if no_rand else None)
        best_err = best_vals[0]["objective"]
    print((f"Done with hyper-parameter optimization - "
           f"{time.time() - tim_srt_s:.2f} seconds"))
    print(f"\nBest params: {best_params}")
    print(f"Best error: {best_err:.4f}%")


if __name__ == "__main__":
    main()
