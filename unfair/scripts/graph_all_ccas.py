#!/usr/bin/env python3

import argparse
import json
import os
from os import path
import pickle
import sys

from unfair.scripts import eval as evl


def load_results(args):
    results = {}
    for fln in os.listdir(args.results_dir):
        dirpath = path.join(args.results_dir, fln)
        if path.isdir(dirpath) and fln.startswith(f"{args.cca1}-"):
            results_pickle_flp = path.join(dirpath, "processed", "results.pickle")
            results_json_flp = path.join(dirpath, "processed", "results.json")

            if not (path.isfile(results_pickle_flp) and path.isfile(results_json_flp)):
                print(f"Warning: Could not find results in: {dirpath}")
                continue
            print(f"Loading results for: {dirpath}")

            with open(results_json_flp, "r", encoding="utf-8") as fil:
                results_json = json.load(fil)

            with open(results_pickle_flp, "rb") as fil:
                results_pickle = pickle.load(fil)

            results[(args.cca1, fln.split("-")[1])] = (results_json, results_pickle)
    return results


def main(args):
    results = load_results(args)
    print(list(results.keys()))

    results_refactored = {}
    for key, (results_json, results_pickle) in results.items():
        # { (cca1, cca2): (
        #     0: results_json
        #     1: jfis_disabled
        #     2: jfis_enabled
        #     3: jfis_deltas
        #     4: jfi_deltas_percent
        #     5: overall_utils_disabled
        #     6: overall_utils_enabled
        #     7: overall_util_deltas_percent
        #     8: fair_flows_utils_disabled
        #     9: fair_flows_utils_enabled
        #     10: fair_flows_util_deltas_percent
        #     11: unfair_flows_utils_disabled
        #     12: unfair_flows_utils_enabled
        #     13: unfair_flows_util_deltas_percent
        # )}
        results_refactored[key] = (results_json, *(list(zip(*results_pickle.values()))))
    results = results_refactored

    evl.plot_hist(
        args,
        lines=[
            # Extract and flatten the JFIs across all CCAs.
            [val for vals in results.values() for val in vals[1]],
            [val for vals in results.values() for val in vals[2]],
        ],
        labels=["Original", "UnfairMon"],
        x_label="JFI",
        filename="jfi_hist.pdf",
        # title="Histogram of JFI,\nwith and without unfairness monitor",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        help="Directory containing results for many CCA pairs.",
        required=True,
        type=str,
    )
    parser.add_argument("--cca1", help="Incumbent CCA.", required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
