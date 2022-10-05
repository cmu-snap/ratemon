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
        #     0: jfis_disabled
        #     1: jfis_enabled
        #     2: jfis_deltas
        #     3: jfi_deltas_percent
        #     4: overall_utils_disabled
        #     5: overall_utils_enabled
        #     6: overall_util_deltas_percent
        #     7: fair_flows_utils_disabled
        #     8: fair_flows_utils_enabled
        #     9: fair_flows_util_deltas_percent
        #     10: unfair_flows_utils_disabled
        #     11: unfair_flows_utils_enabled
        #     12: unfair_flows_util_deltas_percent
        #     13: results_pickle
        # )}
        results_refactored[key] = (*(list(zip(*results_json.values()))), results_pickle)
    results = results_refactored

    evl.plot_hist(
        args,
        lines=[
            # Extract and flatten the JFIs across all CCAs.
            [val for vals in results.values() for val in vals[0]],
            [val for vals in results.values() for val in vals[1]],
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
