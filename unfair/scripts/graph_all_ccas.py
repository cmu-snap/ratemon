#!/usr/bin/env python3

import argparse
import json
import os
from os import path
import pickle
import sys

import numpy as np

from unfair.scripts import eval as evl
from unfair.model import utils


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
    args.out_dir = path.join(args.results_dir, "all_cca_results")
    os.makedirs(args.out_dir, exist_ok=True)

    results = load_results(args)
    print(list(results.keys()))

    results_refactored = {}
    for cca_pair, (results_json, results_pickle) in results.items():
        # Skip DCTCP because we do not have switch ECN marking enabled.
        if cca_pair[1] == "dctcp":
            continue

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
        #     13: experiments
        #     14: results_pickle
        # )}
        keys, values = zip(*results_json.items())
        results_refactored[cca_pair] = (
            *(list(zip(*values))),
            # Convert each experiment name to an Exp object.
            [utils.Exp(exp_name) for exp_name in keys],
            results_pickle,
        )
    results = results_refactored

    evl.plot_bar(
        args,
        lines=[
            # Need to sort to ensure that the order is the same as in x_tick_labels.
            [
                np.mean(vals[3])
                for _, vals in sorted(results.items(), key=lambda p: p[0][1])
            ],
        ],
        labels=[None],
        x_label="Newcomer CCA (cubic vs. X)",
        y_label="JFI change (%)",
        x_tick_labels=[
            cca_pair[1]
            for cca_pair, _ in sorted(results.items(), key=lambda p: p[0][1])
        ],
        filename="jfi_deltas_bar.pdf",
        colors=[evl.COLORS_MAP["blue"]],
        # title="Change in Average JFI, for all CCAs",
        rotate=True,
    )
    evl.plot_bar(
        args,
        lines=[
            # Need to sort to ensure that the order is the same as in x_tick_labels.
            [
                np.mean(vals[0])
                for _, vals in sorted(results.items(), key=lambda p: p[0][1])
            ],
            [
                np.mean(vals[1])
                for _, vals in sorted(results.items(), key=lambda p: p[0][1])
            ],
        ],
        labels=["Original", "UnfairMon"],
        x_label="Newcomer CCA (cubic vs. X)",
        y_label="JFI",
        x_tick_labels=[
            cca_pair[1]
            for cca_pair, _ in sorted(results.items(), key=lambda p: p[0][1])
        ],
        y_max=1,
        filename="jfis_bar.pdf",
        colors=evl.COLORS,
        # title="Average JFI, for all CCAs",
        rotate=True,
    )
    evl.plot_cdf(
        args,
        lines=[
            # Extract and flatten the JFIs across all CCAs.
            [val for vals in results.values() for val in vals[0]],
            [val for vals in results.values() for val in vals[1]],
        ],
        labels=["Original", "UnfairMon"],
        x_label="JFI",
        x_max=1.0,
        filename="jfi_cdf.pdf",
        linestyles=["dashed", "dashdot"],
        colors=evl.COLORS,
        # title="CDF of JFI,\nwith and without unfairness monitor",
    )
    evl.plot_cdf(
        args,
        lines=[
            [val for vals in results.values() for val in vals[4]],
            [val for vals in results.values() for val in vals[5]],
        ],
        labels=["Original", "UnfairMon"],
        x_label="Overall link utilization (%)",
        x_max=100,
        filename="util_cdf.pdf",
        linestyles=["dashed", "dashdot"],
        colors=evl.COLORS,
        legendloc="upper left",
        # title="CDF of overall link utilization,\nwith and without unfairness monitor",
    )
    evl.plot_cdf(
        args,
        lines=[
            # Expected total utilization of incumbent flows.
            [
                exp.cca_1_flws / exp.tot_flws * 100
                for vals in results.values()
                for exp in vals[13]
            ],
            [val for vals in results.values() for val in vals[7]],
            [val for vals in results.values() for val in vals[8]],
        ],
        labels=["Perfectly Fair", "Original", "UnfairMon"],
        x_label="Total link utilization of incumbent flows (%)",
        x_max=100,
        filename="fair_flows_util_cdf.pdf",
        # title='CDF of "incumbent" flows link utilization,\nwith and without unfairness monitor',
        colors=[
            evl.COLORS_MAP["orange"],
            evl.COLORS_MAP["red"],
            evl.COLORS_MAP["blue"],
        ],
    )
    evl.plot_cdf(
        args,
        lines=[
            # Expected total utilization of newcomer flows.
            [
                exp.cca_2_flws / exp.tot_flws * 100
                for vals in results.values()
                for exp in vals[13]
            ],
            [val for vals in results.values() for val in vals[10]],
            [val for vals in results.values() for val in vals[11]],
        ],
        labels=["Perfectly Fair", "Original", "UnfairMon"],
        x_label="Link utilization of newcomer flow (%)",
        x_max=100,
        filename="unfair_flows_util_cdf.pdf",
        # title='CDF of newcomer flow link utilization,\nwith and without unfairness monitor',
        colors=[
            evl.COLORS_MAP["orange"],
            evl.COLORS_MAP["red"],
            evl.COLORS_MAP["blue"],
        ],
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
