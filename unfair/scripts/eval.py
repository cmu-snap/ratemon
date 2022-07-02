#!/usr/bin/env python

import argparse
import json
import logging
import multiprocessing
import os
from os import path
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from unfair.model import defaults, features, gen_features, utils


def plot_cdf(args, disabled, enabled, x_label, filename):
    count, bins_count = np.histogram(disabled, bins=len(disabled))
    plt.plot(
        bins_count[1:],
        np.cumsum(count / sum(count)),
        alpha=0.75,
        color="r",
        label="Disabled",
    )

    count, bins_count = np.histogram(enabled, bins=len(enabled))
    plt.plot(
        bins_count[1:],
        np.cumsum(count / sum(count)),
        alpha=0.75,
        color="g",
        label="Enabled",
    )

    plt.xlabel(x_label)
    plt.ylabel("CDF")
    plt.title(f"CDF of {x_label}, with and without unfairness monitor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    cdf_flp = path.join(args.out_dir, filename)
    plt.savefig(cdf_flp)
    plt.close()
    logging.info("Saved CDF to: %s", cdf_flp)


def plot_hist(args, disabled, enabled, x_label, filename):
    plt.hist(
        disabled, bins=50, density=True, facecolor="r", alpha=0.75, label="Disabled"
    )
    plt.hist(enabled, bins=50, density=True, facecolor="g", alpha=0.75, label="Enabled")

    plt.xlabel(x_label)
    plt.ylabel("probability (%)")
    plt.title(f"Histogram of {x_label}, with and without unfairness monitor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    hist_flp = path.join(args.out_dir, filename)
    plt.savefig(hist_flp)
    plt.close()
    logging.info("Saved histogram to: %s", hist_flp)


def parse_opened_exp(exp, exp_flp, exp_dir, select_tail_percent):
    logging.info("Parsing: %s", exp_flp)
    if exp.name.startswith("FAILED"):
        logging.info("Error: Experimant failed: %s", exp_flp)
        return -1
    if exp.tot_flws == 0:
        logging.info("Error: No flows to analyze in: %s", exp_flp)
        return -1

    server_pcap = path.join(exp_dir, f"server-tcpdump-{exp.name}.pcap")
    if not path.exists(server_pcap):
        logging.info("Warning: Missing server pcap file in: %s", exp_flp)
        return -1

    # Determine flow src and dst ports.
    params_flp = path.join(exp_dir, f"{exp.name}.json")
    if not path.exists(params_flp):
        logging.info("Error: Cannot find params file (%s) in: %s", params_flp, exp_flp)
        return -1
    with open(params_flp, "r", encoding="utf-8") as fil:
        params = json.load(fil)

    # Dictionary mapping a flow to its flow's CCA. Each flow is a tuple of the
    # form: (client port, server port)
    #
    # { (client port, server port): CCA }
    flw_to_cca = {
        (client_port, flw[4]): flw[0]
        for flw in params["flowsets"]
        for client_port in flw[3]
    }
    flws = list(flw_to_cca.keys())
    flw_to_pkts = utils.parse_packets(server_pcap, flw_to_cca, select_tail_percent)
    # Discard the ACK packets.
    flw_to_pkts = {flw: data_pkts for flw, (data_pkts, ack_pkts) in flw_to_pkts.items()}
    logging.info("\tParsed packets: %s", server_pcap)
    flw_to_pkts = utils.drop_packets_after_first_flow_finishes(flw_to_pkts)

    late_flows_port = max(flw[4] for flw in params["flowsets"])
    late_flws = [
        flw for flw in flws if flw[1] == late_flows_port if len(flw_to_pkts[flw]) > 0
    ]
    if len(late_flws) == 0:
        logging.info("\tWarning: No late flows to analyze in: %s", exp_flp)
        return exp, -1, -1
    earliest_late_flow_start_time = min(
        [
            flw_to_pkts[flw][features.ARRIVAL_TIME_FET][0]
            for flw in late_flws
            if len(flw_to_pkts[flw]) > 0
        ]
    )

    # Remove data from before the late flows start.
    for flw in flw_to_pkts.keys():
        if len(flw_to_pkts[flw]) == 0:
            flw_to_pkts[flw] = []
            continue
        for idx, arr_time in enumerate(flw_to_pkts[flw][features.ARRIVAL_TIME_FET]):
            if arr_time >= earliest_late_flow_start_time:
                break
        flw_to_pkts[flw] = flw_to_pkts[flw][idx:]

    # zipped_arr_times, zipped_dat = utils.zip_timeseries(
    #     [flw_to_pkts_server[flw][features.ARRIVAL_TIME_FET] for flw in flws],
    #     [flw_to_pkts_server[flw] for flw in flws],
    # )
    # for idx, arr_time in enumerate(zipped_arr_times):
    #     if arr_time >= earliest_late_flow_start_time:
    #         break
    # zipped_arr_times = zipped_arr_times[idx:]
    # zipped_dat = zipped_dat[idx:]

    flw_to_tput_bps = {
        flw: 0 if len(pkts) == 0 else utils.safe_tput_bps(pkts, 0, len(pkts) - 1)
        for flw, pkts in flw_to_pkts.items()
    }

    jfi = sum(flw_to_tput_bps.values()) ** 2 / (
        len(flw_to_tput_bps) * sum(bits**2 for bits in flw_to_tput_bps.values())
    )

    # Calculate the average combined throughput of all flows by dividing the total bits
    # received by all flows by the time difference between when the first flow started
    # and when the last flow finished.
    bits_times = (
        (
            utils.safe_sum(pkts[features.WIRELEN_FET], 0, len(pkts) - 1),
            utils.safe_min_win(pkts[features.ARRIVAL_TIME_FET], 0, len(pkts) - 1),
            utils.safe_max_win(pkts[features.ARRIVAL_TIME_FET], 0, len(pkts) - 1),
        )
        for pkts in flw_to_pkts.values()
        if len(pkts) > 0
    )
    bits, start_times_us, end_times_us = zip(*bits_times)
    avg_total_tput_bps = (
        sum(bits) * 8 / ((max(end_times_us) - min(start_times_us)) / 1e6)
    )
    avg_util = avg_total_tput_bps / exp.bw_bps

    # # Save the results.
    # if path.exists(out_flp):
    #     logging.info(f"\tOutput already exists: {out_flp}")
    # else:
    #     logging.info(f"\tSaving: {out_flp}")
    #     np.savez_compressed(
    #         out_flp,
    #         **{str(k + 1): v for k, v in enumerate(flw_results[flw] for flw in flws)},
    #     )
    return exp, jfi, avg_util


def main(args):
    log_flp = path.join(args.out_dir, "output.log")
    logging.basicConfig(
        filename=log_flp,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG,
    )
    print("Logging to:", log_flp)
    logging.info("Evaluating experiments in: %s", args.exp_dir)

    # Find all experiments.
    pcaps = [
        (
            path.join(args.exp_dir, exp),
            args.untar_dir,
            args.out_dir,
            False,
            args.select_tail_percent,
            parse_opened_exp,
        )
        for exp in sorted(os.listdir(args.exp_dir))
        if exp.endswith(".tar.gz")
    ]

    logging.info("Num files: %d", len(pcaps))
    start_time_s = time.time()

    data_flp = path.join(args.out_dir, "results.pickle")
    if path.exists(data_flp):
        logging.info("Loading data from: %s", data_flp)
        # Load existing raw JFI results.
        with open(data_flp, "rb") as fil:
            results = pickle.load(fil)
        if len(results) != len(pcaps):
            logging.info(
                (
                    "Error: Expected %d JFI results, but found %d. "
                    "Delete %s and try again."
                ),
                len(pcaps),
                len(results),
                data_flp,
            )
            return 1
    else:
        if defaults.SYNC:
            results = {gen_features.parse_exp(*pcap) for pcap in pcaps}
        else:
            with multiprocessing.Pool(processes=args.parallel) as pol:
                results = set(pol.starmap(gen_features.parse_exp, pcaps))
        # Save raw JFI results from parsed experiments.
        with open(data_flp, "wb") as fil:
            pickle.dump(results, fil)

    # Dict mapping experiment to JFI.
    results = {
        exp_jfi_util[0]: (exp_jfi_util[1], exp_jfi_util[2])
        for exp_jfi_util in results
        if (
            isinstance(exp_jfi_util, tuple)
            and exp_jfi_util[1] != -1
            and exp_jfi_util[2] != -1
        )
    }
    # Experiments in which the unfairness monitor was enabled.
    enabled = {exp for exp in results.keys() if exp.use_unfairness_monitor}
    # Experiments in which the unfairness monitor was disabled.
    disabled = {exp for exp in results.keys() if not exp.use_unfairness_monitor}

    # Match each enabled experiment with its corresponding disabled experiment and
    # compute the JFI delta. matched is a dict mapping the name of the enabled
    # experiment to a tuple of the form:
    #     ( disabled JFI, enabled JFI, difference in JFI from enabled to disabled )
    matched = {}
    for enabled_exp in enabled:
        # Find the corresponding experiment with the unfairness monitor disabled.
        target_disabled_name = enabled_exp.name.replace("unfairTrue", "unfairFalse")
        target_disabled_exp = None
        for disabled_exp in disabled:
            if disabled_exp.name == target_disabled_name:
                target_disabled_exp = disabled_exp
                break
        if target_disabled_exp is None:
            logging.info(
                "Warning: Cannot find experiment with unfairness monitor disabled: %s",
                target_disabled_name,
            )
            continue

        jfi_disabled, util_disabled = results[target_disabled_exp]
        jfi_enabled, util_enabled = results[enabled_exp]

        matched[enabled_exp] = (
            jfi_disabled,
            jfi_enabled,
            jfi_enabled - jfi_disabled,
            (jfi_enabled - jfi_disabled) / jfi_disabled * 100,
            util_disabled * 100,
            util_enabled * 100,
            (util_enabled - util_disabled) * 100,
        )
    # Save JFI results.
    with open(path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fil:
        json.dump({exp.name: val for exp, val in matched.items()}, fil, indent=4)

    logging.info("Matched experiments: %d", len(matched))
    (
        jfis_disabled,
        jfis_enabled,
        _,
        jfi_deltas_percent,
        utils_disabled,
        utils_enabled,
        util_deltas_percent,
    ) = list(zip(*matched.values()))

    plot_hist(args, jfis_disabled, jfis_enabled, "JFI", "jfi_hist.pdf")
    plot_hist(
        args, utils_disabled, utils_enabled, "link utilization (%)", "util_hist.pdf"
    )
    plot_cdf(
        args,
        # [1 - x for x in jfis_disabled],
        # [1 - x for x in jfis_enabled],
        jfis_disabled,
        jfis_enabled,
        "JFI",
        "jfi_cdf.pdf",
    )
    plot_cdf(
        args,
        [100 - x for x in utils_disabled],
        [100 - x for x in utils_enabled],
        "unused link capacity (%)",
        "unused_util_cdf.pdf",
    )
    plot_cdf(
        args,
        utils_disabled,
        utils_enabled,
        "link utilization (%)",
        "util_cdf.pdf",
    )

    logging.info(
        (
            "\nOverall JFI change (percent) --- higher is better:\n"
            "\tAvg: %.4f %%\n"
            "\tStddev: %.4f %%\n"
            "\tVar: %.4f %%"
        ),
        np.mean(jfi_deltas_percent),
        np.std(jfi_deltas_percent),
        np.var(jfi_deltas_percent),
    )
    logging.info(
        "Overall average JFI with monitor enabled: %.4f", np.mean(jfis_enabled)
    )
    logging.info(
        (
            "\nOverall link utilization change "
            "--- higher is better, want to be >= 0%%:\n"
            "\tAvg: %.4f %%\n"
            "\tStddev: %.4f %%\n"
            "\tVar: %.4f %%"
        ),
        np.mean(util_deltas_percent),
        np.std(util_deltas_percent),
        np.var(util_deltas_percent),
    )
    logging.info(
        "Overall average link utilization with monitor enabled: %.4f %%",
        np.mean(utils_enabled),
    )

    logging.info("Done analyzing - time: %.2f seconds", time.time() - start_time_s)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation.")
    parser.add_argument(
        "--exp-dir",
        help="The directory in which the experiment results are stored.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--untar-dir",
        help=(
            "The directory in which the untarred experiment intermediate "
            "files are stored (required)."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--parallel",
        default=multiprocessing.cpu_count(),
        help="The number of files to parse in parallel.",
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        help="The directory in which to store the results.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--select-tail-percent",
        help="The percentage (by time) of the tail of the PCAPs to select.",
        required=False,
        type=float,
    )
    args = parser.parse_args()
    assert path.isdir(args.exp_dir)
    assert path.isdir(args.out_dir)
    return args


if __name__ == "__main__":
    sys.exit(main(parse_args()))
