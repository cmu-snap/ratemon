#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import os
from os import path
import pickle
import sys
import time

import numpy as np

from unfair.model import defaults, features, gen_features, utils


def plot_hist(args, jfis):
    disabled, enabled, _ = zip(*jfis.values())
    plt.hist(
        disabled, bins=50, density=True, facecolor="r", alpha=0.75, label="Disabled"
    )
    plt.hist(enabled, bins=50, density=True, facecolor="g", alpha=0.75, label="Enabled")

    plt.xlabel("JFI")
    plt.ylabel("Probability (%)")
    plt.title("Histogram of JFI, with and without unfairness monitor")
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.join(args.out_dir, "jfi_hist.pdf"))


def parse_opened_exp(exp, exp_flp, exp_dir, out_flp, skip_smoothed):
    print(f"Parsing: {exp_flp}")
    if exp.name.startswith("FAILED"):
        print(f"Error: Experimant failed: {exp_flp}")
        return -1
    if exp.tot_flws == 0:
        print(f"Error: No flows to analyze in: {exp_flp}")
        return -1

    server_pcap = path.join(exp_dir, f"server-tcpdump-{exp.name}.pcap")
    if not path.exists(server_pcap):
        print(f"Warning: Missing server pcap file in: {exp_flp}")
        return -1

    # Determine flow src and dst ports.
    params_flp = path.join(exp_dir, f"{exp.name}.json")
    if not path.exists(params_flp):
        print(f"Error: Cannot find params file ({params_flp}) in: {exp_flp}")
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
    flw_to_pkts = utils.parse_packets(server_pcap, flw_to_cca)
    # Discard the ACK packets.
    flw_to_pkts = {flw: data_pkts for flw, (data_pkts, ack_pkts) in flw_to_pkts.items()}

    print(f"\tParsed packets: {server_pcap}")

    late_flws = [flw for flw in flws if flw[1] == 50001]
    earliest_late_flow_start_time = min(
        [flw_to_pkts[flw][features.ARRIVAL_TIME_FET][0] for flw in late_flws]
    )

    # Remove data from before the late flows start.
    for flw in flw_to_pkts.keys():
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

    flw_to_bits = {
        flw: np.sum(pkts[features.WIRELEN_FET]) for flw, pkts in flw_to_pkts.items()
    }

    jfi = sum(flw_to_bits.values()) ** 2 / (
        len(flw_to_bits) * sum(bits**2 for bits in flw_to_bits.values())
    )

    # # Save the results.
    # if path.exists(out_flp):
    #     print(f"\tOutput already exists: {out_flp}")
    # else:
    #     print(f"\tSaving: {out_flp}")
    #     np.savez_compressed(
    #         out_flp,
    #         **{str(k + 1): v for k, v in enumerate(flw_results[flw] for flw in flws)},
    #     )
    return exp, jfi


def main(args):
    # Find all experiments.
    pcaps = [
        (
            path.join(args.exp_dir, exp),
            args.untar_dir,
            args.out_dir,
            False,
            parse_opened_exp,
        )
        for exp in sorted(os.listdir(args.exp_dir))
        if exp.endswith(".tar.gz")
    ]

    print(f"Num files: {len(pcaps)}")
    start_time_s = time.time()

    data_flp = path.join(args.out_dir, "jfis.pickle")
    if path.exists(data_flp):
        print("Loading data from:", data_flp)
        # Load existing raw JFI results.
        with open(data_flp, "rb") as fil:
            jfis = pickle.load(fil)
        if len(jfis) != len(pcaps):
            print(
                f"Error: Expected {len(pcaps)} JFI results, but found {len(jfis)}. "
                f"Delete {data_flp} and try again."
            )
            return
    else:
        if defaults.SYNC:
            jfis = {gen_features.parse_exp(*pcap) for pcap in pcaps}
        else:
            with multiprocessing.Pool(processes=args.parallel) as pol:
                jfis = set(pol.starmap(gen_features.parse_exp, pcaps))
        # Save raw JFI results from parsed experiments.
        with open(data_flp, "wb") as fil:
            pickle.dump(jfis, fil)

    # Dict mapping experiment to JFI.
    jfis = {exp_jfi[0]: exp_jfi[1] for exp_jfi in jfis if isinstance(exp_jfi, tuple)}
    # Experiments in which the unfairness monitor was enabled.
    enabled = {exp for exp in jfis.keys() if exp.use_unfairness_monitor}
    # Experiments in which the unfairness monitor was disabled.
    disabled = {exp for exp in jfis.keys() if not exp.use_unfairness_monitor}

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
            print(
                "Warning: Cannot find experiment with unfairness monitor disabled:",
                target_disabled_name,
            )
            continue

        matched[enabled_exp] = (
            jfis[target_disabled_exp],
            jfis[enabled_exp],
            jfis[enabled_exp] - jfis[target_disabled_exp],
        )

    plot_hist(args, matched)

    # Save JFI results.
    with open(path.join(args.out_dir, "jfi_deltas.json"), "w") as fil:
        json.dump(
            {exp.name: results for exp, results in matched.items()}, fil, indent=4
        )

    # Extract the deltas as a numpy array.
    jfi_deltas = np.array(list(zip(*matched.values()))[2])

    print(
        f"Overall JFI delta:\n"
        f"\tAvg: {np.mean(jfi_deltas)}\n"
        f"\tStddev: {np.std(jfi_deltas)}\n"
        f"\tVar: {np.var(jfi_deltas)}\n"
    )

    print(f"Done analyzing - time: {time.time() - start_time_s:.2f} seconds")
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
    args = parser.parse_args()
    assert path.isdir(args.exp_dir)
    assert path.isdir(args.out_dir)
    return args


if __name__ == "__main__":
    sys.exit(main(parse_args()))
