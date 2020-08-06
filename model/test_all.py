#! /usr/bin/env python3
""" Evalaute the model on simulations. """

import argparse
import json
import multiprocessing
import os
from os import path
import pickle
from statistics import mean
import time

import numpy as np
import torch

import models
import train
import utils


all_accuracy = []

# Bandwidth in Mbps
bw_dict = {
    1: [],
    10: [],
    30: [],
    50: [],
    1000: []
}

# RTT in us
rtt_dict = {
    1000: [],
    10000: [],
    50000: [],
    100000: [],
    1000000: []
}

# Queue size in number of packets
queue_dict = {
    10: [],
    50: [],
    100: [],
    200: [],
    500: [],
    1000: [],
    5000: []
}

def process_one(sim_flp, out_dir, net, warmup, scl_prms_flp, standardize):
    """ Evaluate a single simulation. """
    # Load and parse the simulation.
    (dat_in, dat_out, dat_out_raw, dat_out_oracle, _), sim = (
        train.process_sim(
            idx=0, total=1, net=net, sim_flp=sim_flp, tmp_dir=out_dir,
            warmup=warmup, prc=100, sequential=True))

    # Load and apply the scaling parameters.
    with open(scl_prms_flp, "r") as fil:
        scl_prms = json.load(fil)
    dat_in = utils.scale_all(dat_in, scl_prms, 0, 1, standardize)

    # Visualize the ground truth data.
    utils.visualize_classes(net, dat_out, isinstance(net, models.SvmWrapper))

    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # Test the simulation.
    accuracy = net.test(
        *utils.Dataset(
            fets=dat_in.dtype.names,
            dat_in=utils.clean(dat_in),
            dat_out=utils.clean(dat_out),
            dat_out_raw=utils.clean(dat_out_raw),
            dat_out_oracle=utils.clean(dat_out_oracle),
            num_flws=np.array(
                [sim.unfair_flws + sim.other_flws] * dat_in.shape[0],
                dtype=float)).raw(),
        graph_prms={
            "out_dir": out_dir,
            "sort_by_unfairness": False,
            "dur_s": sim.dur_s
        })

    all_accuracy.append(accuracy)
    mean_accuracy = mean(all_accuracy)

    for bw_Mbps in bw_dict:
        if sim.bw_Mbps <= bw_Mbps:
            bw_dict[bw_Mbps].append(accuracy)
            break

    rtt_us = (sim.btl_delay_us + 2 * sim.edge_delays[0]) * 2
    for rtt_us_ in rtt_dict:
        if rtt_us <= rtt_us_:
            rtt_dict[rtt_us_].append(accuracy)
            break

    for queue_p in queue_dict:
        if sim.queue_p <= queue_p:
            queue_dict[queue_p].append(accuracy)
            break

    print(
        f"Finish processing {sim.name}\n"
        f"Average accuracy for all the processed simulations: {mean_accuracy}")

    for bw_Mbps in bw_dict:
        if bw_dict[bw_Mbps]:
            mean_accuracy = mean(bw_dict[bw_Mbps])
            print(f"Bandwidth less than {bw_Mbps}Mbps accuracy {mean_accuracy}")

    for rtt_us_ in rtt_dict:
        if rtt_dict[rtt_us_]:
            mean_accuracy = mean(rtt_dict[rtt_us_])
            print(f"Rtt less than {rtt_us_}ns accuracy {mean_accuracy}")

    for queue_p in queue_dict:
        if queue_dict[queue_p]:
            mean_accuracy = mean(queue_dict[queue_p])
            print(f"Queue size less than {queue_p} packets accuracy {mean_accuracy}")


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Hyper-parameter optimizer for train.py.")
    psr.add_argument(
        "--model", help="The path to a trained model file.", required=True,
        type=str)
    psr.add_argument(
        "--simulations", help="The path to a simulations to analyze.", required=True,
        type=str)
    psr.add_argument(
        "--warmup", default=0,
        help=("The number of packets to drop from the beginning of each "
              "simulation."),
        type=int)
    psr.add_argument(
        "--scale-params", help="The path to the input scaling parameters.",
        required=True, type=str)
    psr.add_argument(
        "--standardize", action="store_true",
        help=("Standardize the data so that it has a mean of 0 and a variance "
              "of 1. Otherwise, data will be rescaled to the range [0, 1]."))
    psr.add_argument(
        "--out-dir", default=".",
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()
    mdl_flp = args.model
    sim_dir = args.simulations
    warmup = args.warmup
    scl_prms_flp = args.scale_params
    out_dir = args.out_dir
    standardize = args.standardize
    assert path.exists(mdl_flp), f"Model file does not exist: {mdl_flp}"
    assert path.exists(sim_dir), f"Simulation file does not exist: {sim_dir}"
    assert warmup >= 0, f"Warmup cannot be negative, but is: {warmup}"
    assert path.exists(scl_prms_flp), \
        f"Scaling parameters file does not exist: {scl_prms_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # Parse the model filepath to determine the model type, and instantiate it.
    net = models.MODELS[
        # Convert the model filename to an arguments dictionary, and
        # extract the "model" key.
        utils.str_to_args(
            path.basename(mdl_flp),
            order=sorted(train.DEFAULTS.keys())
        )["model"]]()
    # # Manually remove the loss event rate sqrt feature.
    # net.in_spc.remove("loss event rate sqrt")
    # Load the model.
    if mdl_flp.endswith("pickle"):
        with open(mdl_flp, "rb") as fil:
            mdl = pickle.load(fil)
    elif mdl_flp.endswith("pth"):
        mdl = torch.jit.load(mdl_flp)
    else:
        raise Exception(f"Unknown model type: {mdl_flp}")
    net.net = mdl
    net.graph = True

    func_input = [
        (path.join(sim_dir, sim), path.join(out_dir, sim.split(".")[0]), net,
         warmup, scl_prms_flp, standardize)
        for sim in sorted(os.listdir(sim_dir))]

    print(f"Num files: {len(func_input)}")
    tim_srt_s = time.time()
    with multiprocessing.Pool() as pol:
        pol.starmap(process_one, func_input)

    print(f"Done Processing - time: {time.time() - tim_srt_s:.2f} seconds")


if __name__ == "__main__":
    main()
