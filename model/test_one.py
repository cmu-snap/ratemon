#! /usr/bin/env python3
""" Evalaute a model on a single simulation. """

import argparse

import os
from os import path
import pickle

import json
import numpy as np
import torch

import models
import train
import utils


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Evaluate a single simulation.")
    psr.add_argument(
        "--model", help="The path to a trained model file.", required=True,
        type=str)
    psr.add_argument(
        "--simulation", help="The path to a simulation to analyze.", required=True,
        type=str)
    psr.add_argument(
        "--warmup-percent", default=train.DEFAULTS["warmup_percent"],
        help=("The percent of each simulation's datapoint to drop from the "
              "beginning."),
        type=float)
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
    sim_flp = args.simulation
    warmup_prc = args.warmup_percent
    scl_prms_flp = args.scale_params
    out_dir = args.out_dir
    assert path.exists(mdl_flp), f"Model file does not exist: {mdl_flp}"
    assert path.exists(sim_flp), f"Simulation file does not exist: {sim_flp}"
    assert 0 <= warmup_prc < 100, \
        ("\"warmup_percent\" must be in the range [0, 100), but is: "
         f"{warmup_prc}")
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

    # Load and parse the simulation.
    (dat_in, dat_out, dat_out_raw, dat_out_oracle, _), sim = (
        train.process_sim(
            idx=0, total=1, net=net, sim_flp=sim_flp, tmp_dir=out_dir,
            warmup_prc=warmup_prc, keep_prc=100, sequential=True))

    # Load and apply the scaling parameters.
    with open(scl_prms_flp, "r") as fil:
        scl_prms = json.load(fil)
    # Remove "loss event rate sqrt" from the scaling parameters.
    scl_prms = [scl_prms_ for idx, scl_prms_ in enumerate(scl_prms)]  # if idx != 103]
    dat_in = utils.scale_all(dat_in, scl_prms, 0, 1, args.standardize)

    # Visualize the ground truth data.
    utils.visualize_classes(net, dat_out, isinstance(net, models.SvmWrapper))

    # Test the simulation.
    net.test(
        *utils.Dataset(
            fets=dat_in.dtype.names,
            dat_in=utils.clean(dat_in),
            dat_out=utils.clean(dat_out),
            dat_out_raw=utils.clean(dat_out_raw),
            dat_out_oracle=utils.clean(dat_out_oracle),
            num_flws=np.array(
                [sim.unfair_flws + sim.other_flws] * dat_in.shape[0], dtype=float)).raw(),
        graph_prms={
            "out_dir": out_dir,
            "sort_by_unfairness": False,
            "dur_s": sim.dur_s
        })


if __name__ == "__main__":
    main()
