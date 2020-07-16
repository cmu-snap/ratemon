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
    psr = argparse.ArgumentParser(
        description="Hyper-parameter optimizer for train.py.")
    psr.add_argument(
        "--model", help="The path to a trained model file.", required=True,
        type=str)
    psr.add_argument(
        "--simulation", help="The path to a simulation to analyze.", required=True,
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
    sim_flp = args.simulation
    warmup = args.warmup
    scl_prms_flp = args.scale_params
    out_dir = args.out_dir
    assert path.exists(mdl_flp), f"Model file does not exist: {mdl_flp}"
    assert path.exists(sim_flp), f"Simulation file does not exist: {sim_flp}"
    assert warmup >= 0, f"Warmup cannot be negative, but is: {warmup}"
    assert path.exists(scl_prms_flp), \
        f"Scaling parameters file does not exist: {scl_prms_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # Parse the model filepath to determine the model type, and instantiate it.
    net = models.MODELS[
        dict(
            [k_v.split(":")
             for k_v in path.basename(mdl_flp).split(".")[0].split("-")]
        )["model"]]()
    # Load the model.
    if mdl_flp.endswith("pickle"):
        with open(mdl_flp, "rb") as fil:
            mdl = pickle.load(fil)
    elif mdl_flp.endswith("pth"):
        mdl = torch.jit.load(mdl_flp)
    else:
        raise Exception(f"Unknown model type: {mdl_flp}")
    net.net = mdl

    # Load and parse the simulation.
    (dat_in, dat_out, dat_out_raw, dat_out_oracle, _), sim = (
        train.process_sim(
            idx=0, total=1, net=net, sim_flp=sim_flp, warmup=warmup,
            sequential=True))
    # Load and apply the scaling parameters.
    with open(scl_prms_flp, "r") as fil:
        scl_prms = json.load(fil)
    dat_in = utils.scale_all(dat_in, scl_prms, 0, 1, args.standardize)
    # Test the simulation.
    net.test(
        *utils.Dataset(
            dat_in.dtype.names, utils.clean(dat_in), utils.clean(dat_out),
            utils.clean(dat_out_raw), utils.clean(dat_out_oracle),
            num_flws=np.array(
                [sim.unfair_flws + sim.other_flws] * dat_in.shape[0], dtype=float)
        ).raw())

    # TODO: For 1000 data, loss event rate sqrt is not part of the
    #       training data but it is part of the scaling parameters.
    # TODO: Implement sequential mode in __create_buckets().


if __name__ == "__main__":
    main()
