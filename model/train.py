#!/usr/bin/env python3
"""
Based on:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
"""

import argparse
import json
import math
import multiprocessing
import os
from os import path
import pickle
import random
import sys
import time

import numpy as np
import torch

import models
import utils


# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup": 0,
    "num_sims": sys.maxsize,
    "sims": [],
    "model": models.MODEL_NAMES[0],
    "features": [],
    "epochs": 100,
    "num_gpus": 0,
    "train_batch": sys.maxsize,
    "test_batch": sys.maxsize,
    "learning_rate": 0.001,
    "momentum": 0.09,
    "kernel": "linear",
    "degree": 3,
    "penalty": "l1",
    "max_iter": 10000,
    "graph": False,
    "standardize": True,
    "early_stop": False,
    "val_patience": 10,
    "val_improvement_thresh": 0.1,
    "conf_trials": 1,
    "max_attempts": 10,
    "no_rand": False,
    "timeout_s": 0,
    "out_dir": ".",
    "regen_data": False,
    "sync": False
}
# The maximum number of epochs when using early stopping.
EPCS_MAX = 10_000
# The threshold of the new throughout to the old throughput above which a
# a training example will not be considered. I.e., the throughput must have
# decreased to less than this fraction of the original throughput for a
# training example to be considered.
NEW_TPT_TSH = 0.925
# The random seed.
SEED = 1337
# Set to false to parse the simulations in sorted order.
SHUFFLE = True
# The number of times to log progress during one epoch.
LOGS_PER_EPC = 5
# The number of validation passes per epoch.
VALS_PER_EPC = 15
# Whether to parse simulation files synchronously or in parallel.
SYNC = False


def scale_fets(dat, scl_grps, standardize=False):
    """
    Returns a copy of dat with the columns normalized. If standardize
    is True, then the scaling groups are normalized to a mean of 0 and
    a variance of 1. If standardize is False, then the scaling groups
    are normalized to the range [0, 1]. Also returns an array of shape
    (dat_all[0].shape[1], 2) where row i contains the scaling
    parameters of column i in dat. If standardize is True, then the
    scaling parameters are the mean and standard deviation of that
    column's scaling group. If standardize is False, then the scaling
    parameters are the min and max of that column's scaling group.
    """
    fets = dat.dtype.names
    assert fets is not None, \
        f"The provided array is not structured. dtype: {dat.dtype.descr}"
    assert len(scl_grps) == len(fets), \
        f"Invalid scaling groups ({scl_grps}) for dtype ({dat.dtype.descr})!"

    # Determine the unique scaling groups.
    scl_grps_unique = set(scl_grps)
    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each scaling group.
    scl_grps_prms = np.empty((len(scl_grps_unique), 2), dtype="float64")
    # Function to reduce a structured array.
    rdc = (lambda fnc, arr:
           fnc(np.array([fnc(arr[fet]) for fet in arr.dtype.names if fet != ""])))
    # Determine the min and the max of each scaling group.
    for scl_grp in scl_grps_unique:
        # Determine the features in this scaling group.
        scl_grp_fets = [fet for fet_idx, fet in enumerate(fets)
                        if scl_grps[fet_idx] == scl_grp]
        # Extract the columns corresponding to this scaling group.
        fet_values = dat[scl_grp_fets]
        # Record the min and max of these columns.
        scl_grps_prms[scl_grp] = [
            np.mean(utils.clean(fet_values)) if standardize else rdc(np.min, fet_values),
            np.std(utils.clean(fet_values)) if standardize else rdc(np.max, fet_values)
        ]

    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each column (i.e., feature).
    scl_prms = np.empty((len(fets), 2), dtype="float64")
    # Create an empty array to hold the rescaled features.
    new = np.empty(dat.shape, dtype=dat.dtype)
    # Rescale each feature based on its scaling group's min and max.
    for fet_idx, fet in enumerate(fets):
        # Look up the parameters for this feature's scaling group.
        prm_1, prm_2 = scl_grps_prms[scl_grps[fet_idx]]
        # Store this min and max in the list of per-column scaling parameters.
        scl_prms[fet_idx] = np.array([prm_1, prm_2])
        fet_values = dat[fet]
        if standardize:
            # prm_1 is the mean and prm_2 is the standard deviation.
            scaled = (
                # Handle the rare case where the standard deviation is
                # 0 (meaning that all of the feature values are the
                # same).
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_2 == 0
                else (fet_values - prm_1) / prm_2)
        else:
            # prm_1 is the min and prm_2 is the max.
            scaled = (
                # Handle the rare case where the min and the max are
                # the same (meaning that all of the feature values are
                # the same.
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_1 == prm_2
                else utils.scale(
                    fet_values, prm_1, prm_2, min_out=0, max_out=1))
        new[fet] = scaled

    return new, scl_prms


def process_sim(idx, total, net, sim_flp, warmup, sequential=False):
    """
    Loads and processes data from a single simulation. Drops the first
    "warmup" packets. Uses "net" to determine the relevant input and
    output features. Returns a tuple of numpy arrays of the form:
    (input data, output data).
    """
    sim, dat = utils.load_sim(
        sim_flp, msg=f"{idx + 1:{f'0{len(str(total))}'}}/{total}")
    # Drop the first few packets so that we consider steady-state behavior only.
    assert dat.shape[0] > warmup, f"{sim_flp}: Unable to drop first {warmup} packets!"
    dat = dat[warmup:]
    # Split each data matrix into two separate matrices: one with the input
    # features only and one with the output features only. The names of the
    # columns correspond to the feature names in in_spc and out_spc.
    assert net.in_spc, "{sim_flp}: Empty in spec."
    assert net.out_spc, "{sim_flp}: Empty out spec."
    dat_in = dat[net.in_spc]
    dat_out = dat[net.out_spc]
    # Convert output features to class labels.
    dat_out_raw = dat_out
    dat_out = net.convert_to_class(sim, dat_out)

    # If the results contains NaNs or Infs, then discard this
    # simulation.
    def has_non_finite(arr):
        for fet in arr.dtype.names:
            if not np.isfinite(arr[fet]).all():
                print(f"    Simulation {sim_flp} has NaNs of Infs in feature {fet}")
                return True
        return False
    if has_non_finite(dat_in) or has_non_finite(dat_out):
        return None

    # Verify data.
    assert dat_in.shape[0] == dat_out.shape[0], \
        f"{sim_flp}: Input and output should have the same number of rows."
    # Find the uniques classes in the output features and make sure
    # that they are properly formed. Assumes that dat_out is a
    # structured numpy array containing a column named "class".
    for cls in set(dat_out["class"].tolist()):
        assert 0 <= cls < net.num_clss, f"Invalid class: {cls}"

    # Transform the data as required by this specific model.
    return (
        net.modify_data(
            sim, dat_in, dat_out, dat_out_raw,
            # Must put the column name in a list for the result to be
            # a structured array.
            dat_out_oracle=dat[["mathis model label-ewma-alpha0.5"]],
            sequential=sequential),
        sim)


def make_datasets(net, args):
    """
    Parses the simulation files in data_dir and transforms them (e.g., by
    scaling) into the correct format for the network.

    If num_sims is not None, then this function selects the first num_sims
    simulations only. If shuffle is True, then the simulations will be parsed in
    sorted order. Use num_sims and shuffle=True together to simplify debugging.
    """
    sims = args["sims"]
    if not sims:
        dat_dir = args["data_dir"]
        sims = [path.join(dat_dir, sim) for sim in sorted(os.listdir(dat_dir))]
    if SHUFFLE:
        random.shuffle(sims)
    num_sims = args["num_sims"]
    if num_sims is not None:
        num_sims_actual = len(sims)
        assert num_sims_actual >= num_sims, \
            (f"Insufficient simulations. Requested {num_sims}, but only "
             f"{num_sims_actual} availabled.")
        sims = sims[:num_sims]

    tot_sims = len(sims)
    print(f"Found {tot_sims} simulations.")
    sims_args = [(idx, tot_sims, net, sim, args["warmup"])
                 for idx, sim in enumerate(sims)]
    if SYNC or args["sync"]:
        dat_all = [process_sim(*sim_args) for sim_args in sims_args]
    else:
        with multiprocessing.Pool() as pol:
            # Each element of dat_all corresponds to a single simulation.
            dat_all = pol.starmap(process_sim, sims_args)
    # Throw away results from simulations that could not be parsed.
    dat_all = [dat for dat in dat_all if dat is not None]
    print(f"Discarded {tot_sims - len(dat_all)} simulations!")
    assert dat_all, "No valid simulations found!"

    dat_all, sims = zip(*dat_all)

    # Validate data.
    dim_in = None
    dtype_in = None
    dim_out = None
    dtype_out = None
    scl_grps = None
    for dat_in, dat_out, _, _, scl_grps_cur in dat_all:
        dim_in_cur = len(dat_in.dtype.names)
        dim_out_cur = len(dat_out.dtype.names)
        dtype_in_cur = dat_in.dtype
        dtype_out_cur = dat_out.dtype
        if dim_in is None:
            dim_in = dim_in_cur
        if dim_out is None:
            dim_out = dim_out_cur
        if dtype_in is None:
            dtype_in = dtype_in_cur
        if dtype_out is None:
            dtype_out = dtype_out_cur
        if scl_grps is None:
            scl_grps = scl_grps_cur
        assert dim_in_cur == dim_in, \
            f"Invalid input feature dim: {dim_in_cur} != {dim_in}"
        assert dim_out_cur == dim_out, \
            f"Invalid output feature dim: {dim_out_cur} != {dim_out}"
        assert dtype_in_cur == dtype_in, \
            f"Invalud input dtype: {dtype_in_cur} != {dtype_in}"
        assert dtype_out_cur == dtype_out, \
            f"Invalid output dtype: {dtype_out_cur} != {dtype_out}"
        assert scl_grps_cur == scl_grps, \
            f"Invalid scaling groups: {scl_grps_cur} != {scl_grps}"
    assert dim_in is not None, "Unable to compute input feature dim!"
    assert dim_out is not None, "Unable to compute output feature dim!"
    assert dtype_in is not None, "Unable to compute input dtype!"
    assert dtype_out is not None, "Unable to compute output dtype!"
    assert scl_grps is not None, "Unable to compte scaling groups!"

    # Build combined feature lists.
    dat_in_all, dat_out_all, dat_out_all_raw, dat_out_all_oracle, _ = zip(*dat_all)
    # Determine the number of flows in each example.
    num_flws = [sim.unfair_flws + sim.other_flws for sim in sims]
    num_flws = [np.array([num_flws_] * dat_in.shape[0], dtype=[("num_flws", "int")])
                for num_flws_, dat_in in zip(num_flws, dat_in_all)]
    num_flws = np.concatenate(num_flws, axis=0)
    # Stack the arrays.
    dat_in_all = np.concatenate(dat_in_all, axis=0)
    dat_out_all = np.concatenate(dat_out_all, axis=0)
    dat_out_all_raw = np.concatenate(dat_out_all_raw, axis=0)
    dat_out_all_oracle = np.concatenate(dat_out_all_oracle, axis=0)

    # Convert all instances of -1 (feature value unknown) to the mean
    # for that feature.
    bad_fets = []
    for fet in dat_in_all.dtype.names:
        fet_values = dat_in_all[fet]
        if (fet_values == -1).all():
            bad_fets.append(fet)
            continue
        dat_in_all[fet] = np.where(
            fet_values == -1, np.mean(fet_values), fet_values)
        assert (dat_in_all[fet] != -1).all(), f"Found \"-1\" in feature: {fet}"
    assert not bad_fets, f"Features contain only \"-1\": {bad_fets}"

    # Scale input features. Do this here instead of in process_sim()
    # because all of the features must be scaled using the same
    # parameters.
    dat_in_all, prms_in = scale_fets(dat_in_all, scl_grps, args["standardize"])

    # # Check if any of the data is malformed and discard features if
    # # necessary.
    # fets = []
    # for fet in dat_in_all.dtype.names:
    #     fet_values = dat_in_all[fet]
    #     if ((not np.isnan(fet_values).any()) and
    #             (not np.isinf(fet_values).any())):
    #         fets.append(fet)
    #     else:
    #         print(f"Discarding: {fet}")
    # dat_in_all = dat_in_all[fets]

    return (dat_in_all, dat_out_all, dat_out_all_raw, dat_out_all_oracle, num_flws,
            prms_in)


def split_data(fets, net, dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws,
               bch_trn, bch_tst, use_val=False):
    """
    Divides the input and output data into training, validation, and
    testing sets and constructs data loaders.
    """
    print("Creating train/val/test data...")
    #assert len(dat_out.shape) == 1
    #assert len(dat_out_raw.shape) == 1
    #assert len(dat_out_oracle.shape) == 1
    #assert len(num_flws.shape) == 1

    #fets = dat_in.dtype.names
    # Destroy columns names to make merging the matrices easier. I.e.,
    # convert from structured to regular numpy arrays.
    # dat_in = utils.clean(dat_in)
    # dat_out = utils.clean(dat_out)
    # dat_out_raw = utils.clean(dat_out_raw)
    # dat_out_oracle = utils.clean(dat_out_oracle)
    # num_flws = utils.clean(num_flws)
    # Shuffle the data to ensure that the training, validation, and
    # test sets are uniformly sampled. To shuffle all of the arrays
    # together, we must first merge them into a combined matrix.
    num_cols_in = dat_in.shape[1]
    merged = np.concatenate(
        (dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws), axis=1)
    np.random.shuffle(merged)
    dat_in = merged[:, :num_cols_in]
    dat_out = merged[:, num_cols_in]
    dat_out_raw = merged[:, num_cols_in + 1]
    dat_out_oracle = merged[:, num_cols_in + 2]
    num_flws = merged[:, num_cols_in + 3]

    # 50% for training, 20% for validation, 30% for testing.
    num_exps = dat_in.shape[0]
    num_val = int(round(num_exps * 0.2)) if use_val else 0
    num_tst = int(round(num_exps * 0.3))
    print((f"    Data - train: {num_exps - num_val - num_tst}, val: {num_val}, "
           f"test: {num_tst}"))
    # Validation.
    dat_val_in = dat_in[:num_val]
    dat_val_out = dat_out[:num_val]
    # Testing.
    dat_tst_in = dat_in[num_val:num_val + num_tst]
    dat_tst_out = dat_out[num_val:num_val + num_tst]
    dat_tst_out_raw = dat_out_raw[num_val:num_val + num_tst]
    dat_tst_out_oracle = dat_out_oracle[num_val:num_val + num_tst]
    num_flws_tst = num_flws[num_val:num_val + num_tst]
    # Training.
    dat_trn_in = dat_in[num_val + num_tst:]
    dat_trn_out = dat_out[num_val + num_tst:]

    # Create the dataloaders.
    dataset_trn = utils.Dataset(fets, dat_trn_in, dat_trn_out)
    ldr_trn = (
        torch.utils.data.DataLoader(
            dataset_trn, batch_size=bch_tst, shuffle=True, drop_last=False)
        if isinstance(net, models.SvmSklearnWrapper)
        else torch.utils.data.DataLoader(
            dataset_trn,
            batch_sampler=utils.BalancedSampler(
                dataset_trn, bch_trn, drop_last=False)))
    ldr_val = (
        torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_val_in, dat_val_out), batch_size=bch_tst,
            shuffle=False, drop_last=False)
        if use_val else None)
    ldr_tst = torch.utils.data.DataLoader(
        utils.Dataset(
            fets, dat_tst_in, dat_tst_out, dat_tst_out_raw, dat_tst_out_oracle,
            num_flws_tst),
        batch_size=bch_tst, shuffle=False, drop_last=False)
    return ldr_trn, ldr_val, ldr_tst


def init_hidden(net, bch, dev):
    """
    Initialize the hidden state. The hidden state is what gets built
    up over time as the LSTM processes a sequence. It is specific to a
    sequence, and is different than the network's weights. It needs to
    be reset for every new sequence.
    """
    hidden = net.init_hidden(bch)
    hidden[0].to(dev)
    hidden[1].to(dev)
    return hidden


def inference(ins, labs, net_raw, dev,
              hidden=(torch.zeros(()), torch.zeros(())), los_fnc=None):
    """
    Runs a single inference pass. Returns the output of net, or the
    loss if los_fnc is not None.
    """
    # Move input and output data to the proper device.
    ins = ins.to(dev)
    labs = labs.to(dev)

    if isinstance(net_raw, models.Lstm):
        # LSTMs want the sequence length to be first and the batch
        # size to be second, so we need to flip the first and
        # second dimensions:
        #   (batch size, sequence length, LSTM.in_dim) to
        #   (sequence length, batch size, LSTM.in_dim)
        ins = ins.transpose(0, 1)
        # Reduce the labels to a 1D tensor.
        # TODO: Explain this better.
        labs = labs.transpose(0, 1).view(-1)
        # The forward pass.
        out, hidden = net_raw(ins, hidden)
    else:
        # The forward pass.
        out = net_raw(ins)
    if los_fnc is None:
        return out, hidden
    return los_fnc(out, labs), hidden


def train(net, num_epochs, ldr_trn, ldr_val, dev, ely_stp, val_pat_max, out_flp,
          val_imp_thresh, tim_out_s, opt_params):
    """ Trains a model. """
    print("Training...")
    los_fnc = net.los_fnc()
    opt = net.opt(net.net.parameters(), **opt_params)
    # If using early stopping, then this is the lowest validation loss
    # encountered so far.
    los_val_min = None
    # If using early stopping, then this tracks the *remaining* validation
    # patience (initially set to the maximum amount of patience). This is
    # decremented for every validation pass that does not improve the
    # validation loss by at least val_imp_thresh percent. When this reaches
    # zero, training aborts.
    val_pat = val_pat_max
    # The number of batches per epoch.
    num_bchs_trn = len(ldr_trn)
    # Print a lot statement every few batches.
    if LOGS_PER_EPC == 0:
        # Disable logging.
        bchs_per_log = sys.maxsize
    else:
        bchs_per_log = math.ceil(num_bchs_trn / LOGS_PER_EPC)
    # Perform a validation pass every few batches.
    assert not ely_stp or VALS_PER_EPC > 0, \
        f"Early stopping configured with erroneous VALS_PER_EPC: {VALS_PER_EPC}"
    bchs_per_val = math.ceil(num_bchs_trn / VALS_PER_EPC)
    if ely_stp:
        print(f"Will validate after every {bchs_per_val} batches.")

    tim_srt_s = time.time()
    # Loop over the dataset multiple times...
    for epoch_idx in range(num_epochs):
        tim_del_s = time.time() - tim_srt_s
        if tim_out_s != 0 and tim_del_s > tim_out_s:
            print((f"Training timed out after after {epoch_idx} epochs "
                   f"({tim_del_s:.2f} seconds)."))
            break

        # For each batch...
        for bch_idx_trn, (ins, labs) in enumerate(ldr_trn, 0):
            if bch_idx_trn % bchs_per_log == 0:
                print(f"Epoch: {epoch_idx + 1:{f'0{len(str(num_epochs))}'}}/"
                      f"{'?' if ely_stp else num_epochs}, "
                      f"batch: {bch_idx_trn + 1:{f'0{len(str(num_bchs_trn))}'}}/"
                      f"{num_bchs_trn}", end=" ")
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
            # Zero out the parameter gradients.
            opt.zero_grad()
            loss, hidden = inference(ins, labs, net.net, dev, hidden, los_fnc)
            # The backward pass.
            loss.backward()
            opt.step()
            if bch_idx_trn % bchs_per_log == 0:
                print(f"    Training loss: {loss:.5f}")

            # Run on validation set, print statistics, and (maybe) checkpoint
            # every VAL_PER batches.
            if ely_stp and not bch_idx_trn % bchs_per_val:
                print("    Validation pass:")
                # For efficiency, convert the model to evaluation mode.
                net.net.eval()
                with torch.no_grad():
                    los_val = 0
                    for bch_idx_val, (ins_val, labs_val) in enumerate(ldr_val):
                        print(f"    Validation batch: {bch_idx_val + 1}/{len(ldr_val)}")
                        # Initialize the hidden state for every new sequence.
                        hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
                        los_val += inference(
                            ins_val, labs_val, net.net, dev, hidden, los_fnc)[0].item()
                # Convert the model back to training mode.
                net.net.train()

                if los_val_min is None:
                    los_val_min = los_val
                # Calculate the percent improvement in the validation loss.
                prc = (los_val_min - los_val) / los_val_min * 100
                print(f"    Validation error improvement: {prc:.2f}%")

                # If the percent improvement in the validation loss is greater
                # than a small threshold, then take this as the new best version
                # of the model.
                if prc > val_imp_thresh:
                    # This is the new best version of the model.
                    los_val_min = los_val
                    # Reset the validation patience.
                    val_pat = val_pat_max
                    # Save the new best version of the model. Convert the
                    # model to Torch Script first.
                    torch.jit.save(torch.jit.script(net.net), out_flp)
                else:
                    val_pat -= 1
                    if path.exists(out_flp):
                        # Resume training from the best model.
                        net.net = torch.jit.load(out_flp)
                        net.net.to(dev)
                if val_pat <= 0:
                    print(f"Stopped after {epoch_idx + 1} epochs")
                    return net
    if not ely_stp:
        # Save the final version of the model. Convert the model to Torch Script
        # first.
        print(f"Saving final model: {out_flp}")
        torch.jit.save(torch.jit.script(net.net), out_flp)
    return net


def test(net, ldr_tst, dev):
    """ Tests a model. """
    print("Testing...")
    # The number of testing samples that were predicted correctly.
    num_correct = 0
    # Total testing samples.
    total = 0
    num_bchs_tst = len(ldr_tst)
    # For efficiency, convert the model to evaluation mode.
    net.net.eval()
    with torch.no_grad():
        for bch_idx, (ins, labs) in enumerate(ldr_tst):
            print(f"Test batch: {bch_idx + 1:{f'0{len(str(num_bchs_tst))}'}}/"
                  f"{num_bchs_tst}")
            if isinstance(net, models.LstmWrapper):
                bch_tst, seq_len, _ = ins.size()
            else:
                bch_tst, _ = ins.size()
                seq_len = 1
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=bch_tst, dev=dev)
            # Run inference. The first element of the output is the
            # number of correct predictions.
            num_correct += inference(
                ins, labs, net.net, dev, hidden, los_fnc=net.check_output)[0]
            total += bch_tst * seq_len
    # Convert the model back to training mode.
    net.net.train()
    acc_tst = num_correct / total
    print(f"Test accuracy: {acc_tst * 100:.2f}%")
    return acc_tst


def run_sklearn(args, dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws,
                out_flp):
    """
    Trains an sklearn model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Construct the model.
    print("Building model...")
    net = models.MODELS[args["model"]]()
    net.new(**{param: args[param] for param in net.params})
    # Split the data into training, validation, and test loaders.
    ldr_trn, _, ldr_tst = split_data(
        args["features"], net, dat_in, dat_out, dat_out_raw, dat_out_oracle,
        num_flws, args["train_batch"], args["test_batch"])
    # Training.
    print("Training...")
    tim_srt_s = time.time()
    net.train(*(ldr_trn.dataset.raw()[1:3]))
    print(f"Finished training - time: {time.time() - tim_srt_s:.2f} seconds")
    # Save the model.
    print(f"Saving final model: {out_flp}")
    with open(out_flp, "wb") as fil:
        pickle.dump(net.net, fil)
    # Testing.
    print("Testing...")
    tim_srt_s = time.time()
    los_tst = net.test(*ldr_tst.dataset.raw())
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")
    return los_tst


def run_torch(args, dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws,
              out_flp):
    """
    Trains a PyTorch model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Instantiate and configure the network. Move it to the proper device.
    net = models.MODELS[args["model"]]()
    net.new()
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net.net = torch.nn.DataParallel(net.net)
    dev = torch.device("cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.net.to(dev)

    # Split the data into training, validation, and test loaders.
    ldr_trn, ldr_val, ldr_tst = split_data(
        net, dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws,
        args["train_batch"], args["test_batch"])

    # Explicitly move the training (and maybe validation) data to the target
    # device.
    ldr_trn.dataset.to(dev)
    ely_stp = args["early_stop"]
    if ely_stp:
        ldr_val.dataset.to(dev)

    # Training.
    tim_srt_s = time.time()
    net = train(
        net, args["epochs"], ldr_trn, ldr_val, dev, args["early_stop"],
        args["val_patience"], out_flp, args["val_improvement_thresh"],
        args["timeout_s"],
        opt_params={param: args[param] for param in net.params})
    print(f"Finished training - time: {time.time() - tim_srt_s:.2f} seconds")

    # Explicitly delete the training and validation data so that they are
    # removed from the target device.
    del ldr_trn
    del ldr_val
    # This is necessary for the GPU memory to be released.
    torch.cuda.empty_cache()

    # Read the best version of the model from disk.
    net.net = torch.jit.load(out_flp)
    net.net.to(dev)

    # Testing.
    ldr_tst.dataset.to(dev)
    tim_srt_s = time.time()
    los_tst = test(net, ldr_tst, dev)
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")
    del ldr_tst
    torch.cuda.empty_cache()
    return los_tst


def run_many(args_):
    """
    Run args["conf_trials"] trials and survive args["max_attempts"] failed
    attempts.
    """
    # Initially, accept all default values. Then, override the defaults with
    # any manually-specified values. This allows the caller to specify values
    # only for parameters that they care about while ensuring that all
    # parameters have values.
    args = DEFAULTS
    args.update(args_)
    if args["early_stop"]:
        args["epochs"] = EPCS_MAX
    degree = args["degree"]
    assert degree >= 0, \
        ("\"degree\" must be an integer greater than or equal to 0, but is: "
         f"{degree}")
    max_iter = args["max_iter"]
    assert max_iter > 0, \
        f"\"max_iter\" must be greater than 0, but is: {max_iter}"
    print(f"Arguments: {args}")

    if args["no_rand"]:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    out_dir = args["out_dir"]
    if not path.isdir(out_dir):
        print(f"Output directory does not exist. Creating it: {out_dir}")
        os.makedirs(out_dir)
    net_tmp = models.MODELS[args["model"]]()
    # Verify that the necessary supplemental parameters are present.
    for param in net_tmp.params:
        assert param in args, f"\"{param}\" not in args: {args}"
    # Assemble the output filepath.
    out_flp = path.join(
        args["out_dir"],
        (utils.args_to_str(args, order=sorted(DEFAULTS.keys()))
        ) + (
            # Determine the proper extension based on the type of
            # model.
            ".pickle" if isinstance(net_tmp, models.SvmSklearnWrapper)
            else ".pth"))
    # If custom features are specified, then overwrite the model's
    # default features.
    fets = args["features"]
    if fets:
        net_tmp.in_spc = fets
    else:
        assert "arrival time us" not in args["features"]
        args["features"] = net_tmp.in_spc
    # If a trained model file already exists, then delete it.
    if path.exists(out_flp):
        os.remove(out_flp)

    # Load or geenrate training data.
    dat_flp = path.join(out_dir, "data.npz")
    scl_prms_flp = path.join(out_dir, "scale_params.json")
    # Check for the presence of both the data and the scaling
    # parameters because the resulting model is useless without the
    # proper scaling parameters.
    if (not args["regen_data"] and path.exists(dat_flp) and
            path.exists(scl_prms_flp)):
        print("Found existing data!")
        dat = np.load(dat_flp)
        assert "in" in dat.files and "out" in dat.files, \
            f"Improperly formed data file: {dat_flp}"
        dat_in = dat["in"]
        dat_out = dat["out"]
        dat_out_raw = dat["out_raw"]
        dat_out_oracle = dat["out_oracle"]
        num_flws = dat["num_flws"]
        dat_in_shape = dat_in.shape
        dat_out_shape = dat_out.shape
        assert dat_in_shape[0] == dat_out_shape[0], \
            f"Data has invalid shapes! in: {dat_in_shape}, out: {dat_out_shape}"
    else:
        print("Regenerating data...")
        dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws, scl_prms = (
            make_datasets(net_tmp, args))
        # Clean the data before saving it to fix an error in
        # savez_compressed().
        dat_in = utils.clean(dat_in)
        dat_out = utils.clean(dat_out)
        dat_out_raw = utils.clean(dat_out_raw)
        dat_out_oracle = utils.clean(dat_out_oracle)
        num_flws = utils.clean(num_flws)
        # Save the processed data so that we do not need to process it again.
        print(f"Saving data: {dat_flp}")
        np.savez_compressed(
            dat_flp,
            **{"in": dat_in, "out": dat_out, "out_raw": dat_out_raw,
               "out_oracle": dat_out_oracle, "num_flws": num_flws})
        # Save scaling parameters.
        print(f"Saving scaling parameters: {scl_prms_flp}")
        with open(scl_prms_flp, "w") as fil:
            json.dump(scl_prms.tolist(), fil)

    # Visualaize the ground truth data.
    utils.visualize_classes(
        net_tmp, dat_out, isinstance(net_tmp, models.SvmWrapper))

    # TODO: Parallelize attempts.
    trls = args["conf_trials"]
    apts = 0
    apts_max = args["max_attempts"]
    ress = []
    while trls > 0 and apts < apts_max:
        apts += 1
        res = (run_sklearn if isinstance(net_tmp, models.SvmSklearnWrapper) else run_torch)(
            args, dat_in, dat_out, dat_out_raw, dat_out_oracle, num_flws, out_flp)
        if res == 100:
            print(
                (f"Training failed (attempt {apts}/{apts_max}). Trying again!"))
        else:
            ress.append(res)
            trls -= 1
    if ress:
        print(("Resulting accuracies: "
               f"{', '.join([f'{res:.2f}' for res in ress])}"))
        max_acc = max(ress)
        print(f"Maximum accuracy: {max_acc:.2f}")
        # Return the minimum error instead of the maximum accuracy.
        return 1 - max_acc
    print(f"Model cannot be trained with args: {args}")
    return float("inf")


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="An LSTM training framework.")
    psr.add_argument(
        "--data-dir",
        help=("The path to a directory containing the"
              "training/validation/testing data (required)."),
        required=True, type=str)
    psr.add_argument(
        "--warmup", default=DEFAULTS["warmup"],
        help=("The number of packets to drop from the beginning of each "
              "simulation."),
        type=int)
    psr.add_argument(
        "--num-sims", default=DEFAULTS["num_sims"],
        help="The number of simulations to consider.", type=int)
    psr.add_argument(
        "--model", choices=models.MODEL_NAMES, default=DEFAULTS["model"],
        help="The model to use.", type=str)
    psr.add_argument(
        "--epochs", default=DEFAULTS["epochs"],
        help="The number of epochs to train for.", type=int)
    psr.add_argument(
        "--num-gpus", default=DEFAULTS["num_gpus"],
        help="The number of GPUs to use.", type=int)
    psr.add_argument(
        "--train-batch", default=DEFAULTS["train_batch"],
        help="The batch size to use during training.", type=int)
    psr.add_argument(
        "--test-batch", default=DEFAULTS["test_batch"],
        help="The batch size to use during validation and testing.", type=int)
    psr.add_argument(
        "--learning-rate", default=DEFAULTS["learning_rate"],
        help="Learning rate for SGD training.", type=float)
    psr.add_argument(
        "--momentum", default=DEFAULTS["momentum"],
        help="Momentum for SGD training.", type=float)
    psr.add_argument(
        "--kernel", default=DEFAULTS["kernel"],
        choices=["linear", "poly", "rbf", "sigmoid"],
        help=("If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type kernel. Ignored otherwise."),
        type=str)
    psr.add_argument(
        "--degree", default=DEFAULTS["degree"],
        help=("If the model is of type \"{models.SvmSklearnWrapper().name()}\" "
              "and \"--kernel=poly\", then this is the degree of the polynomial "
              "that will be fit. Ignored otherwise."),
        type=int)
    psr.add_argument(
        "--penalty", default=DEFAULTS["penalty"], choices=["l1", "l2"],
        help=(f"If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type of regularization. Ignored otherwise."))
    psr.add_argument(
        "--max-iter", default=DEFAULTS["max_iter"],
        help=("If the model is an sklearn model, then this is the maximum "
              "number of iterations to use during the fitting process. Ignored "
              "otherwise."),
        type=int)
    psr.add_argument(
        "--graph", action="store_true",
        help=("If the model is an sklearn model, then analyze and graph the "
              "testing results."))
    psr.add_argument(
        "--standardize", action="store_true",
        help=("Standardize the data so that it has a mean of 0 and a variance "
              "of 1. Otherwise, data will be rescaled to the range [0, 1]."))
    psr.add_argument(
        "--early-stop", action="store_true", help="Enable early stopping.")
    psr.add_argument(
        "--val-patience", default=DEFAULTS["val_patience"],
        help=("The number of times that the validation loss can increase "
              "before training is automatically aborted."),
        type=int)
    psr.add_argument(
        "--val-improvement-thresh", default=DEFAULTS["val_improvement_thresh"],
        help="Threshold for percept improvement in validation loss.",
        type=float)
    psr.add_argument(
        "--conf-trials", default=DEFAULTS["conf_trials"],
        help="The number of trials to run.", type=int)
    psr.add_argument(
        "--max-attempts", default=DEFAULTS["max_attempts"],
        help="The maximum number of failed training attempts to survive.",
        type=int)
    psr.add_argument(
        "--no-rand", action="store_true", help="Use a fixed random seed.")
    psr.add_argument(
        "--timeout-s", default=DEFAULTS["timeout_s"],
        help="Automatically stop training after this amount of time (seconds).",
        type=float)
    psr.add_argument(
        "--out-dir", default=DEFAULTS["out_dir"],
        help="The directory in which to store output files.", type=str)
    args = vars(psr.parse_args())
    # Verify that all arguments are reflected in DEFAULTS.
    for arg in args.keys():
        assert arg in DEFAULTS, f"Argument {arg} missing from DEFAULTS!"
    run_many(args)


if __name__ == "__main__":
    main()
