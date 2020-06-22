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
import random
import sys
import time

import numpy as np
import torch

import models
import utils


# Parameter defaults.
DEFAULTS = {
    "epochs": 100,
    "num_gpus": 0,
    "warmup": 1000,
    "num_sims": 10,
    "train_batch": 10,
    "test_batch": 10_000,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "early_stop": False,
    "val_patience": 10,
    "val_improvement_thresh": 0.1,
    "conf_trials": 1,
    "max_attempts": 10,
    "no_rand": False,
    "timeout_s": -1,
    "out_dir": "."
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
# Set to true to parse the simulations in sorted order.
SHUFFLE = False
# The number of times to log progress during one epoch.
LOGS_PER_EPC = 5
# The number of validation passes per epoch.
VALS_PER_EPC = 15


class Dataset(torch.utils.data.Dataset):
    """ A simple Dataset that wraps arrays of input and output features. """

    def __init__(self, dat_in, dat_out, dev):
        """
        dat_out is assumed to have only a single practical dimension (e.g.,
        (X,), or (X, 1)).
        """
        super(Dataset).__init__()
        shp_in = dat_in.shape
        shp_out = dat_out.shape
        assert shp_in[0] == shp_out[0], \
            "Mismatched dat_in ({shp_in}) and dat_out ({shp_out})!"
        # Convert the numpy arrays to Torch tensors.
        self.dat_in = torch.tensor(dat_in, dtype=torch.float)
        # Reshape the output into a 1D array first, because
        # CrossEntropyLoss expects a single value. The dtype must be
        # long because the loss functions expect longs.
        self.dat_out = torch.tensor(
            dat_out.reshape(shp_out[0]), dtype=torch.long)
        # Move the entire dataset to the target device. This will fail
        # if the device has insufficient memory.
        self.dat_in = self.dat_in.to(dev)
        self.dat_out = self.dat_out.to(dev)

    def __len__(self):
        """ Returns the number of items in this Dataset. """
        return len(self.dat_in)

    def __getitem__(self, idx):
        """ Returns a specific (input, output) pair from this Dataset. """
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat_in[idx], self.dat_out[idx]


def scale_fets(dat):
    """
    Returns a copy of dat with the columns scaled between 0 and
    1. Also returns an array of shape (dat_all[0].shape[1], 2) where
    row i contains the min and the max of column i in dat.
    """
    assert dat.dtype.names is not None, \
        f"The provided array is not structured. dtype: {dat.dtype.descr}"
    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each column (i.e., feature).
    fets = dat.dtype.names
    scl_prms = np.empty((len(fets), 2))
    # The new scaled array.
    new = np.empty(dat.shape, dtype=dat.dtype)
    # For all features...
    for j, fet in enumerate(fets):
        # Determine the min and max values of this feature.
        fet_values = dat[fet]
        min_in = fet_values.min()
        max_in = fet_values.max()
        scl_prms[j] = np.array([min_in, max_in])
        if min_in == max_in:
            # Handle the rare case where all of the feature values are the same.
            scaled = np.zeros(fet_values.shape, dtype=fet_values.dtype)
        else:
            scaled = utils.scale(fet_values, min_in, max_in, min_out=0, max_out=1)
        new[fet] = scaled
    return new, scl_prms


def process_sim(net, sim_flp, warmup):
    """
    Loads and processes data from a single simulation. Drops the first
    "warmup" packets. Uses "net" to determine the relevant input and
    output features. Returns a tuple of numpy arrays of the form:
    (input data, output data).
    """
    sim, dat = utils.load_sim(sim_flp)
    # Drop the first few packets so that we consider steady-state behavior only.
    assert dat.shape[0] > warmup, f"{sim_flp}: Unable to drop first {warmup} packets!"
    dat = dat[warmup:]
    # Split each data matrix into two separate matrices: one with the input
    # features only and one with the output features only. The names of the
    # columns correspond to the feature names in in_spc and out_spc.
    assert net.in_spc, "{sim_flp}: Empty in spec."
    assert net.out_spc, "{sim_flp}: Empty out spec."
    arr_times = dat["arrival time"]
    dat_in = dat[net.in_spc]
    dat_out = dat[net.out_spc]
    # Convert output features to class labels.
    dat_out = net.convert_to_class(sim, dat_out)

    # Verify data.
    assert dat_in.shape[0] == dat_out.shape[0], \
        f"{sim_flp}: Input and output should have the same number of rows."
    # Find the uniques classes in the output features and make sure
    # that they are properly formed. Assumes that dat_out is a
    # structured numpy array containing a column named "class".
    if not isinstance(net, models.SVM):
        for cls in set(dat_out["class"].tolist()):
            assert 0 <= cls < net.num_clss, "Invalid class: {cls}"

    # Transform the data as required by this specific model.
    return net.modify_data(sim, dat_in, dat_out, arr_times=arr_times)


def make_datasets(net, dat_dir, warmup, num_sims, shuffle):
    """
    Parses the simulation files in data_dir and transforms them (e.g., by
    scaling) into the correct format for the network.

    If num_sims is not None, then this function selects the first num_sims
    simulations only. If shuffle is True, then the simulations will be parsed in
    sorted order. Use num_sims and shuffle=True together to simplify debugging.
    """
    sims = sorted(os.listdir(dat_dir))
    if shuffle:
        random.shuffle(sims)
    if num_sims is not None:
        sims = sims[:num_sims]

    print(f"Found {len(sims)} simulations.")
    with multiprocessing.Pool() as pol:
        # Each element of dat_all corresponds to a single simulation.
        dat_all = pol.starmap(
            process_sim,
            [(net, path.join(dat_dir, sim), warmup) for sim in sims])

    # Validate data.
    dim_in = None
    dtype_in = None
    dim_out = None
    dtype_out = None
    for dat_in, dat_out in dat_all:
        dim_in_cur = len(dat_in.dtype.descr)
        dim_out_cur = len(dat_out.dtype.descr)
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
        assert dim_in_cur == dim_in, \
            f"Invalid input feature dim: {dim_in_cur} != {dim_in}"
        assert dim_out_cur == dim_out, \
            f"Invalid output feature dim: {dim_out_cur} != {dim_out}"
        assert dtype_in_cur == dtype_in, \
            f"Invalud input dtype: {dtype_in_cur} != {dtype_in}"
        assert dtype_out_cur == dtype_out, \
            f"Invalid output dtype: {dtype_out_cur} != {dtype_out}"
    assert dim_in is not None, "Unable to compute input feature dim!"
    assert dim_out is not None, "Unable to compute output feature dim!"
    assert dtype_in is not None, "Unable to compute input dtype!"
    assert dtype_out is not None, "Unable to compute output dtype!"

    # Build combined feature lists.
    dat_in_all, dat_out_all = zip(*dat_all)
    dat_in_all = np.concatenate(dat_in_all, axis=0)
    dat_out_all = np.concatenate(dat_out_all, axis=0)
    # Scale input features. Do this here instead of in process_sim()
    # because all of the features must be scaled using the same
    # parameters.
    dat_in_all, prms_in = scale_fets(dat_in_all)

    return dat_in_all, dat_out_all, prms_in


def split_data(dat_in, dat_out, bch_trn, bch_tst, dev):
    """
    Divides dat_in and dat_out into training, validation, and test
    sets and constructs data loaders.
    """
    print("Creating train/val/test data...")
    # Destroy columns names to make merging the matrices easier. I.e.,
    # convert from structured to regular numpy arrays.
    dat_in = utils.clean(dat_in)
    dat_out = utils.clean(dat_out)

    # Shuffle the data to ensure that the training, validation, and
    # test sets are uniformly sampled. To shuffle the input and output
    # data together, we must first merge them into a combined matrix.
    num_exps = dat_in.shape[0]
    num_cols_in = dat_in.shape[1]
    merged = np.empty((num_exps, num_cols_in + dat_out.shape[1]), dtype=float)
    merged[:, :num_cols_in] = dat_in
    merged[:, num_cols_in:] = dat_out
    np.random.shuffle(merged)
    dat_in = merged[:, :num_cols_in]
    dat_out = merged[:, num_cols_in:]

    # 50% for training, 20% for validation, 30% for testing.
    num_val = int(round(num_exps * 0.2))
    num_tst = int(round(num_exps * 0.3))
    print((f"    Data - train: {num_exps - num_val - num_tst}, val: {num_val}, "
           f"test: {num_tst}"))
    dat_val_in = dat_in[:num_val]
    dat_val_out = dat_out[:num_val]
    dat_tst_in = dat_in[num_val:num_val + num_tst]
    dat_tst_out = dat_out[num_val:num_val + num_tst]
    dat_trn_in = dat_in[num_val + num_tst:]
    dat_trn_out = dat_out[num_val + num_tst:]

    # Create the dataloaders.
    ldr_trn = torch.utils.data.DataLoader(
        Dataset(dat_trn_in, dat_trn_out, dev),
        batch_size=bch_trn, shuffle=True)
    ldr_val = torch.utils.data.DataLoader(
        Dataset(dat_val_in, dat_val_out, dev),
        batch_size=bch_tst, shuffle=False)
    ldr_tst = torch.utils.data.DataLoader(
        Dataset(dat_tst_in, dat_tst_out, dev),
        batch_size=bch_tst, shuffle=False)
    print("Done.")
    return ldr_trn, ldr_val, ldr_tst


def init_hidden(net, bch, dev):
    """
    Initialize the hidden state. The hidden state is what gets built
    up over time as the LSTM processes a sequence. It is specific to a
    sequence, and is different than the network's weights. It needs to
    be reset for every new sequence.
    """
    hidden = (net.module.init_hidden if isinstance(net, torch.nn.DataParallel)
              else net.init_hidden)(bch)
    if hidden is not None:
        hidden[0].to(dev)
        hidden[1].to(dev)
    return hidden


def inference(ins, labs, net, hidden=None, los_fnc=None):
    """
    Runs a single inference pass. Returns the output of net, or the
    loss if los_fnc is not None.
    """
    if net.is_lstm:
        # LSTMs want the sequence length to be first and the batch
        # size to be second, so we need to flip the first and
        # second dimensions:
        #   (batch size, sequence length, LSTM.in_dim) to
        #   (sequence length, batch size, LSTM.in_dim)
        ins = ins.transpose(0, 1)
        # Reduce the labels to a 1D tensor.
        # TODO: Explain this better.
        labs = labs.transpose(0, 1).view(-1)
    # The forwards pass.
    out, hidden = net(ins, hidden)
    if los_fnc is None:
        return out, hidden
    return los_fnc(out, labs), hidden


def train(net, num_epochs, ldr_trn, ldr_val, dev, ely_stp,
          val_pat_max, out_flp, lr, momentum, val_imp_thresh,
          tim_out_s):
    """ Trains a model. """
    print("Training...")
    los_fnc = net.los_fnc()
    opt = net.opt(net.parameters(), lr=lr)
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
    # Log the training process every few batches.
    bchs_per_log = math.floor(num_bchs_trn / LOGS_PER_EPC)
    # To avoid a divide-by-zero error below, the number of validation passes
    # per epoch much be at least 2.
    assert VALS_PER_EPC >= 2, "Must validate at least twice per epoch."
    if num_bchs_trn < VALS_PER_EPC:
        # If the number of batches per epoch is less than the desired number of
        # validation passes per epoch, then do a validation pass after every
        # batch.
        bchs_per_val = 1
    else:
        # Using floor() means that dividing by (VALS_PER_EPC - 1) will result in
        # VALS_PER_EPC validation passes per epoch.
        bchs_per_val = math.floor(num_bchs_trn / (VALS_PER_EPC - 1))
    if ely_stp:
        print(f"Will validate after every {bchs_per_val} batches.")

    tim_srt_s = time.time()
    # Loop over the dataset multiple times...
    for epoch_idx in range(num_epochs):
        tim_del_s = time.time() - tim_srt_s
        if tim_out_s != -1 and tim_del_s > tim_out_s:
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
            loss, hidden = inference(ins, labs, net, hidden, los_fnc)
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
                net.eval()
                with torch.no_grad():
                    los_val = 0
                    for bch_idx_val, (ins_val, labs_val) in enumerate(ldr_val):
                        print(f"    Validation batch: {bch_idx_val + 1}/{len(ldr_val)}")
                        # Initialize the hidden state for every new sequence.
                        hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
                        los_val += inference(
                            ins_val, labs_val, net, hidden, los_fnc)[0].item()
                # Convert the model back to training mode.
                net.train()

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
                    # # Save the new best version of the model. Convert the
                    # # model to Torch Script first.
                    # torch.jit.save(torch.jit.script(net), out_flp)
                else:
                    val_pat -= 1
                    # if path.exists(out_flp):
                    #     # Resume training from the best model.
                    #     net = torch.jit.load(out_flp)
                    #     net.to(dev)
                if val_pat <= 0:
                    print(f"Stopped after {epoch_idx + 1} epochs")
                    return net
    # if not ely_stp:
    #     # Save the final version of the model. Convert the model to Torch Script
    #     # first.
    #     print(f"Saving: {out_flp}")
    #     torch.jit.save(torch.jit.script(net), out_flp)
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
    net.eval()
    with torch.no_grad():
        for bch_idx, (ins, labs) in enumerate(ldr_tst):
            print(f"Test batch: {bch_idx + 1:{f'0{len(str(num_bchs_tst))}'}}/"
                  f"{num_bchs_tst}")
            if net.is_lstm:
                bch_tst, seq_len, _ = ins.size()
            else:
                bch_tst, _ = ins.size()
                seq_len = 1
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=bch_tst, dev=dev)
            # Run inference. The first element of the output is the
            # number of correct predictions.
            num_correct += inference(
                ins, labs, net, hidden,
                los_fnc=lambda a, b: (
                    # argmax(): The class is the index of the output
                    #     entry with greatest value (i.e., highest
                    #     probability). dim=1 because the output has an
                    #     entry for every entry in the input sequence.
                    # eq(): Compare the outputs to the labels.
                    # type(): Cast the resulting bools to ints.
                    # sum(): Sum them up to get the total number of correct
                    #     predictions.
                    torch.argmax(
                        a, dim=1).eq(b).type(torch.IntTensor).sum().item()))[0]
            total += bch_tst * seq_len
    # Convert the model back to training mode.
    net.train()
    acc_tst = num_correct / total
    print(f"Test accuracy: {acc_tst * 100:.2f}%")
    return acc_tst


def run(args, dat_in, dat_out, out_flp):
    """
    Trains a model according to the supplied parameters. Returns the test error
    (lower is better).
    """
    # Instantiate and configure the network. Move it to the proper device.
    net = models.MODELS[args["model"]](disp=True)
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net = torch.nn.DataParallel(net)
    dev = torch.device("cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.to(dev)

    # Split the data into training, validation, and test loaders.
    ldr_trn, ldr_val, ldr_tst = split_data(
        dat_in, dat_out, args["train_batch"], args["test_batch"], dev)

    # Training.
    tim_srt_s = time.time()
    net = train(
        net, args["epochs"], ldr_trn, ldr_val, dev, args["early_stop"],
        args["val_patience"], out_flp, args["learning_rate"], args["momentum"],
        args["val_improvement_thresh"], args["timeout_s"])
    print(f"Finished training - time: {time.time() - tim_srt_s:.2f} seconds")

    # # Read the best version of the model from disk.
    # net = torch.jit.load(out_flp)
    # net.to(dev)

    # Testing.
    tim_srt_s = time.time()
    los_tst = test(net, ldr_tst, dev)
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")
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
    print(f"Arguments: {args}")

    if args["no_rand"]:
        random.seed(SEED)
        torch.manual_seed(SEED)

    out_dir = args["out_dir"]
    if not path.isdir(out_dir):
        os.makedirs(out_dir)
    out_flp = path.join(args["out_dir"], "net.pth")
    # If a trained model file already exists, then delete it.
    if path.exists(out_flp):
        os.remove(out_flp)

    # Load or geenrate training data.
    dat_flp = path.join(out_dir, "data.npz")
    scl_prms_flp = path.join(out_dir, "scale_params.json")
    net_tmp = models.MODELS[args["model"]]()
    # Check for the presence of both the data and the scaling
    # parameters because the resulting model is useless without the
    # proper scaling parameters.
    if path.exists(dat_flp) and path.exists(scl_prms_flp):
        print("Found existing data!")
        dat = np.load(dat_flp)
        assert "in" in dat.files and "out" in dat.files, \
            f"Improperly formed data file: {dat_flp}"
        dat_in = dat["in"]
        dat_out = dat["out"]
    else:
        print("Regenerating data...")
        dat_in, dat_out, scl_prms = make_datasets(
            net_tmp, args["data_dir"],
            args["warmup"], args["num_sims"], SHUFFLE)
        # Save the processed data so that we do not need to process it again.
        print(f"Saving data: {dat_flp}")
        np.savez_compressed(dat_flp, **{"in": dat_in, "out": dat_out})
        # Save scaling parameters.
        print(f"Saving scaling parameters: {scl_prms_flp}")
        with open(scl_prms_flp, "w") as fil:
            json.dump(scl_prms.tolist(), fil)

    # Visualaize the ground truth data.
    clss = ([-1, 1] if isinstance(net_tmp, models.SVM) 
        else list(range(net_tmp.num_clss)))

    # Assumes that dat_out is a structured numpy array containing a
    # column named "class".
    tots = [(dat_out["class"] == cls).sum() for cls in clss]
    tot = sum(tots)
    print("Ground truth:\n" + "\n".join(
        [f"    {cls}: {tot_cls} packets ({tot_cls / tot * 100:.2f}%)"
         for cls, tot_cls in zip(clss, tots)]))
    tot_actual = np.prod(np.array(dat_out.shape))
    assert tot == tot_actual, \
        f"Error visualizing ground truth! {tot} != {tot_actual}"

    # TODO: Parallelize attempts.
    trls = args["conf_trials"]
    apts = 0
    apts_max = args["max_attempts"]
    ress = []
    while trls > 0 and apts < apts_max:
        apts += 1
        res = run(args, dat_in, dat_out, out_flp)
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
        "--warmup", default=0,
        help=("The number of packets to drop from the beginning of each "
              "simulation."),
        type=int)
    psr.add_argument(
        "--num-sims", default=sys.maxsize,
        help="The number of simulations to consider.", type=int)
    model_opts = sorted(models.MODELS.keys())
    psr.add_argument(
        "--model", default=model_opts[0], help="The model to use.",
        choices=model_opts, type=str)
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

    run_many(vars(psr.parse_args()))


if __name__ == "__main__":
    main()
