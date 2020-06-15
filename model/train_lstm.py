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
# The number of output classes.
NUM_CLASSES = 5


class Dataset(torch.utils.data.Dataset):
    """ A simple Dataset that wraps an array of (input, output) value pairs. """

    def __init__(self, dat):
        super(Dataset).__init__()
        self.dat = dat

    def __len__(self):
        """ Returns the number of items in this Dataset. """
        return len(self.dat)

    def __getitem__(self, idx):
        """ Returns a specific item from this Dataset. """
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat[idx]


def scale_fets(dat_all):
    """
    dat_all is a list of numpy arrays, each corresponding to a simulation.
    Returns a copy of dat_all with the columns scaled between 0 and 1. Also
    returns an array of shape (dat_all[0].shape[1], 2) where row i contains the
    min and the max of column i in the entries in dat_all.
    """
    assert dat_all, "No data!"
    # Pick the first simulation result and determine the column specification.
    fets = dat_all[0].dtype.names
    # Create an empty array to hold the min and max values for each feature.
    scl_prms = np.empty((len(fets), 2))
    # First, look at all features across simulations to determine the
    # global scale paramters for each feature. For each feature column...
    for j, fet in enumerate(fets):
        # Find the global min and max values for this feature.
        min_global = float("inf")
        max_global = float("-inf")
        # For each simulation...
        for dat in dat_all:
            fet_values = dat[fet]
            min_global = min(min_global, fet_values.min())
            max_global = max(max_global, fet_values.max())
        scl_prms[j] = np.array([min_global, max_global])

    def normalize(dat):
        """ Rescale all of the features in dat to the range [0, 1]. """
        nrm = np.empty(dat.shape, dtype=dat.dtype)
        for j, fet in enumerate(fets):
            min_in, max_in = scl_prms[j]
            fet_values = dat[fet]
            if min_in == max_in:
                scaled = np.zeros(fet_values.shape, dtype=fet_values.dtype)
            else:
                scaled = utils.scale(
                    fet_values, min_in, max_in, min_out=0, max_out=1)
            nrm[fet] = scaled
        return nrm

    return [normalize(dat) for dat in dat_all], scl_prms


def make_datasets(net, dat_dir, bch_trn, bch_tst, warmup, num_sims, shuffle):
    """
    Parses the files in data_dir, transforms them (e.g., by scaling)
    into the correct format for the network, and returns training,
    validation, and test data loaders.

    If num_sims is not None, then selects only the first num_sims
    simulations. If shuffle is True, then the simulations will be parsed in
    sorted order. Use num_sims and shuffle=True together to simplify debugging.

    """
    # Load all results.
    print("Loading data...")
    sims = sorted(os.listdir(dat_dir))
    if shuffle:
        random.shuffle(sims)
    if num_sims is not None:
        sims = sims[:num_sims]

    print(f"    Found {len(sims)} simulations.")
    with multiprocessing.Pool() as pol:
        # Each element of dat_all corresponds to a single simulation.
        dat_all = pol.map(
            utils.load_sim, [path.join(dat_dir, sim) for sim in sims])
    print("Done.")

    print("Formatting data...")
    # Drop the first few packets so that we consider steady-state behavior only.
    assert np.array([dat.shape[0] > warmup for _, _, dat in dat_all]).all(), \
        "Unable to drop first {warmup} packets!"
    dat_all = [(sim, num_flws, dat[warmup:]) for sim, num_flws, dat in dat_all]
    # Split each data matrix into two separate matrices: one with the input
    # features only and one with the output features only. The names of the
    # columns correspond to the feature names in in_spc and out_spc.
    assert net.in_spc, "Empty in spec."
    assert net.out_spc, "Empty out spec."
    dat_all = [(sim, num_flws, dat["arrival time"], dat[net.in_spc], dat[net.out_spc])
               for sim, num_flws, dat in dat_all]
    # Unzip dat from a list of pairs of in and out features into a pair of lists
    # of in and out features.
    sims_all, num_flws_all, arr_times_all, dat_in_all, dat_out_all = (
        zip(*dat_all))
    # Scale input features.
    dat_in_all, prms_in = scale_fets(dat_in_all)
    # Convert output features to class labels. Must call list()
    # because the value gets used more than once.
    dat_out_all = net.convert_to_class(list(zip(dat_out_all, num_flws_all)))
    print("Done.")

    print("Verifying and visualizing data...")
    # Verify data.
    for (dat_in, dat_out) in zip(dat_in_all, dat_out_all):
        assert dat_in.shape[0] == dat_out.shape[0], \
            "Should have the same number of rows."
    # Find the uniques classes in the output features and make sure that they
    # are properly formed.
    for cls in {x for y in [set(dat_out.tolist()) for dat_out in dat_out_all]
                for x in y}:
        assert 0 <= cls <= net.num_clss, "Invalid class: {cls}"

    # Visualaize the ground truth data.
    def count(x):
        """ Returns the number of entries that have a particular value. """
        return sum([(dat_out == x).sum()
                    for dat_out in dat_out_all])
    clss = list(range(net.num_clss))
    tots = [count(cls) for cls in clss]
    tot = sum(tots)
    print("\n    Ground truth:")
    for cls, tot_cls in zip(clss, tots):
        print(f"        {cls}: {tot_cls} packets ({tot_cls / tot * 100:.2f}%)")
    print()
    assert (tot == sum([dat_out.shape[0] for dat_out in dat_out_all])), \
        "Error visualizing ground truth!"
    print("Done.")

    print("Creating train/val/test data...")
    # Convert each training input/output pair to Torch tensors.
    dat_all = [(torch.tensor(dat_in.tolist(), dtype=torch.float),
                torch.tensor(dat_out, dtype=torch.long))
               for dat_in, dat_out in zip(dat_in_all, dat_out_all)]
    # Transform the data as required by this specific model.
    dat_all = net.modify_data(list(zip(
        sims_all,
        [torch.tensor(arr_times.tolist()) for arr_times in arr_times_all],
        dat_all)))

    # Shuffle the data to ensure that the training, validation, and test sets
    # are uniformly sampled.
    random.shuffle(dat_all)
    # 50% for training, 20% for validation, 30% for testing.
    tot = len(dat_all)
    num_val = int(round(tot * 0.2))
    num_tst = int(round(tot * 0.3))
    print((f"    Data - train: {tot - num_val - num_tst}, val: {num_val}, "
           f"test: {num_tst}"))
    dat_val = dat_all[:num_val]
    dat_tst = dat_all[num_val:num_val + num_tst]
    dat_trn = dat_all[num_val + num_tst:]
    # Create the dataloaders.
    ldr_trn = torch.utils.data.DataLoader(
        Dataset(dat_trn), batch_size=bch_trn, shuffle=True)
    ldr_val = torch.utils.data.DataLoader(
        Dataset(dat_val), batch_size=bch_tst, shuffle=False)
    ldr_tst = torch.utils.data.DataLoader(
        Dataset(dat_tst), batch_size=bch_tst, shuffle=False)
    print("Done.")
    return ldr_trn, ldr_val, ldr_tst, prms_in


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


def inference(ins, labs, net, dev, hidden=None, los_fnc=None):
    """
    Runs a single inference pass. Returns the output of net, or the
    loss if los_fnc is not None.
    """
    # Move the training data to the specified device.
    ins = ins.to(dev)
    labs = labs.to(dev)
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

    num_bchs_trn = len(ldr_trn)
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
                print(f"Epoch: {epoch_idx + 1}/{'?' if ely_stp else num_epochs}, "
                      f"batch: {bch_idx_trn + 1}/{num_bchs_trn}", end=" ")
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
            # Zero out the parameter gradients.
            opt.zero_grad()
            loss, hidden = inference(ins, labs, net, dev, hidden, los_fnc)
            # The backward pass.
            loss.backward()
            opt.step()
            if bch_idx_trn % bchs_per_log == 0:
                print(f"\tTraining loss: {loss:.5f}")

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
                            ins_val, labs_val, net, dev, hidden, los_fnc)[0].item()
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
    # For efficiency, convert the model to evaluation mode.
    net.eval()
    with torch.no_grad():
        for bch_idx, (ins, labs) in enumerate(ldr_tst):
            print(f"Test batch: {bch_idx + 1}/{len(ldr_tst)}")
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
                ins, labs, net, dev, hidden,
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


def run(args_):
    """
    Trains a model according to the supplied parameters. Returns the test error
    (lower is better).
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

    # Instantiate and configure the network.
    net = models.MODELS[args["model"]]()
    base_net = net
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net = torch.nn.DataParallel(net)
        base_net = net.module

    dev = torch.device(
        "cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.to(dev)

    bch_trn = args["train_batch"]
    ldr_trn, ldr_val, ldr_tst, scl_prms = make_datasets(
        base_net, args["data_dir"], bch_trn, args["test_batch"], args["warmup"],
        args["num_sims"], SHUFFLE)

    # Save scaling parameters.
    with open(path.join(out_dir, "scale_params.json"), "w") as fil:
        json.dump(scl_prms.tolist(), fil)

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


def run_many(args):
    """
    Run args["conf_trials"] trials and survive args["max_attempts"] failed
    attempts.
    """
    # TODO: Parallelize attempts.
    trls = args["conf_trials"]
    apts = 0
    apts_max = args["max_attempts"]
    ress = []
    while trls > 0 and apts < apts_max:
        apts += 1
        res = run(args)
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
              "before training is automatically aborted."), type=int)
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
