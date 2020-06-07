#!/usr/bin/env python3
"""
Based on the PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import argparse
import json
import math
import os
from os import path
import random
import time

from matplotlib import pyplot
import numpy
import torch

import models


# Parameter defaults.
DEFAULTS = {
    "epochs": 100,
    "num_gpus": 0,
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
# Whether to generate training data graphs.
PLT = False
# The random seed.
SEED = 1337
# The number of validation passes per epoch.
VALS_PER_EPC = 15


def scale(x, min_in, max_in, min_out, max_out):
    return min_out + (x - min_in) * (max_out - min_out) / (max_in - min_in)


def scale_fets(dat, fet_nam, inv=False):
    # Select fields we care about
    fets = numpy.array([row[fet_nam] for row in dat])
    if inv:
        fets = 1. / fets
    min_in = fets.min()
    max_in = fets.max()
    return (scale(fets, min_in, max_in, min_out=0, max_out=1).tolist(),
            (min_in, max_in))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dat):
        super(Dataset).__init__()
        self.dat = dat

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat[idx]


def make_datasets(dat_flp, net, bch_trn, bch_tst):
    # Load the training, validation, and testing data.
    with open(dat_flp, "r") as fil:
        dat = json.load(fil)

    len_bfr = len(dat)
    dat = [row for row in dat
           # Drop data with no ACK pacing and where the decrease in throughput
           # is less than 7.5%.
           if (row["ack_period_us"] != 0 and
               (row["average_throughput_after_bps"] /
                row["average_throughput_before_bps"]) < NEW_TPT_TSH)]
    # Sort for graphing purposes.
    dat = sorted(dat, key=lambda x: x["average_throughput_after_bps"])
    print(f"Dropped {len_bfr - len(dat)} examples whose ground truth is 0.")

    # for row in dat:
    #     print(row)

    # Select the specified features for this model.
    fets_in, prms_in = zip(
        *(scale_fets(dat, fet_in) for fet_in in net.in_spc))
    # Hack: If the feature name is "ack_period_us", then take the reciprocal of
    #       the feature value.
    # TODO: Encode feature transformations like this in the model.
    fets_out, prms_out = zip(*(
        scale_fets(dat, fet_out, inv=fet_out == "ack_period_us")
        for fet_out in net.out_spc))

    if PLT:
        fontsize = 16
        pyplot.plot(fets_in[0], fets_out[0], marker="o")
        pyplot.xlabel("Desired throughput (b/s) - scaled", fontsize=fontsize)
        pyplot.ylabel("ACK frequency (/us) - scaled", fontsize=fontsize)
        pyplot.xticks(fontsize=fontsize)
        pyplot.yticks(fontsize=fontsize)
        pyplot.tight_layout()
        pyplot.savefig("training_data_scaled.pdf")
        pyplot.close()

    len_bfr = len(dat)
    # Convert from:
    #     [ [all values for input feature 0], ... ]
    # and
    #     [ [all values for output feature 0], ... ]
    # to
    #     [ ([ example 0 input feature 0, ...],
    #        [ example 0 output feature 0, ...]) ]
    # Basically, create per-example feature tuples.
    # dat = []
    # for fet_in, fet_out in zip(zip(*fets_in), zip(*fets_out)):

    #     dat.append(
    #         (torch.tensor(fet_in, dtype=torch.float),
    #          torch.tensor(fet_out, dtype=torch.float))
    #     )
    # for row in dat:
    #     print(row)

    # return
    dat = [(torch.tensor(fet_in, dtype=torch.float),
            torch.tensor(fet_out, dtype=torch.float))
           for fet_in, fet_out in zip(zip(*fets_in), zip(*fets_out))
           # Drop examples where any output features was rescaled to 0 because
           # these examples break the calculation of percent error.
           if (numpy.array(fet_out) != numpy.zeros(len(fet_out))).all()]
    print((f"Dropped {len_bfr - len(dat)} examples whose ground truth scaled "
           "to 0."))

    # Shuffle the data to ensure that the training, validation, and test sets
    # are uniformly sampled.
    random.shuffle(dat)
    # 50% for training, 20% for validation, 30% for testing.
    tot = len(dat)
    num_val = int(round(tot * 0.2))
    num_tst = int(round(tot * 0.3))
    print((f"Data - train: {tot - num_val - num_tst}, val: {num_val}, test: "
           f"{num_tst}"))
    dat_val = dat[:num_val]
    dat_tst = dat[num_val:num_val + num_tst]
    dat_trn = dat[num_val + num_tst:]
    # Create the dataloaders.
    ldr_trn = torch.utils.data.DataLoader(
        Dataset(dat_trn), batch_size=bch_trn, shuffle=True)
    ldr_val = torch.utils.data.DataLoader(
        Dataset(dat_val), batch_size=bch_tst, shuffle=False)
    ldr_tst = torch.utils.data.DataLoader(
        Dataset(dat_tst), batch_size=bch_tst, shuffle=False)

    return ldr_trn, ldr_val, ldr_tst, (prms_in, prms_out)


def train(net, num_epochs, ldr_trn, ldr_val, dev, ely_stp, val_pat_max, out_flp,
          lr, momentum, val_imp_thresh, tim_out_s):
    los_fnc = torch.nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # If using early stopping, then this is the lowest validation loss
    # encountered so far.
    los_val_min = None
    # If using early stopping, then this tracks the *remaining* validation
    # patience (initially set to the maximum amount of patience). This is
    # decremented for every validation pass that does not improve the
    # validation loss by at least val_imp_thresh percent. When this reaches
    # zero, training aborts.
    val_pat = val_pat_max

    # To avoid a divide-by-zero error below, the number of validation passes
    # per epoch much be at least 2.
    assert VALS_PER_EPC >= 2
    if len(ldr_trn) < VALS_PER_EPC:
        # If the number of batches per epoch is less than the desired number of
        # validation passes per epoch, when do a validation pass after every
        # batch.
        bchs_per_val = 1
    else:
        # Using floor() means that dividing by (VALS_PER_EPC - 1) will result in
        # VALS_PER_EPC validation passes per epoch.
        bchs_per_val = math.floor(len(ldr_trn) / (VALS_PER_EPC - 1))

    # net.apply(models.init_weights)

    tim_srt_s = time.time()
    # Loop over the dataset multiple times...
    for epoch_idx in range(num_epochs):
        tim_del_s = time.time() - tim_srt_s
        if tim_out_s != -1 and tim_del_s > tim_out_s:
            print((f"Training timed out after after {epoch_idx} epochs "
                   f"({tim_del_s:.2f} seconds)."))
            break
        # print(f"Epoch: {epoch_idx + 1}")
        # For each batch...
        for bch_idx, bch_trn in enumerate(ldr_trn, 0):
            # Get the training data and move it to the specified device.
            ins, labs = bch_trn
            ins = ins.to(dev)
            labs = labs.to(dev)
            # Zero out the parameter gradients.
            opt.zero_grad()
            # Forward pass + backward pass + optimize.
            los_fnc(net(ins), labs).backward()
            opt.step()

            # Run on validation set, print statistics, and (maybe) checkpoint
            # every VAL_PER batches.
            if ely_stp and not bch_idx % bchs_per_val:
                # For efficiency, convert the model to evaluation mode.
                net.eval()
                with torch.no_grad():
                    los_val = 0
                    for bch_val in ldr_val:
                        # Get the validation data and move it to the specified
                        # device.
                        ins, labs = bch_val
                        ins = ins.to(dev)
                        labs = labs.to(dev)
                        los_val += los_fnc(net(ins), labs).item()
                # Convert the model back to training mode.
                net.train()

                if los_val_min is None:
                    los_val_min = los_val
                # Calculate the percent improvement in the validation loss.
                prc = (los_val_min - los_val) / los_val_min * 100
                print(f"[epoch {epoch_idx:5d}, batch {bch_idx:5d}] Validation "
                      f"error improvement: {prc:.2f}%")

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
                    torch.jit.save(torch.jit.script(net), out_flp)
                else:
                    val_pat -= 1
                    if path.exists(out_flp):
                        # Resume training from the best model.
                        net = torch.jit.load(out_flp)
                        net.to(dev)
                if val_pat <= 0:
                    print(f"Stopped after {epoch_idx + 1} epochs")
                    return net
    if not ely_stp:
        # Save the final version of the model. Convert the model to Torch Script
        # first.
        print(f"Saving: {out_flp}")
        torch.jit.save(torch.jit.script(net), out_flp)
    return net


def test(net, ldr_tst, dev):
    # The test loss, accumulated across all batches.
    los_tst_men = 0
    los_tst_med = 0
    # For efficiency, convert the model to evaluation mode.
    net.eval()
    with torch.no_grad():
        for bch_tst in ldr_tst:
            # Get the testing data and move it to the specified device.
            ins, labs = bch_tst
            ins = ins.to(dev)
            labs = labs.to(dev)
            # Run inference. Take the L1 distance between the labs and outs,
            # convert it to a percent of the label value, and take the median
            # across all samples in the batch.
            outs = net(ins)
            prc_errs = torch.abs(labs - outs) / labs * 100
            los_men = torch.mean(prc_errs).item()
            los_med = torch.median(prc_errs).item()
            # print("-----")
            # print(f"ins: {ins}")
            # print(f"outs: {outs}")
            # print(f"labs: {labs}")
            # print(f"abs(labs - outs): {torch.abs(labs - outs)}")
            # print(f"abs(labs - outs) / labs: {torch.abs(labs - outs) / labs}")
            # print(f"abs(labs - outs) / labs * 100: {torch.abs(labs - outs) / labs * 100}")
            # print(f"los_men: {los_men}")
            # print(f"los_med: {los_med}")
            # return
            los_tst_men += los_men
            los_tst_med += los_med
    # Convert the model back to training mode.
    net.train()
    # Average across all batches.
    num_bchs = len(ldr_tst)
    los_tst_men /= num_bchs
    los_tst_med /= num_bchs
    print(f"Average L1 test error: {los_tst_men:.2f}%")
    print(f"Average median L1 test error: {los_tst_med:.2f}%")
    return los_tst_med


def run(args_):
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
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net = torch.nn.DataParallel(net)
    dev = torch.device(
        "cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.to(dev)

    ldr_trn, ldr_val, ldr_tst, scl_prms = \
        make_datasets(
            args["data"], net, args["train_batch"], args["test_batch"])
    # Save scaling parameters.
    with open(path.join(out_dir, "scale_params.json"), "w") as fil:
        json.dump(scl_prms, fil)
        ##input min,input max,output min,output max
        #fil.write(f"{','.join([str(prm) for prm in scl_prms])}\n")

    # Training.
    tim_srt_s = time.time()
    net = train(
        net, args["epochs"], ldr_trn, ldr_val, dev, args["early_stop"],
        args["val_patience"], out_flp, args["learning_rate"], args["momentum"],
        args["val_improvement_thresh"], args["timeout_s"])
    print(f"Finished training - time: {time.time() - tim_srt_s:.2f} seconds")

    # Read the best version of the model from disk.
    net = torch.jit.load(out_flp)
    net.to(dev)

    # Testing.
    tim_srt_s = time.time()
    los_tst = test(net, ldr_tst, dev)
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")

    # Print trained parameters.
    print("Trained parameters:")
    for nam, prm in net.named_parameters():
        if prm.requires_grad:
            print(f"\t{nam}: {prm}")
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
        min_err = min(ress)
        print(("Resulting errors: "
               f"{', '.join([f'{res:.2f}%' for res in ress])}"))
        print(f"Minimum error: {min_err:.2f}%")
        return min_err
    print(f"Model cannot be trained with args: {args}")
    return 1e9


def main():
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="A basic DNN training framework.")
    psr.add_argument(
        "--data",
        help="The path to the training/validation/testing data (required).",
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
