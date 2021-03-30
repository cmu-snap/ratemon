#!/usr/bin/env python3
"""
Based on:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
"""

import argparse
import copy
import functools
import math
import multiprocessing
import os
from os import path
import pickle
import sys
import time

import torch

import cl_args
import data
import defaults
import models
import utils


# The threshold of the new throughout to the old throughput above which a
# a training example will not be considered. I.e., the throughput must have
# decreased to less than this fraction of the original throughput for a
# training example to be considered.
NEW_TPT_TSH = 0.925
# Set to false to parse the experiments in sorted order.
SHUFFLE = True
# The number of times to log progress during one epoch.
LOGS_PER_EPC = 5
# The number of validation passes per epoch.
VALS_PER_EPC = 15


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


def inference_torch(ins, labs, net_raw, dev,
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


def train_torch(net, num_epochs, ldr_trn, ldr_val, dev, ely_stp, val_pat_max,
                out_flp, val_imp_thresh, tim_out_s, opt_params):
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
                      f"{'?' if ely_stp else num_epochs}, batch: "
                      f"{bch_idx_trn + 1:{f'0{len(str(num_bchs_trn))}'}}/"
                      f"{num_bchs_trn}", end=" ")
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
            # Zero out the parameter gradients.
            opt.zero_grad()
            loss, hidden = inference_torch(
                ins, labs, net.net, dev, hidden, los_fnc)
            # The backward pass.
            loss.backward()
            opt.step()
            if bch_idx_trn % bchs_per_log == 0:
                print(f"\tTraining loss: {loss:.5f}")

            # Run on validation set, print statistics, and (maybe) checkpoint
            # every VAL_PER batches.
            if ely_stp and not bch_idx_trn % bchs_per_val:
                print("\tValidation pass:")
                # For efficiency, convert the model to evaluation mode.
                net.net.eval()
                with torch.no_grad():
                    los_val = 0
                    for bch_idx_val, (ins_val, labs_val) in enumerate(ldr_val):
                        print(
                            "\tValidation batch: "
                            f"{bch_idx_val + 1}/{len(ldr_val)}")
                        # Initialize the hidden state for every new sequence.
                        hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
                        los_val += inference_torch(
                            ins_val, labs_val, net.net, dev, hidden,
                            los_fnc)[0].item()
                # Convert the model back to training mode.
                net.net.train()

                if los_val_min is None:
                    los_val_min = los_val
                # Calculate the percent improvement in the validation loss.
                prc = (los_val_min - los_val) / los_val_min * 100
                print(f"\tValidation error improvement: {prc:.2f}%")

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


def test_torch(net, ldr_tst, dev):
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
            num_correct += inference_torch(
                ins, labs, net.net, dev, hidden, los_fnc=net.check_output)[0]
            total += bch_tst * seq_len
    # Convert the model back to training mode.
    net.net.train()
    acc_tst = num_correct / total
    print(f"Test accuracy: {acc_tst * 100:.2f}%")
    return acc_tst


def run_sklearn(args, out_dir, out_flp, ldrs):
    """
    Trains an sklearn model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Unpack the dataloaders.
    ldr_trn, _, ldr_tst = ldrs
    # Construct the model.
    print("Building model...")
    net = models.MODELS[args["model"]]()
    net.new(**{param: args[param] for param in net.params})

    # Extract the training data from the training dataloader.
    dat_in, dat_out = list(ldr_trn)[0]
    if args["balance"]:
        print("Balanced training data:")
        utils.visualize_classes(net, dat_out)

    # Training.
    print("Training...")
    tim_srt_s = time.time()
    net.train(ldr_trn.dataset.fets, dat_in, dat_out)
    tim_trn_s = time.time() - tim_srt_s
    print(f"Finished training - time: {tim_trn_s:.2f} seconds")
    # Save the model.
    print(f"Saving final model: {out_flp}")
    with open(out_flp, "wb") as fil:
        pickle.dump(net.net, fil)
    # Testing.
    print("Testing...")
    tim_srt_s = time.time()
    acc_tst = net.test(
        *ldr_tst.dataset.raw(),
        graph_prms={
            "out_dir": out_dir, "sort_by_unfairness": True, "dur_s": None})
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")

    # Optionally perform feature elimination.
    if args["analyze_features"]:
        utils.select_fets(
            utils.analyze_feature_correlation(
                net, out_dir, dat_in, args["cluster_threshold"]),
            utils.analyze_feature_importance(
                net, out_dir, dat_in, dat_out, args["fets_to_pick"],
                args["perm_imp_repeats"]))

    return acc_tst, tim_trn_s


def run_torch(args, out_dir, out_flp, ldrs):
    """
    Trains a PyTorch model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Unpack the dataloaders.
    ldr_trn, ldr_val, ldr_tst = ldrs

    # Instantiate and configure the network. Move it to the proper device.
    net = models.MODELS[args["model"]]()
    net.new()
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net.net = torch.nn.DataParallel(net.net)
    dev = torch.device("cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.net.to(dev)

    # Explicitly move the training (and maybe validation) data to the target
    # device.
    ldr_trn.dataset.to(dev)
    ely_stp = args["early_stop"]
    if ely_stp:
        ldr_val.dataset.to(dev)

    # Training.
    tim_srt_s = time.time()
    net = train_torch(
        net, args["epochs"], ldr_trn, ldr_val, dev, args["early_stop"],
        args["val_patience"], out_flp, args["val_improvement_thresh"],
        args["timeout_s"],
        opt_params={param: args[param] for param in net.params})
    tim_trn_s = time.time() - tim_srt_s
    print(f"Finished training - time: {tim_trn_s:.2f} seconds")

    # Explicitly delete the training and validation data to save
    # memory on the target device.
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
    acc_tst = test_torch(net, ldr_tst, dev)
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")

    # Explicitly delete the test data to save memory on the target device.
    del ldr_tst
    # This is necessary for the GPU memory to be released.
    torch.cuda.empty_cache()

    return acc_tst, tim_trn_s


def prepare_args(args_):
    """ Updates the default arguments with the specified values. """
    # Initially, accept all default values. Then, override the defaults with
    # any manually-specified values. This allows the caller to specify values
    # only for parameters that they care about while ensuring that all
    # parameters have values.
    args = copy.copy(defaults.DEFAULTS)
    args.update(args_)
    return args


def run_trials(args):
    """
    Runs args["conf_trials"] trials and survives args["max_attempts"] failed
    attempts.
    """
    print(f"Arguments: {args}")

    if args["no_rand"]:
        utils.set_rand_seed()
    # Prepare the output directory.
    out_dir = args["out_dir"]
    if not path.isdir(out_dir):
        print(f"Output directory does not exist. Creating it: {out_dir}")
        os.makedirs(out_dir)
    # Create a temporary model to use during the data preparation
    # process. Another model will be created for the actual training.
    net_tmp = models.MODELS[args["model"]]()
    # Verify that the necessary supplemental parameters are present.
    for param in net_tmp.params:
        assert param in args, f"\"{param}\" not in args: {args}"
    # Assemble the output filepath.
    out_flp = path.join(
        args["out_dir"],
        (utils.args_to_str(args, order=sorted(defaults.DEFAULTS.keys()))
         ) + (
            # Determine the proper extension based on the type of
            # model.
            ".pickle" if isinstance(net_tmp, models.SvmSklearnWrapper)
            else ".pth"))
    # If a trained model file already exists, then delete it.
    if path.exists(out_flp):
        os.remove(out_flp)
    # If custom features are specified, then overwrite the model's
    # default features.
    fets = args["features"]
    if fets:
        net_tmp.in_spc = tuple(fets)
    else:
        args["features"] = net_tmp.in_spc

    # Load the training, validation, and test data.
    ldrs = data.get_dataloaders(args, net_tmp)
    utils.visualize_classes(net_tmp, ldrs)

    # TODO: Parallelize attempts.
    trls = args["conf_trials"]
    apts = 0
    apts_max = args["max_attempts"]
    ress = []
    while trls > 0 and apts < apts_max:
        apts += 1
        res = (
            run_sklearn
            if isinstance(net_tmp, models.SvmSklearnWrapper)
            else run_torch)(args, out_dir, out_flp, ldrs)
        if res[0] == 100:
            print(
                (f"Training failed (attempt {apts}/{apts_max}). Trying again!"))
        else:
            ress.append(res)
            trls -= 1
    if ress:
        print(("Resulting accuracies: "
               f"{', '.join([f'{acc:.2f}' for acc, _ in ress])}"))
        max_acc, tim_s = max(ress, key=lambda p: p[0])
        print(f"Maximum accuracy: {max_acc:.2f}")
        # Return the minimum error instead of the maximum accuracy.
        return 1 - max_acc, tim_s
    print(f"Model cannot be trained with args: {args}")
    return float("NaN"), float("NaN")


def run_cnf(cnf, gate_func=None, post_func=None):
    """
    Executes a single configuration. Assumes that the arguments have already
    been processed with prepare_args().
    """
    func = run_trials
    # Optionally decide whether to run a configuration.
    if gate_func is not None:
        func = functools.partial(gate_func, func=func)
    res = func(cnf)
    # Optionally process the output of each configuration.
    if post_func is not None:
        res = post_func(cnf, res)
    return res


def run_cnfs(cnfs, sync=False, gate_func=None, post_func=None):
    """
    Executes many configurations. Assumes that the arguments have already been
    processed with prepare_args().
    """
    num_cnfs = len(cnfs)
    print(f"Training {num_cnfs} configurations.")
    # The configurations themselves should execute synchronously if
    # and only if sync is False or the configuration is explicity
    # configured to run synchronously.
    cnfs = zip(
        [{**cnf,
          "sync": (not sync) or cnf.get("sync", defaults.DEFAULTS["sync"])}
         for cnf in cnfs],
        [gate_func, ] * num_cnfs, [post_func, ] * num_cnfs)

    if defaults.SYNC:
        res = [run_cnf(*cnf) for cnf in cnfs]
    else:
        with multiprocessing.Pool(processes=3) as pol:
            res = pol.starmap(run_cnf, cnfs)
    return res


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Train a model on the output of gen_features.py.")
    psr.add_argument(
        "--graph", action="store_true",
        help=("If the model is an sklearn model, then analyze and graph the "
              "testing results."))
    psr.add_argument(
        "--balance", action="store_true",
        help="Balance the training data classes")
    psr.add_argument(
        "--drop-popular", action="store_true",
        help=("Drop samples from popular classes instead of adding samples to "
              "unpopular classes. Must be used with \"--balance\"."))
    psr.add_argument(
        "--analyze-features", action="store_true",
        help="Analyze feature importance.")
    psr.add_argument(
        "--l2-regularization", default=defaults.DEFAULTS["l2_regularization"],
        required=False, type=float,
        help=("If the model is of type \"{models.HistGbdtSklearnWrapper().name}\", "
              "then use this as the L2 regularization parameter."))
    psr.add_argument(
        "--clusters", default=defaults.DEFAULTS["clusters"],
        required=False, type=int,
        help=("If \"--analyze-features\" is specified, then pick this many "
              "clusters."))
    # TODO: Make comments specific to HistGbdt
    psr.add_argument(
        "--features-to-pick", default=defaults.DEFAULTS["fets_to_pick"],
        required=False, type=int, dest="fets_to_pick",
        help=("If \"--analyze-features\" is specified, then pick this many of "
              "the top features."))
    psr.add_argument(
        "--permutation-importance-repeats", dest="perm_imp_repeats",
        default=defaults.DEFAULTS["perm_imp_repeats"], required=False, type=int,
        help=("If \"--analyze-features\" is specificed, then perform "
              "permutation importance analysis with this many repeats."))
    psr, psr_verify = cl_args.add_training(psr)
    args = vars(psr_verify(psr.parse_args()))
    assert (not args["drop_popular"]) or args["balance"], \
        "\"--drop-popular\" must be used with \"--balance\"."
    assert ((not args["analyze_features"]) or
            (args["balance"] and args["drop_popular"])), \
        ("Refusing to use \"--analyze-features\" with \"--balance\" but without "
         "\"--drop-popular\".")
    assert args["clusters"] >= 1, \
        f"\"--clusters\" must be at least 1, but is: {args['clusters']}"
    # Verify that all arguments are reflected in defaults.DEFAULTS.
    for arg in args.keys():
        assert arg in defaults.DEFAULTS, \
            f"Argument {arg} missing from defaults.DEFAULTS!"
    run_trials(prepare_args(args))


if __name__ == "__main__":
    main()
