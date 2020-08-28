#!/usr/bin/env python3
"""
Reads parsed simulation files created by parse_dumbell.py and
creates unified training, validation, and test sets.
"""

import argparse
import math
import os
from os import path
import time
import random

import numpy as np

import train
import utils


def __get_num_packets_and_dtype(sim_flps):
    print("Surveying simulations...")
    assert sim_flps, "Must provide at least one simulation."

    # Extract the header from each simulation.
    headers = np.array(
        [utils.get_npz_headers(flp) for flp in sim_flps]).reshape((len(sim_flps), 3))
    # Count the total number of packets by looking at the header shapes.
    num_pkts = sum(shape[0] for shape in headers[:, 1])
    # Extract the dtype and verify that all dtypes are the same.
    dtype = headers[0][2]
    assert (headers[1:][:, 2] == dtype).all(), "Not all simulations agree on dtype!"
    return num_pkts, dtype


def __merge(sim_flps, out_dir, num_pkts, dtype, splits):
    # Create the final output files (version 1).
        # For each simulation file, load it.
        # Select each split randomly and append those entries to each split.
        # Shuffle the training split.
    print("Preparing target files...")

    trn = np.memmap(
        path.join(out_dir, "train.npy"), dtype, mode="w+",
        shape=(math.ceil(num_pkts * splits["trn"] / 100),))
    val = np.memmap(
        path.join(out_dir, "val.npy"), dtype, mode="w+",
        shape=(math.ceil(num_pkts * splits["val"] / 100),))
    tst = np.memmap(
        path.join(out_dir, "test.npy"), dtype, mode="w+",
        shape=(math.ceil(num_pkts * splits["tst"] / 100),))
    trn.fill(-1)
    val.fill(-1)
    tst.fill(-1)

    forgotten = 0
    idxs = {"trn": 0, "val": 0, "tst": 0}
    for sim_flp in sim_flps:
        dat = utils.load_sim(sim_flp)[1]

        # Start with the list of all indices, then remove the new
        # indices that will be used for the validation and test
        # data. The remaining indices will be used for the training
        # data. This guarantees that the amount of training data is at
        # least as large as specified.
        num_sim_pkts = dat.shape[0]
        all_idxs = set(range(num_sim_pkts))

        num_new_trn = math.floor(num_sim_pkts * splits["trn"] / 100)
        new_trn_idxs = random.sample(all_idxs, num_new_trn)
        all_idxs -= set(new_trn_idxs)

        num_new_val = math.floor(num_sim_pkts * splits["val"] / 100)
        new_val_idxs = random.sample(all_idxs, num_new_val)
        all_idxs -= set(new_val_idxs)

        num_new_tst = math.floor(num_sim_pkts * splits["tst"] / 100)
        new_tst_idxs = random.sample(all_idxs, num_new_tst)
        all_idxs -= set(new_tst_idxs)

        # Record how many packets are not being moved to one of the
        # merged files.
        forgotten += len(all_idxs)

        assert num_new_trn > 0 or splits["trn"] == 0
        assert num_new_val > 0 or splits["val"] == 0
        assert num_new_tst > 0 or splits["tst"] == 0

        trn_end_idx = idxs["trn"] + num_new_trn
        val_end_idx = idxs["val"] + num_new_val
        tst_end_idx = idxs["tst"] + num_new_tst

        assert trn_end_idx <= trn.shape[0], \
            f"Index {trn_end_idx} into training set does not fit within shape {trn.shape}"
        assert val_end_idx <= val.shape[0]
        assert tst_end_idx <= tst.shape[0]

        trn[idxs["trn"]:trn_end_idx] = dat[new_trn_idxs]
        trn[idxs["val"]:val_end_idx] = dat[new_val_idxs]
        trn[idxs["tst"]:tst_end_idx] = dat[new_tst_idxs]

        idxs["trn"] = trn_end_idx
        idxs["val"] = val_end_idx
        idxs["tst"] = tst_end_idx

    print(f"Forgot {forgotten}/{num_pkts} packets ({forgotten / num_pkts:.2f}%)")

    # Shuffle the training data. The validation and test data do not
    # need to be shuffled.
    np.random.shuffle(trn)

    del trn
    del val
    del tst

    # Create the final output files (version 2).
        # For each simulation file, load it and append it to the final output file.
        # Shuffle the final output file.
        # Divide the final output file into training, validation, and test sets.


def __main():
    psr = argparse.ArgumentParser(
        description=(
            "Merges parsed simulation files into unified training, validation, "
            "and test data."))
    psr.add_argument(
        "--data-dir",
        help="The path to a directory containing the simulation files.",
        required=True, type=str)
    psr.add_argument(
        "--out-dir", default=train.DEFAULTS["out_dir"],
        help="The directory in which to store the merged files.", type=str)
    psr.add_argument(
        "--train-split", default=50, help="Training data fraction",
        required=False, type=float)
    psr.add_argument(
        "--val-split", default=20, help="Validation data fraction",
        required=False, type=float)
    psr.add_argument(
        "--test-split", default=30, help="Test data fraction",
        required=False, type=float)
    psr.add_argument(
        "--num-sims", help="The number of simulations to consider.",
        required=False, type=int)
    args = psr.parse_args()

    splits = {
        "trn": args.train_split, "val": args.val_split, "tst": args.test_split}
    tot_split = sum(splits.values())
    assert tot_split == 100, \
        ("The sum of the training, validation, and test splits must equal 100, "
         f"not {tot_split}")

    out_dir = args.out_dir
    if not path.isdir(out_dir):
        print(f"Output directory does not exist. Creating it: {out_dir}")
        os.makedirs(out_dir)

    tim_srt_s = time.time()

    # Determine the simulation filepaths.
    sims_dir = args.data_dir
    sim_flns = os.listdir(sims_dir)
    num_sims = args.num_sims
    num_sims = len(sim_flns) if num_sims is None else num_sims
    sim_flps = [
        path.join(sims_dir, sim_fln) for sim_fln in sim_flns[:num_sims]]
    num_pkts, dtype = __get_num_packets_and_dtype(sim_flps)
    fets = dtype.names
    print(f"Total packets: {num_pkts}\nFeatures:\n    " + '\n    '.join(sorted(fets)))

    # Create the merged training, validation, and test files.
    __merge(sim_flps, out_dir, num_pkts, dtype, splits)
    print(f"Finished - time: {time.time() - tim_srt_s:.2f} seconds")
    return 0


if __name__ == "__main__":
    __main()
