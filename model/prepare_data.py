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


class Split:
    """ Represents either the training, validation, or test split. """

    def __init__(self, name, frac, flp, dtype, num_pkts_tot):
        self.name = name
        self.frac = frac / 100
        # Create an empty file for each split, and set all entries
        # to -1. This matches the behavior of parse_dumbbell.py:
        # Features values that cannot be computed are replaced
        # with -1. When reading the splits later, we can detect
        # incomplete feature values by looking for -1s.
        print(f"Creating file for split \"{self.name}\"...")
        self.dat = np.memmap(
            flp, dtype, mode="w+", shape=(math.ceil(num_pkts_tot * frac),))
        self.dat.fill(-1)
        # The next available index in self.dat.
        self.idx = 0

    def take(self, sim_dat, available_idxs):
        """
        Takes this Split's specified fraction of data from sim_dat,
        choosing from available_idxs. Removes the chosen indices from
        available_idxs and returns it.
        """
        num_sim_pkts = sim_dat.shape[0]
        num_new = math.floor(num_sim_pkts * self.frac)
        # Verify that if a split fraction was nonzero, then at
        # least one packet was selected. This is a common-case
        # heuristic rather than an invariant, since it is
        # reasonable for zero packets to be selected if the
        # simulation has either very few packets or the split
        # fraction is very low.
        assert num_new > 0 or self.frac == 0, \
            (f"Selecting 0 of {num_sim_pkts} packets, but fraction is: "
             f"{self.frac}")
        end_idx = self.idx + num_new
        assert end_idx <= self.dat.shape[0], \
            (f"Index {end_idx} into \"{self.name}\" split does not fit within "
             f"shape {self.dat.shape}")
        new_idxs = random.sample(available_idxs, num_new)
        self.dat[self.idx:end_idx] = sim_dat[new_idxs]
        self.idx = end_idx
        available_idxs -= set(new_idxs)
        return available_idxs


def __survey(sim_flps):
    """
    Surveys the provided simulations to determine their total packets and dtype,
    which are returned.
    """
    num_sims = len(sim_flps)
    print(f"Surveying {num_sims} simulations...")
    assert sim_flps, "Must provide at least one simulation."

    # Extract the header from each simulation. Reshape to remove a
    # middle dimension of size 1.
    headers = np.array(
        [utils.get_npz_headers(flp) for flp in sim_flps]).reshape((num_sims, 3))
    # Count the total number of packets by looking at the header shapes.
    num_pkts = sum(shape[0] for shape in headers[:, 1])
    # Extract the dtype and verify that all dtypes are the same.
    dtype = headers[0][2]
    assert (headers[1:][:, 2] == dtype).all(), "Not all simulations agree on dtype!"
    return num_pkts, dtype


def __merge(sim_flps, out_dir, num_pkts, dtype, split_prcs):
    """
    Merges the provided simulations into training, validation, and
    test splits as defined by the percents in split_prcs. Stores the
    resulting files in out_dir. The simulations contain a total of
    num_pkts packets and have the provided dtype.
    """
    # Create the final output files (version 1).
        # For each simulation file, load it.
        # Select each split randomly and append those entries to each split.
        # Shuffle the training split.
    print("Preparing split files...")
    splits = {
        name: Split(
            name, frac, path.join(out_dir, f"{name}.npy"), dtype, num_pkts)
        for name, frac in split_prcs.items()}
    # Keep track of the number of packets that do not get selected for
    # any of the splits.
    pkts_forgotten = 0
    num_sims = len(sim_flps)
    for idx, sim_flp in enumerate(sim_flps):
        # Load the simulation.
        dat = utils.load_sim(
            sim_flp, msg=f"{idx + 1:{f'0{len(str(num_sims))}'}}/{num_sims}")[1]
        if dat is None:
            continue
        # Start with the list of all indices. Each split select some
        # indices for itself, then removes them from this set.
        all_idxs = set(range(dat.shape[0]))
        # For each split, take a fraction of the simulation packets.
        for split in splits.values():
            all_idxs = split.take(dat, all_idxs)
        # Record how many packets are not being moved to one of the
        # merged files.
        pkts_forgotten += len(all_idxs)
    print(f"Forgot {pkts_forgotten}/{num_pkts} packets ({pkts_forgotten / num_pkts:.2f}%)")

    # Shuffle the training data. The validation and test data do not
    # need to be shuffled.
    print("Shuffling training data...")
    np.random.shuffle(splits["train"].dat)
    # Delete the splits to force their data to be written to disk.
    print("Flushing splits to disk...")
    del splits


def __main():
    """ This program's entrypoint. """
    # Set the relevant random seeds.
    random.seed(utils.SEED)
    np.random.seed(utils.SEED)

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

    split_prcs = {
        "train": args.train_split, "val": args.val_split,
        "test": args.test_split}
    tot_split = sum(split_prcs.values())
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
    random.shuffle(sim_flns)
    num_sims = args.num_sims
    num_sims = len(sim_flns) if num_sims is None else num_sims
    print(f"Selected {num_sims} simulations")
    sim_flps = [
        path.join(sims_dir, sim_fln) for sim_fln in sim_flns[:num_sims]]
    num_pkts, dtype = __survey(sim_flps)
    fets = dtype.names
    print(f"Total packets: {num_pkts}\nFeatures:\n    " + '\n    '.join(sorted(fets)))

    # Create the merged training, validation, and test files.
    __merge(sim_flps, out_dir, num_pkts, dtype, split_prcs)
    print(f"Finished - time: {time.time() - tim_srt_s:.2f} seconds")
    return 0


if __name__ == "__main__":
    __main()
